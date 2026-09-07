// @effect-diagnostics nodeBuiltinImport:off - archive work and incremental hashes stay outside RPC payloads.
// @effect-diagnostics preferSchemaOverJson:off - canonical evidence serialization is part of the bundle format.
import * as NodeCrypto from "node:crypto";

import {
  RL_EVIDENCE_MAX_BYTES,
  RL_EVIDENCE_MAX_FILES,
  RL_MAX_ARTIFACT_PAGE_SIZE,
  RL_MAX_SNAPSHOT_METRIC_BATCHES,
  RlEvidenceError,
  RlExportEvidenceInput,
  RlResearchRecord,
  RlResearchRecordInput,
  RlSha256,
  isTerminalRlRunState,
  type RlArtifactMetadata,
  type RlEvidenceExport,
  type RlEvidenceReference,
} from "@t3tools/contracts";
import * as DateTime from "effect/DateTime";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Path from "effect/Path";
import * as Schema from "effect/Schema";
import * as Semaphore from "effect/Semaphore";
import * as Tar from "tar";

import * as ServerConfig from "../config.ts";
import { computeArtifactIdentity } from "./ArtifactIdentity.ts";
import { resolveArtifactPath, runDirectory } from "./Artifacts.ts";
import { evidenceExportPath, resolveEvidenceProjectDirectory } from "./EvidencePaths.ts";
import { RlManager } from "./Manager.ts";
import { RunStore } from "./RunStore.ts";

const MAX_RECORDS = 1024;
const MAX_EXPORTS = 32;
const MAX_PROJECT_EXPORT_BYTES = 1024 * 1024 * 1024;
const MAX_ARCHIVE_BYTES = RL_EVIDENCE_MAX_BYTES + RL_EVIDENCE_MAX_FILES * 2048;
const MAX_JSON_BYTES = 4 * 1024 * 1024;
const mutation = Semaphore.makeUnsafe(1);
const fail = (detail: string) => new RlEvidenceError({ detail: detail.slice(0, 2048) });
const sha256 = (text: string) => NodeCrypto.createHash("sha256").update(text).digest("hex");
const isEvidenceError = Schema.is(RlEvidenceError);
const decodeRecord = Schema.decodeUnknownEffect(Schema.fromJsonString(RlResearchRecord));
const decodeResearchInput = Schema.decodeUnknownEffect(RlResearchRecordInput);
const decodeExportInput = Schema.decodeUnknownEffect(RlExportEvidenceInput);

export function canonicalEvidenceJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalEvidenceJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalEvidenceJson(entry)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

/** Included in every archive; verifies bytes without extracting or importing project code. */
export const EVIDENCE_VERIFIER = `import argparse, hashlib, json, pathlib, tarfile

def verify(archive, expected_index=None):
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        if len(members) > 4098:
            raise ValueError("too many bundle files")
        files = {}
        for entry in members:
            name = pathlib.PurePosixPath(entry.name)
            if not entry.isfile() or name.is_absolute() or ".." in name.parts or "\\\\" in entry.name or entry.name in files:
                raise ValueError("invalid or duplicate bundle entry: " + entry.name)
            files[entry.name] = entry
        if sum(entry.size for entry in members) > 520 * 1024 * 1024:
            raise ValueError("bundle exceeds byte limit")
        if files["index.json"].size > 4 * 1024 * 1024 or files["index.sha256"].size > 128:
            raise ValueError("oversized index")
        raw = bundle.extractfile(files["index.json"]).read()
        digest = hashlib.sha256(raw).hexdigest()
        declared = bundle.extractfile(files["index.sha256"]).read().decode().strip()
        if digest != declared or (expected_index is not None and digest != expected_index):
            raise ValueError("index hash mismatch")
        index = json.loads(raw)
        if index["version"] != 1:
            raise ValueError("unsupported bundle version")
        expected = {entry["path"] for entry in index["files"]}
        if len(expected) != len(index["files"]) or set(files) != expected | {"index.json", "index.sha256"}:
            raise ValueError("missing or unlisted bundle files")
        for entry in index["files"]:
            info = files[entry["path"]]
            if info.size != entry["bytes"]:
                raise ValueError("size mismatch: " + entry["path"])
            hashed = hashlib.sha256()
            with bundle.extractfile(info) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hashed.update(chunk)
            if hashed.hexdigest() != entry["sha256"]:
                raise ValueError("file hash mismatch: " + entry["path"])
        return {"verified": True, "indexSha256": digest, "files": len(expected), "omissions": len(index["omissions"]), "guarantees": index["guarantees"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a T3RL evidence archive independently of its server/database.")
    parser.add_argument("archive")
    parser.add_argument("--index-sha256", help="Expected digest returned by the exporting server")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.archive, args.index_sha256), sort_keys=True))
    except (ValueError, KeyError, OSError, tarfile.TarError, UnicodeError) as error:
        parser.exit(1, "Verification failed: " + str(error) + "\\n")
`;

const scopedRun = Effect.fn("RlEvidence.scopedRun")(function* (projectId: string, runId: string) {
  const manager = yield* RlManager;
  const detail = yield* manager.get({ runId });
  if (detail.summary.projectId !== projectId)
    return yield* fail("Run not found in the selected project.");
  if (!isTerminalRlRunState(detail.summary.state))
    return yield* fail("Evidence requires a terminal run; active results may still change.");
  return detail;
});

const confinedArtifact = Effect.fn("RlEvidence.confinedArtifact")(function* (
  reference: RlEvidenceReference,
) {
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const config = yield* ServerConfig.ServerConfig;
  const store = yield* RunStore;
  if (reference.artifactId === null) return yield* fail("An artifact ID is required.");
  const artifact = yield* store.findArtifact({
    runId: reference.runId,
    artifactId: reference.artifactId,
  });
  if (
    artifact === null ||
    artifact.metadata.state !== "ready" ||
    artifact.metadata.sha256 == null
  ) {
    return yield* fail("Selected evidence has no verified, ready artifact.");
  }
  const root = runDirectory({ rlRunsDir: config.rlRunsDir, runId: reference.runId });
  const target = resolveArtifactPath({
    rlRunsDir: config.rlRunsDir,
    runId: reference.runId,
    relativePath: artifact.relativePath,
  });
  if (root === null || target === null) return yield* fail("Invalid evidence path.");
  const canonicalRoot = yield* fs.realPath(root);
  const canonical = yield* fs.realPath(target);
  const canonicalRuns = yield* fs.realPath(config.rlRunsDir);
  if (
    canonicalRoot !== path.join(canonicalRuns, reference.runId) ||
    canonical !== path.join(canonicalRoot, artifact.relativePath) ||
    !canonical.startsWith(`${canonicalRoot}${path.sep}`)
  )
    return yield* fail("Evidence path escaped its run or contains a symlink.");
  const identity = yield* computeArtifactIdentity({
    artifactPath: target,
    maxBytes: RL_EVIDENCE_MAX_BYTES,
  });
  if (identity.sha256 !== artifact.metadata.sha256 || identity.bytes !== artifact.metadata.bytes)
    return yield* fail("Artifact integrity verification failed.");
  return { ...artifact, path: target, identity };
});

const failure = (error: unknown) =>
  isEvidenceError(error)
    ? error
    : fail(error instanceof Error ? error.message : "Evidence operation failed.");

const readRecord = Effect.fn("RlEvidence.readRecord")(function* (input: {
  readonly projectId: string;
  readonly recordId: string;
}) {
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const config = yield* ServerConfig.ServerConfig;
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(input.recordId))
    return yield* fail("Invalid research record ID.");
  const file = path.join(
    yield* resolveEvidenceProjectDirectory(config.rlRunsDir, input.projectId),
    "records",
    `${input.recordId}.json`,
  );
  if ((yield* fs.realPath(file)) !== file || Number((yield* fs.stat(file)).size) > MAX_JSON_BYTES)
    return yield* fail("Invalid research record file.");
  const record = yield* decodeRecord(yield* fs.readFileString(file));
  const { sha256: expected, ...body } = record;
  if (
    record.input.projectId !== input.projectId ||
    record.recordId !== input.recordId ||
    sha256(canonicalEvidenceJson(body)) !== expected
  )
    return yield* fail("Research record integrity verification failed.");
  return record;
});

export const getResearchRecord = Effect.fn("RlEvidence.getResearchRecord")(
  readRecord,
  Effect.mapError(failure),
);

export const recordResearch = Effect.fn("RlEvidence.recordResearch")(
  function* (rawInput: RlResearchRecordInput) {
    const input = yield* decodeResearchInput(rawInput);
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const config = yield* ServerConfig.ServerConfig;
    const directory = path.join(
      yield* resolveEvidenceProjectDirectory(config.rlRunsDir, input.projectId),
      "records",
    );
    yield* fs.makeDirectory(directory, { recursive: true });
    if ((yield* fs.realPath(directory)) !== directory)
      return yield* fail("Research storage contains a symlink.");
    const recordId = input.requestId;
    const target = path.join(directory, `${recordId}.json`);
    if (yield* fs.exists(target)) {
      const existing = yield* readRecord({ projectId: input.projectId, recordId });
      if (canonicalEvidenceJson(existing.input) !== canonicalEvidenceJson(input))
        return yield* fail("Research request ID already belongs to different content.");
      return existing;
    }
    if ((yield* fs.readDirectory(directory)).length >= MAX_RECORDS)
      return yield* fail("Project research record limit reached.");
    if (input.parentRecordId !== null)
      yield* readRecord({ projectId: input.projectId, recordId: input.parentRecordId });
    const evidence = yield* Effect.forEach(input.evidence, (reference) =>
      Effect.gen(function* () {
        const run = yield* scopedRun(input.projectId, reference.runId);
        const digest =
          reference.artifactId === null
            ? sha256(
                canonicalEvidenceJson({
                  summary: run.summary,
                  manifest: run.manifest,
                  lineage: run.lineage,
                }),
              )
            : (yield* confinedArtifact(reference)).identity.sha256;
        return {
          ...reference,
          sha256: digest,
          sourceRevision: run.manifest?.sourceRevision ?? null,
          sourceDirty: run.manifest?.sourceDirty ?? null,
        };
      }),
    );
    const body = {
      version: 1 as const,
      recordId,
      recordedAt: DateTime.formatIso(yield* DateTime.now),
      input,
      evidence,
    };
    const record = { ...body, sha256: sha256(canonicalEvidenceJson(body)) };
    const temporary = yield* fs.makeTempDirectoryScoped({ directory, prefix: ".record-" });
    yield* fs.writeFileString(
      path.join(temporary, "record.json"),
      `${JSON.stringify(record, null, 2)}\n`,
    );
    yield* fs.rename(path.join(temporary, "record.json"), target);
    return record;
  },
  Effect.scoped,
  (effect) => mutation.withPermit(effect),
  Effect.mapError(failure),
);

const ResolvedInputs = Schema.Array(
  Schema.Struct({
    role: Schema.String,
    logicalName: Schema.String.check(Schema.isPattern(/^[^/\\]+$/)),
    sha256: RlSha256,
    bytes: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  }),
).check(Schema.isMaxLength(16));
const decodeResolvedInputs = Schema.decodeUnknownEffect(ResolvedInputs);

export const exportEvidence = Effect.fn("RlEvidence.exportEvidence")(
  function* (rawInput: RlExportEvidenceInput) {
    const input = yield* decodeExportInput(rawInput);
    if (new Set(input.runIds).size !== input.runIds.length)
      return yield* fail("Select distinct run IDs.");
    if (
      input.additionalArtifacts.some(
        (reference) => !input.runIds.includes(reference.runId) || reference.artifactId === null,
      )
    )
      return yield* fail("Additional artifacts must name an artifact in a selected run.");
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const config = yield* ServerConfig.ServerConfig;
    const manager = yield* RlManager;
    const runs = yield* Effect.forEach([...input.runIds].sort(), (runId) =>
      scopedRun(input.projectId, runId),
    );
    const directory = path.join(
      yield* resolveEvidenceProjectDirectory(config.rlRunsDir, input.projectId),
      "exports",
    );
    yield* fs.makeDirectory(directory, { recursive: true });
    if ((yield* fs.realPath(directory)) !== directory)
      return yield* fail("Evidence storage contains a symlink.");
    const temporary = yield* fs.makeTempDirectoryScoped({ directory, prefix: ".export-" });
    const staging = path.join(temporary, "bundle");
    yield* fs.makeDirectory(staging);
    const files: Array<{ path: string; bytes: number; sha256: string }> = [];
    const omissions: Array<{ runId: string | null; artifactId: string | null; reason: string }> =
      [];
    const includedEvidence = new Map<string, string>();
    let totalBytes = 0;
    const account = Effect.fn("RlEvidence.account")(function* (
      name: string,
      bytes: number,
      digest: string,
    ) {
      totalBytes += bytes;
      if (totalBytes > RL_EVIDENCE_MAX_BYTES || files.length >= RL_EVIDENCE_MAX_FILES - 2)
        return yield* fail(
          "Evidence bundle exceeds its byte or file budget; select fewer artifacts/runs.",
        );
      files.push({ path: name, bytes, sha256: digest });
    });
    const addText = Effect.fn("RlEvidence.addText")(function* (name: string, text: string) {
      if (Buffer.byteLength(text) > MAX_JSON_BYTES)
        return yield* fail("Evidence metadata exceeds its byte budget.");
      yield* account(name, Buffer.byteLength(text), sha256(text));
      const destination = path.join(staging, name);
      yield* fs.makeDirectory(path.dirname(destination), { recursive: true });
      yield* fs.writeFileString(destination, text);
    });
    const addCopy = Effect.fn("RlEvidence.addCopy")(function* (
      source: string,
      name: string,
      expected: string,
    ) {
      const identity = yield* computeArtifactIdentity({
        artifactPath: source,
        maxBytes: RL_EVIDENCE_MAX_BYTES - totalBytes,
      });
      if (identity.sha256 !== expected)
        return yield* fail("Input evidence changed since the run was recorded.");
      const entries = identity.directory
        ? identity.contentManifest
        : [{ path: "", bytes: identity.bytes, sha256: identity.sha256 }];
      for (const entry of entries) {
        const relative = entry.path === "" ? name : `${name}/${entry.path}`;
        const destination = path.join(staging, relative);
        yield* account(relative, entry.bytes, entry.sha256);
        yield* fs.makeDirectory(path.dirname(destination), { recursive: true });
        yield* fs.copyFile(entry.path === "" ? source : path.join(source, entry.path), destination);
        const copied = yield* computeArtifactIdentity({
          artifactPath: destination,
          maxBytes: entry.bytes,
        });
        if (copied.sha256 !== entry.sha256) return yield* fail("Evidence changed during export.");
      }
    });
    yield* addText("verify_evidence.py", EVIDENCE_VERIFIER);
    for (const detail of runs) {
      const runId = detail.summary.runId;
      includedEvidence.set(
        runId,
        sha256(
          canonicalEvidenceJson({
            summary: detail.summary,
            manifest: detail.manifest,
            lineage: detail.lineage,
          }),
        ),
      );
      yield* addText(
        `runs/${runId}/run.json`,
        `${JSON.stringify({ ...detail, artifacts: undefined, metrics: undefined }, null, 2)}\n`,
      );
      yield* addText(
        `runs/${runId}/metrics.json`,
        `${JSON.stringify({ selection: "bounded-reconnect-snapshot", limit: RL_MAX_SNAPSHOT_METRIC_BATCHES, batches: detail.metrics }, null, 2)}\n`,
      );
      const inventory: RlArtifactMetadata[] = [];
      let cursor: string | undefined;
      do {
        const page = yield* manager.listArtifacts({
          runId,
          limit: RL_MAX_ARTIFACT_PAGE_SIZE,
          ...(cursor === undefined ? {} : { cursor }),
        });
        inventory.push(...page.artifacts);
        if (inventory.length > RL_EVIDENCE_MAX_FILES)
          return yield* fail("Artifact inventory exceeds the bundle limit.");
        cursor = page.nextCursor ?? undefined;
      } while (cursor !== undefined);
      yield* addText(`runs/${runId}/artifacts.json`, `${JSON.stringify(inventory, null, 2)}\n`);
      for (const artifact of inventory) {
        const explicit = input.additionalArtifacts.some(
          (reference) => reference.runId === runId && reference.artifactId === artifact.artifactId,
        );
        const selected =
          explicit ||
          ["evaluation", "summary", "config", "dataset", "source", "adapter"].includes(
            artifact.kind,
          );
        if (!selected || artifact.state !== "ready" || artifact.sha256 == null) {
          if (explicit)
            return yield* fail("An explicitly selected artifact is unavailable or unverified.");
          omissions.push({
            runId,
            artifactId: artifact.artifactId,
            reason: !selected ? "not-selected" : "unavailable-or-unverified",
          });
          continue;
        }
        const verified = yield* confinedArtifact({ runId, artifactId: artifact.artifactId });
        yield* addCopy(
          verified.path,
          `runs/${runId}/artifacts/${artifact.artifactId}/${path.basename(verified.path)}`,
          verified.identity.sha256,
        );
        includedEvidence.set(`${runId}/${artifact.artifactId}`, verified.identity.sha256);
      }
      for (const reference of input.additionalArtifacts.filter((entry) => entry.runId === runId)) {
        if (!inventory.some((entry) => entry.artifactId === reference.artifactId))
          return yield* fail("An explicitly selected artifact was not found.");
      }
      const manifest = detail.manifest;
      if (manifest?.effectiveConfig.resolvedProjectInputs !== undefined) {
        const inputs = yield* decodeResolvedInputs(manifest.effectiveConfig.resolvedProjectInputs);
        for (const item of inputs) {
          const source = resolveArtifactPath({
            rlRunsDir: yield* fs.realPath(config.rlRunsDir),
            runId,
            relativePath: `inputs/project/${item.logicalName}`,
          });
          if (source === null || (yield* fs.realPath(source)) !== source)
            return yield* fail("Project input escaped its snapshot.");
          yield* addCopy(source, `runs/${runId}/inputs/${item.logicalName}`, item.sha256);
        }
      } else
        omissions.push({ runId, artifactId: null, reason: "project-input-snapshot-not-recorded" });
      if (manifest?.environmentLock != null) {
        const lock = manifest.environmentLock;
        if (lock.files !== undefined && lock.files.length === 2) {
          for (const file of lock.files) {
            const source = resolveArtifactPath({
              rlRunsDir: yield* fs.realPath(config.rlRunsDir),
              runId,
              relativePath: `inputs/environment/${file.name}`,
            });
            if (source === null || (yield* fs.realPath(source)) !== source)
              return yield* fail("Environment input escaped its snapshot.");
            yield* addCopy(source, `runs/${runId}/environment/${file.name}`, file.sha256);
          }
        } else {
          omissions.push({
            runId,
            artifactId: null,
            reason: "environment-project-file-not-snapshotted",
          });
          const current = yield* computeArtifactIdentity({
            artifactPath: lock.lockfilePath,
            maxBytes: 4 * 1024 * 1024,
          }).pipe(Effect.orElseSucceed(() => null));
          if (current?.sha256 === lock.lockfileSha256)
            yield* addCopy(
              lock.lockfilePath,
              `runs/${runId}/environment/uv.lock`,
              lock.lockfileSha256,
            );
          else
            omissions.push({
              runId,
              artifactId: null,
              reason: "original-environment-lock-bytes-unavailable",
            });
        }
      } else omissions.push({ runId, artifactId: null, reason: "environment-lock-not-recorded" });
      if (
        !inventory.some(
          (artifact) =>
            artifact.kind === "source" && artifact.state === "ready" && artifact.sha256 != null,
        )
      )
        omissions.push({
          runId,
          artifactId: null,
          reason: "exact-workspace-source-not-snapshotted",
        });
    }
    if (input.study !== null) {
      const study = yield* manager.getStudy({ studyId: input.study.studyId });
      if (study.projectId !== input.projectId)
        return yield* fail("Study not found in the selected project.");
      const comparison = yield* manager.compareStudy(input.study);
      yield* addText("study.json", `${JSON.stringify({ study, comparison }, null, 2)}\n`);
      for (const run of study.runs)
        if (run.runId !== null && !input.runIds.includes(run.runId))
          omissions.push({
            runId: run.runId,
            artifactId: null,
            reason: "study-member-not-selected",
          });
    }
    const recordIds = [...new Set(input.recordIds)].sort();
    for (let position = 0; position < recordIds.length; position += 1) {
      const recordId = recordIds[position]!;
      const record = yield* readRecord({ projectId: input.projectId, recordId });
      if (
        record.input.parentRecordId !== null &&
        !recordIds.includes(record.input.parentRecordId)
      ) {
        if (recordIds.length >= 16)
          return yield* fail("Research ancestry exceeds the sixteen-record bundle limit.");
        recordIds.push(record.input.parentRecordId);
      }
      yield* addText(`research/${recordId}.json`, `${JSON.stringify(record, null, 2)}\n`);
      for (const reference of record.evidence) {
        const digest = includedEvidence.get(
          reference.artifactId === null
            ? reference.runId
            : `${reference.runId}/${reference.artifactId}`,
        );
        if (digest !== undefined && digest !== reference.sha256)
          return yield* fail("Research evidence no longer matches its recorded identity.");
        if (digest === undefined)
          omissions.push({
            runId: reference.runId,
            artifactId: reference.artifactId,
            reason: input.runIds.includes(reference.runId)
              ? "research-reference-artifact-not-selected"
              : "research-reference-not-selected",
          });
      }
    }
    const guarantees = {
      integrity: "all-included-files-verified",
      environment: "recorded-evidence-only; inspect omissions and platform requirements",
      trainerResume: "requires an explicitly included complete compatible checkpoint",
      numericalReproducibility: "not-established-by-export",
      scientificConclusion: "requires independent held-out evaluation and adequate replication",
    };
    const index = `${JSON.stringify({ version: 1, projectId: input.projectId, runIds: [...input.runIds].sort(), guarantees, omissions, files: files.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0)) }, null, 2)}\n`;
    if (Buffer.byteLength(index) > MAX_JSON_BYTES)
      return yield* fail("Evidence index exceeds its byte budget.");
    const indexSha256 = sha256(index);
    const exportId = `export_${indexSha256.slice(0, 48)}`;
    const archivePath = evidenceExportPath({
      rlRunsDir: config.rlRunsDir,
      projectId: input.projectId,
      exportId,
    });
    if (archivePath === null) return yield* fail("Invalid evidence export ID.");
    yield* fs.writeFileString(path.join(staging, "index.json"), index);
    yield* fs.writeFileString(path.join(staging, "index.sha256"), `${indexSha256}\n`);
    const archives = (yield* fs.readDirectory(directory)).filter((entry) => entry.endsWith(".tar"));
    let existingBytes = 0;
    for (const entry of archives)
      existingBytes += Number((yield* fs.stat(path.join(directory, entry))).size);
    if (
      !(yield* fs.exists(archivePath)) &&
      (archives.length >= MAX_EXPORTS || existingBytes + totalBytes > MAX_PROJECT_EXPORT_BYTES)
    )
      return yield* fail(
        "Project export quota reached; retain a copy before explicitly removing old exports.",
      );
    const temporaryArchive = path.join(temporary, "bundle.tar");
    yield* Effect.tryPromise({
      try: () =>
        Tar.create(
          { cwd: staging, file: temporaryArchive, portable: true, noMtime: true, strict: true },
          [...files.map((entry) => entry.path), "index.json", "index.sha256"],
        ),
      catch: failure,
    }).pipe(Effect.uninterruptible);
    const archive = yield* computeArtifactIdentity({
      artifactPath: temporaryArchive,
      maxBytes: MAX_ARCHIVE_BYTES,
    });
    if (
      existingBytes + archive.bytes > MAX_PROJECT_EXPORT_BYTES &&
      !(yield* fs.exists(archivePath))
    )
      return yield* fail("Project export byte quota reached.");
    yield* fs.rename(temporaryArchive, archivePath);
    return {
      version: 1,
      exportId,
      projectId: input.projectId,
      sha256: archive.sha256,
      indexSha256,
      bytes: archive.bytes,
      fileCount: files.length,
      omittedCount: omissions.length,
    } satisfies RlEvidenceExport;
  },
  Effect.scoped,
  (effect) => mutation.withPermit(effect),
  Effect.mapError(failure),
);
