// @effect-diagnostics nodeBuiltinImport:off - independent archive verification uses the system Python standard library.
// @effect-diagnostics preferSchemaOverJson:off - independent reader and corruption fixtures deliberately do not reuse writer codecs.
import * as NodeChildProcess from "node:child_process";
import * as NodeCrypto from "node:crypto";
import * as NodeUtil from "node:util";

import * as NodeServices from "@effect/platform-node/NodeServices";
import { expect, it } from "@effect/vitest";
import {
  type RlArtifactMetadata,
  type RlResearchRecordInput,
  type RlRunSummary,
} from "@t3tools/contracts";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Path from "effect/Path";
import * as Tar from "tar";

import * as ServerConfig from "../config.ts";
import {
  EVIDENCE_VERIFIER,
  exportEvidence,
  getResearchRecord,
  recordResearch,
} from "./EvidenceBundle.ts";
import { evidenceExportPath, evidenceProjectDirectory } from "./EvidencePaths.ts";
import { RlManager, type RlRunDetail } from "./Manager.ts";
import { RunStore } from "./RunStore.ts";

const digest = (value: string) => NodeCrypto.createHash("sha256").update(value).digest("hex");
const exec = NodeUtil.promisify(NodeChildProcess.execFile);
const summary: RlRunSummary = {
  runId: "run-evidence",
  projectId: "project-evidence",
  experimentId: "tiny-sft",
  state: "completed",
  requestedAt: "2026-09-05T01:00:00Z",
  startedAt: "2026-09-05T01:00:01Z",
  endedAt: "2026-09-05T01:00:02Z",
  lastMessageAt: "2026-09-05T01:00:02Z",
  errorCode: null,
  errorMessage: null,
};
const content = '{"phase":"after","loss":0.4}\n';
const artifact: RlArtifactMetadata = {
  artifactId: "evaluation-1",
  kind: "evaluation",
  bytes: Buffer.byteLength(content),
  contentType: "application/json",
  producedAt: summary.endedAt!,
  sha256: digest(content),
  state: "ready",
  logicalName: "evaluation.json",
  format: "json",
  fileCount: 1,
};
const request: RlResearchRecordInput = {
  projectId: summary.projectId,
  requestId: "research-baseline",
  parentRecordId: null,
  hypothesis: "Updating adapter parameters lowers held-out loss.",
  evidence: [{ runId: summary.runId, artifactId: artifact.artifactId }],
  proposedChange:
    "Compare the declared learning rate with zero learning rate using the same evaluation.",
  authorizationReference: "Researcher-approved bounded three-seed comparison.",
  outcome: "inconclusive",
  interpretation:
    "A small fixture checks execution and evidence; it does not establish generalization.",
  limitations: "Synthetic data and tiny model.",
};

const fixture = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-evidence-" });
  const runs = path.join(root, "rl");
  const runRoot = path.join(runs, summary.runId);
  yield* fs.makeDirectory(runRoot, { recursive: true });
  yield* fs.writeFileString(path.join(runRoot, "evaluation.json"), content);
  yield* fs.writeFileString(path.join(root, ".env"), "UNRELATED_SECRET=not-exported");
  const detail: RlRunDetail = {
    summary,
    manifest: null,
    lineage: { edges: [], truncated: false },
    artifacts: [artifact],
    metrics: [{ step: 1, wallClockMs: 4, values: { "eval/loss": 0.4 } }],
  };
  const manager = RlManager.of({
    capabilities: () => Effect.die("unused"),
    start: () => Effect.die("unused"),
    resume: () => Effect.die("unused"),
    warmStart: () => Effect.die("unused"),
    cancel: () => Effect.die("unused"),
    list: () => Effect.die("unused"),
    get: () => Effect.succeed(detail),
    listArtifacts: () => Effect.succeed({ artifacts: [artifact], nextCursor: null }),
    subscribe: () => Effect.die("unused"),
    sweepInterruptedRuns: () => Effect.die("unused"),
    createStudy: () => Effect.die("unused"),
    getStudy: () => Effect.die("unused"),
    compareStudy: () => Effect.die("unused"),
    validateExperiment: () => Effect.die("unused"),
  });
  const store = {
    findArtifact: () =>
      Effect.succeed({
        metadata: artifact,
        relativePath: "evaluation.json",
        contentManifest: null,
      }),
  } as unknown as RunStore["Service"];
  const provide = <A, E, R>(effect: Effect.Effect<A, E, R>) =>
    effect.pipe(
      Effect.provideService(RlManager, manager),
      Effect.provideService(RunStore, store),
      Effect.provide(
        ServerConfig.layer({ rlRunsDir: runs } as ServerConfig.ServerConfig["Service"]),
      ),
    );
  const input = {
    projectId: summary.projectId,
    runIds: [summary.runId],
    study: null,
    recordIds: [],
    additionalArtifacts: [],
  };
  return { fs, path, root, runs, runRoot, detail, manager, provide, input };
});

it.effect(
  "exports verified evidence and immutable research records that verify without the database",
  () =>
    Effect.gen(function* () {
      const f = yield* fixture;
      const record = yield* f.provide(recordResearch(request));
      expect(yield* f.provide(recordResearch(request))).toEqual(record);
      const child = yield* f.provide(
        recordResearch({
          ...request,
          requestId: "research-result",
          parentRecordId: record.recordId,
          outcome: "rejected",
        }),
      );
      expect(child.input.parentRecordId).toBe(record.recordId);
      expect(
        yield* f.provide(
          getResearchRecord({ projectId: request.projectId, recordId: record.recordId }),
        ),
      ).toEqual(record);
      const input = { ...f.input, recordIds: [child.recordId] };
      const exported = yield* f.provide(exportEvidence(input));
      const again = yield* f.provide(exportEvidence(input));
      expect(again).toEqual(exported);
      expect(exported.omittedCount).toBeGreaterThan(0);
      expect(yield* f.fs.readFileString(f.path.join(f.runRoot, "evaluation.json"))).toBe(content);
      const archive = evidenceExportPath({
        rlRunsDir: f.runs,
        projectId: request.projectId,
        exportId: exported.exportId,
      })!;
      const verifier = f.path.join(f.root, "verify_evidence.py");
      yield* f.fs.writeFileString(verifier, EVIDENCE_VERIFIER);
      // Remove the source, proving that verification does not consult the originating environment.
      yield* f.fs.remove(f.runs, { recursive: true });
      const verified = yield* Effect.tryPromise(() =>
        exec("python3", [verifier, archive, "--index-sha256", exported.indexSha256]),
      );
      expect(JSON.parse(verified.stdout)).toMatchObject({
        verified: true,
        files: exported.fileCount,
        omissions: exported.omittedCount,
      });
      const names: string[] = [];
      yield* Effect.tryPromise(() =>
        Tar.list({
          file: archive,
          onReadEntry: (entry) => {
            names.push(entry.path);
          },
        }),
      );
      expect(names).not.toContain(".env");
      expect(names).toContain(`research/${record.recordId}.json`);
      expect(names).toContain(
        `runs/${summary.runId}/artifacts/${artifact.artifactId}/evaluation.json`,
      );
    }).pipe(Effect.provide(NodeServices.layer), Effect.scoped),
);

it.effect(
  "rejects corrupt bytes, foreign references, active runs, and changed idempotency content",
  () =>
    Effect.gen(function* () {
      const f = yield* fixture;
      const record = yield* f.provide(recordResearch(request));
      const changed = yield* f
        .provide(recordResearch({ ...request, hypothesis: "Changed after seeing the result." }))
        .pipe(Effect.flip);
      expect(changed.detail).toContain("different content");
      const foreign = yield* f
        .provide(exportEvidence({ ...f.input, projectId: "foreign-project" }))
        .pipe(Effect.flip);
      expect(foreign.detail).toContain("selected project");
      const active = {
        ...f.manager,
        get: () =>
          Effect.succeed({ ...f.detail, summary: { ...summary, state: "running" as const } }),
      };
      const activeError = yield* f
        .provide(exportEvidence(f.input).pipe(Effect.provideService(RlManager, active)))
        .pipe(Effect.flip);
      expect(activeError.detail).toContain("terminal run");
      yield* f.fs.writeFileString(
        f.path.join(f.runRoot, "evaluation.json"),
        content.replace("0.4", "0.9"),
      );
      const corrupted = yield* f.provide(exportEvidence(f.input)).pipe(Effect.flip);
      expect(corrupted.detail).toContain("integrity");
      const recordFile = f.path.join(
        evidenceProjectDirectory(f.runs, request.projectId),
        "records",
        `${record.recordId}.json`,
      );
      yield* f.fs.writeFileString(
        recordFile,
        JSON.stringify({ ...record, input: { ...record.input, outcome: "supported" } }),
      );
      const tampered = yield* f
        .provide(getResearchRecord({ projectId: request.projectId, recordId: record.recordId }))
        .pipe(Effect.flip);
      expect(tampered.detail).toContain("integrity");
    }).pipe(Effect.provide(NodeServices.layer), Effect.scoped),
);

it.effect("standalone verifier detects changed, missing, and unlisted archive files", () =>
  Effect.gen(function* () {
    const f = yield* fixture;
    const result = yield* f.provide(exportEvidence(f.input));
    const archive = evidenceExportPath({
      rlRunsDir: f.runs,
      projectId: request.projectId,
      exportId: result.exportId,
    })!;
    const unpacked = f.path.join(f.root, "unpacked");
    yield* f.fs.makeDirectory(unpacked);
    yield* Effect.tryPromise(() => Tar.extract({ file: archive, cwd: unpacked }));
    const verifier = f.path.join(f.root, "verify_evidence.py");
    yield* f.fs.writeFileString(verifier, EVIDENCE_VERIFIER);
    const index = JSON.parse(yield* f.fs.readFileString(f.path.join(unpacked, "index.json"))) as {
      files: Array<{ path: string }>;
    };
    const names = [...index.files.map((entry) => entry.path), "index.json", "index.sha256"];
    const target = `runs/${summary.runId}/artifacts/${artifact.artifactId}/evaluation.json`;
    const corruptArchive = f.path.join(f.root, "corrupt.tar");
    yield* f.fs.writeFileString(f.path.join(unpacked, target), content.replace("0.4", "0.9"));
    yield* Effect.tryPromise(() => Tar.create({ file: corruptArchive, cwd: unpacked }, names));
    const corrupted = yield* Effect.tryPromise(() =>
      exec("python3", [verifier, corruptArchive]),
    ).pipe(Effect.flip);
    expect(String(corrupted.cause)).toContain("file hash mismatch");
    yield* f.fs.writeFileString(f.path.join(unpacked, target), content);
    yield* Effect.tryPromise(() =>
      Tar.create(
        { file: corruptArchive, cwd: unpacked },
        names.filter((entry) => entry !== target),
      ),
    );
    const missing = yield* Effect.tryPromise(() =>
      exec("python3", [verifier, corruptArchive]),
    ).pipe(Effect.flip);
    expect(String(missing.cause)).toContain("missing or unlisted");
    yield* f.fs.writeFileString(f.path.join(unpacked, "extra.txt"), "not indexed");
    yield* Effect.tryPromise(() =>
      Tar.create({ file: corruptArchive, cwd: unpacked }, [...names, "extra.txt"]),
    );
    const extra = yield* Effect.tryPromise(() => exec("python3", [verifier, corruptArchive])).pipe(
      Effect.flip,
    );
    expect(String(extra.cause)).toContain("missing or unlisted");
  }).pipe(Effect.provide(NodeServices.layer), Effect.scoped),
);

it.effect("allows configured parent aliases while retaining canonical research storage", () =>
  Effect.gen(function* () {
    const f = yield* fixture;
    const aliasRoot = yield* f.fs.makeTempDirectoryScoped({ prefix: "t3rl-evidence-alias-" });
    const alias = f.path.join(aliasRoot, "state");
    yield* f.fs.symlink(f.root, alias);
    const aliasConfig = {
      rlRunsDir: f.path.join(alias, "rl"),
    } as ServerConfig.ServerConfig["Service"];
    const provide = <A, E, R>(effect: Effect.Effect<A, E, R>) =>
      f.provide(effect.pipe(Effect.provideService(ServerConfig.ServerConfig, aliasConfig)));
    const record = yield* provide(recordResearch(request));
    expect(
      yield* provide(
        getResearchRecord({ projectId: request.projectId, recordId: record.recordId }),
      ),
    ).toEqual(record);
    expect(
      (yield* provide(exportEvidence({ ...f.input, recordIds: [record.recordId] }))).fileCount,
    ).toBeGreaterThan(0);
  }).pipe(Effect.provide(NodeServices.layer), Effect.scoped),
);

it.effect("declares research references to artifacts omitted from a selected run", () =>
  Effect.gen(function* () {
    const f = yield* fixture;
    const record = yield* f.provide(recordResearch(request));
    const logManager = {
      ...f.manager,
      listArtifacts: () =>
        Effect.succeed({ artifacts: [{ ...artifact, kind: "log" as const }], nextCursor: null }),
    };
    const exported = yield* f.provide(
      exportEvidence({ ...f.input, recordIds: [record.recordId] }).pipe(
        Effect.provideService(RlManager, logManager),
      ),
    );
    const archive = evidenceExportPath({
      rlRunsDir: f.runs,
      projectId: request.projectId,
      exportId: exported.exportId,
    })!;
    const unpacked = f.path.join(f.root, "omission-check");
    yield* f.fs.makeDirectory(unpacked);
    yield* Effect.tryPromise(() => Tar.extract({ file: archive, cwd: unpacked }, ["index.json"]));
    expect(
      JSON.parse(yield* f.fs.readFileString(f.path.join(unpacked, "index.json"))).omissions,
    ).toContainEqual({
      runId: summary.runId,
      artifactId: artifact.artifactId,
      reason: "research-reference-artifact-not-selected",
    });
  }).pipe(Effect.provide(NodeServices.layer), Effect.scoped),
);
