import {
  RL_MAX_ARTIFACT_PAGE_SIZE,
  RlEvaluationProtocol,
  RlEvaluationResult,
  type RlStudy,
  type RlStudyExcludedRun,
} from "@t3tools/contracts";
import * as Crypto from "effect/Crypto";
import * as Effect from "effect/Effect";
import * as Encoding from "effect/Encoding";
import * as FileSystem from "effect/FileSystem";
import * as Path from "effect/Path";
import * as Result from "effect/Result";
import * as Schema from "effect/Schema";

import * as Artifacts from "./Artifacts.ts";
import { RunStore } from "./RunStore.ts";
import type { StudyObservation } from "./StudyStatistics.ts";

export const MAX_STUDY_EVALUATION_BYTES = 8 * 1024 * 1024;
const decodeResult = Schema.decodeUnknownEffect(Schema.fromJsonString(RlEvaluationResult));
const isEvaluationProtocol = Schema.is(RlEvaluationProtocol);

/** Reads only completed, hash-verified final evaluation evidence, never training telemetry. */
export const collectStudyObservation = Effect.fn("StudyEvidence.collect")(function* (input: {
  readonly study: RlStudy;
  readonly member: RlStudy["runs"][number];
  readonly metricKey: string;
  readonly rlRunsDir: string;
}) {
  const { member, study } = input;
  const exclude = (reason: RlStudyExcludedRun["reason"]) => ({
    _tag: "Excluded" as const,
    excluded: {
      runId: member.runId,
      trainingSeed: member.seeds.training,
      variantLabel: member.variantLabel,
      reason,
    },
  });
  if (member.state === "failed" || member.state === "cancelled") return exclude(member.state);
  if (member.state !== "completed") return exclude("not-completed");
  if (member.runId === null) return exclude("missing-run");
  const runId = member.runId;
  const store = yield* RunStore;
  const run = yield* store.getRun({ runId }).pipe(
    Effect.catchTag("RlRunNotFoundError", () => Effect.succeed(null)),
    Effect.orDie,
  );
  if (run === null) return exclude("missing-run");
  if (run.summary.state !== "completed") {
    return exclude(
      run.summary.state === "failed" ||
        run.summary.state === "cancelled" ||
        run.summary.state === "interrupted"
        ? run.summary.state
        : "not-completed",
    );
  }
  const protocol = study.definition.evaluationProtocol;
  const manifestProtocol = run.manifest?.effectiveConfig.evaluationProtocol;
  if (
    run.summary.projectId !== study.projectId ||
    run.manifest?.seed !== member.seeds.training ||
    run.manifest.seeds?.data !== member.seeds.data ||
    run.manifest.seeds?.evaluationSample !== member.seeds.evaluationSample ||
    run.manifest.seeds?.generation !== member.seeds.generation ||
    !isEvaluationProtocol(manifestProtocol) ||
    manifestProtocol.protocolSha256 !== study.protocolSha256
  )
    return exclude("incompatible-protocol");

  let cursor: string | undefined;
  let artifactId: string | undefined;
  do {
    const page = yield* store
      .listArtifacts({
        runId,
        limit: RL_MAX_ARTIFACT_PAGE_SIZE,
        ...(cursor === undefined ? {} : { cursor }),
      })
      .pipe(Effect.orDie);
    const matches = page.artifacts.filter(
      (artifact) =>
        artifact.kind === "evaluation" && artifact.logicalName === "study-evaluation.json",
    );
    if (matches.length > 1 || (matches.length > 0 && artifactId !== undefined))
      return exclude("invalid-evaluation");
    artifactId = matches[0]?.artifactId ?? artifactId;
    cursor = page.nextCursor ?? undefined;
  } while (cursor !== undefined);
  if (artifactId === undefined) return exclude("missing-evaluation");
  const source = yield* store.findArtifact({ runId, artifactId }).pipe(Effect.orDie);
  if (
    source === null ||
    source.metadata.state !== "ready" ||
    typeof source.metadata.sha256 !== "string" ||
    source.metadata.bytes > MAX_STUDY_EVALUATION_BYTES ||
    source.metadata.fileCount !== 1
  )
    return exclude("invalid-evaluation");

  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const crypto = yield* Crypto.Crypto;
  const read = yield* Effect.result(
    Effect.gen(function* () {
      const root = Artifacts.runDirectory({ rlRunsDir: input.rlRunsDir, runId });
      const file = Artifacts.resolveArtifactPath({
        rlRunsDir: input.rlRunsDir,
        runId,
        relativePath: source.relativePath,
      });
      if (root === null || file === null) return null;
      const canonical = yield* Effect.all({ root: fs.realPath(root), file: fs.realPath(file) });
      const relative = path.relative(canonical.root, canonical.file);
      if (relative.startsWith("..") || path.isAbsolute(relative) || relative === "") return null;
      const stat = yield* fs.stat(canonical.file);
      if (stat.type !== "File" || Number(stat.size) > MAX_STUDY_EVALUATION_BYTES) return null;
      const bytes = yield* fs.readFile(canonical.file);
      if (
        bytes.byteLength !== source.metadata.bytes ||
        bytes.byteLength > MAX_STUDY_EVALUATION_BYTES
      )
        return null;
      const sha256 = yield* crypto.digest("SHA-256", bytes).pipe(Effect.map(Encoding.encodeHex));
      if (sha256 !== source.metadata.sha256) return null;
      return yield* decodeResult(new TextDecoder().decode(bytes));
    }),
  );
  if (Result.isFailure(read) || read.success === null) return exclude("invalid-evaluation");
  const evaluation = read.success;
  if (evaluation.protocolSha256 !== study.protocolSha256) return exclude("incompatible-protocol");
  const expectedIds = new Set(protocol.sampleIds);
  const actualIds = new Set(evaluation.samples.map((sample) => sample.sampleId));
  if (actualIds.size !== evaluation.samples.length) return exclude("invalid-evaluation");
  if (evaluation.samples.some((sample) => !expectedIds.has(sample.sampleId)))
    return exclude("incompatible-protocol");

  yield* store
    .indexEvaluationSamples({
      runId,
      artifactId,
      artifactSha256: source.metadata.sha256,
      evaluation,
    })
    .pipe(Effect.orDie);
  const indexed = yield* store
    .listEvaluationSamples({
      runId,
      protocolSha256: study.protocolSha256,
      artifactId,
      artifactSha256: source.metadata.sha256,
    })
    .pipe(Effect.orDie);
  const samples: Record<string, number> = Object.create(null);
  const generationSeeds: Record<string, number | null> = Object.create(null);
  const values: number[] = [];
  for (const sample of indexed) {
    const value = sample.values[input.metricKey];
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    samples[sample.sampleId] = value;
    generationSeeds[sample.sampleId] = sample.generationSeed;
    values.push(value);
  }
  if (values.length === 0) return exclude("missing-metric");
  return {
    _tag: "Observation" as const,
    observation: {
      trainingSeed: member.seeds.training,
      value: values.reduce((sum, value) => sum + value / values.length, 0),
      samples,
      generationSeeds,
    } satisfies StudyObservation,
  };
});
