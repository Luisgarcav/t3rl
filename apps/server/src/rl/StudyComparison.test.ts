// @effect-diagnostics preferSchemaOverJson:off - fixture artifact bytes intentionally model worker output.
import type { RlEvaluationResult, RlStudy, RlStudyEstimator } from "@t3tools/contracts";
import * as NodeCrypto from "node:crypto";
import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Path from "effect/Path";

import * as ServerConfig from "../config.ts";
import { SqlitePersistenceMemory } from "../persistence/Layers/Sqlite.ts";
import { computeArtifactIdentity } from "./ArtifactIdentity.ts";
import { Capabilities } from "./Capabilities.ts";
import * as Experiments from "./Experiments.ts";
import { RlManager, RlManagerLive } from "./Manager.ts";
import { RunStore, RunStoreLive } from "./RunStore.ts";
import * as SourceEvidence from "./SourceEvidence.ts";
import { FakeWorkerProcess, fakeWorkerSpawnerLayer } from "./testing/FakeWorkerSpawner.ts";

const at = "2026-09-05T00:00:00.000Z";
const protocol = {
  version: 1 as const,
  protocolSha256: "a".repeat(64),
  datasetFingerprint: "dataset-v1",
  split: "test",
  sampleIds: ["a", "b"],
  generationSeedPolicy: "fixed-per-sample" as const,
  decoding: {},
  verifierSha256: "b".repeat(64),
};
const protocolBody = Object.fromEntries(
  Object.entries(protocol)
    .filter(([key]) => key !== "protocolSha256")
    .sort(([left], [right]) => left.localeCompare(right)),
);
protocol.protocolSha256 = NodeCrypto.createHash("sha256")
  .update(JSON.stringify(protocolBody))
  .digest("hex");
const estimator: RlStudyEstimator = {
  version: 2,
  statistic: "paired-mean-delta",
  statisticalUnit: "paired-sample-within-run-seed",
  confidenceLevel: 0.95,
  resamplingSeed: 17,
  resampleCount: 1_000,
  missingPairPolicy: "exclude",
};
const input = {
  studyId: "study_fixture",
  baselineLabel: "baseline",
  candidateLabel: "candidate",
  metricKey: "eval_after/loss",
  estimator,
};

const configLayer = Layer.effect(
  ServerConfig.ServerConfig,
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const stateDir = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-study-test-" });
    return ServerConfig.make({ baseDir: stateDir, rlRunsDir: `${stateDir}/rl` } as never);
  }),
);
const layer = RlManagerLive.pipe(
  Layer.provide(fakeWorkerSpawnerLayer(new FakeWorkerProcess())),
  Layer.provide(
    Layer.succeed(
      Capabilities,
      Capabilities.of({
        report: () => Effect.succeed({ runners: [], experiments: [] }),
        resolvePython: () => Effect.die("comparison must not start a worker"),
        resolveRunner: () => Effect.die("comparison must not start a worker"),
      }),
    ),
  ),
  Layer.provide(Experiments.layerFromRecord({})),
  Layer.provide(
    SourceEvidence.layerFromResolver(() => Effect.die("comparison must not read a project")),
  ),
  Layer.provideMerge(configLayer),
  Layer.provideMerge(RunStoreLive),
  Layer.provideMerge(SqlitePersistenceMemory),
  Layer.provideMerge(NodeServices.layer),
);

const seedStudy = Effect.fn("StudyComparisonTest.seed")(function* (
  options: {
    readonly legacyRun?: string;
    readonly nonterminalRun?: string;
    readonly failedRun?: string;
    readonly evaluation?: (runId: string, evaluation: RlEvaluationResult) => RlEvaluationResult;
  } = {},
) {
  const store = yield* RunStore;
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const config = yield* ServerConfig.ServerConfig;
  const seeds = [1, 2].map((training) => ({
    training,
    data: 3,
    evaluationSample: 4,
    generation: 5,
  }));
  const variants = [
    { label: "baseline", experimentId: "fake" },
    { label: "candidate", experimentId: "fake" },
  ];
  const study: RlStudy = {
    studyId: input.studyId,
    projectId: "project_fixture",
    state: "completed",
    definition: { variants, seeds, maxConcurrency: 1, maxRuns: 4, evaluationProtocol: protocol },
    protocolSha256: protocol.protocolSha256,
    createdAt: at,
    updatedAt: at,
    runs: seeds.flatMap((seed) =>
      variants.map((variant) => ({
        variantLabel: variant.label,
        seeds: seed,
        runId: `${variant.label}_${seed.training}`,
        state: options.failedRun === `${variant.label}_${seed.training}` ? "failed" : "completed",
      })),
    ),
  };
  yield* store.createStudy(study);
  for (const member of study.runs) {
    const runId = member.runId!;
    yield* store.insertRequested({
      runId,
      projectId: study.projectId,
      experimentId: "fake",
      requestedAt: at,
    });
    yield* store.setManifest({
      runId,
      manifest: {
        experimentId: "fake",
        runnerId: "fake",
        runnerVersion: "0.1.0",
        protocolVersion: 1,
        seed: member.seeds.training,
        seeds: member.seeds,
        effectiveConfig: { evaluationProtocol: protocol },
        sourceRevision: null,
        sourceDirty: null,
        pythonExecutable: "python",
        pythonVersion: "3.12",
        environmentFingerprint: "test",
        instrumentationLevel: "minimal",
        hardwareSummary: "cpu",
      },
    });
    yield* store.updateState({
      runId,
      state:
        runId === options.nonterminalRun
          ? "running"
          : member.state === "failed"
            ? "failed"
            : "completed",
      at,
    });
    yield* store.appendMetrics({
      runId,
      seq: 1,
      batch: { step: 999, wallClockMs: 1, values: { "eval_after/loss": 9999 } },
      at,
    });
    if (runId === options.legacyRun) continue;
    const evaluation: RlEvaluationResult = {
      version: 1,
      protocolSha256: protocol.protocolSha256,
      samples: [
        { sampleId: "a", generationSeed: null, values: { "eval_after/loss": 0 } },
        {
          sampleId: "b",
          generationSeed: null,
          values: { "eval_after/loss": member.variantLabel === "candidate" ? 4 : 2 },
        },
      ],
    };
    const root = path.join(config.rlRunsDir, runId);
    yield* fs.makeDirectory(root, { recursive: true });
    const file = path.join(root, "study-evaluation.json");
    yield* fs.writeFileString(
      file,
      JSON.stringify(options.evaluation?.(runId, evaluation) ?? evaluation),
    );
    const identity = yield* computeArtifactIdentity({
      artifactPath: file,
      maxBytes: 8 * 1024 * 1024,
    });
    yield* store.recordArtifact({
      runId,
      kind: "evaluation",
      relativePath: "study-evaluation.json",
      logicalName: "study-evaluation.json",
      contentType: "application/json",
      producedAt: at,
      format: "json",
      ...identity,
    });
  }
  return study;
});

it.effect(
  "compares verified final sample artifacts through the public manager and ignores telemetry",
  () =>
    Effect.gen(function* () {
      yield* seedStudy();
      const manager = yield* RlManager;
      const result = yield* manager.compareStudy(input);
      assert.equal(result.n, 2);
      assert.equal(result.baselineMean, 1);
      assert.equal(result.candidateMean, 2);
      assert.equal(result.pairedDelta, 1);
      assert.equal(result.conclusion, "interval");
      assert.isBelow(result.interval![0], 1);
      assert.isAbove(result.interval![1], 1);
      const runSeed = yield* manager.compareStudy({
        ...input,
        estimator: { ...estimator, statisticalUnit: "run-seed" },
      });
      assert.deepEqual(runSeed.interval, [1, 1]);
    }).pipe(Effect.provide(layer), Effect.scoped),
);

it.effect("rejects source-byte changes even after the sample index has been populated", () =>
  Effect.gen(function* () {
    yield* seedStudy();
    const manager = yield* RlManager;
    yield* manager.compareStudy(input);
    const config = yield* ServerConfig.ServerConfig;
    const fs = yield* FileSystem.FileSystem;
    const file = `${config.rlRunsDir}/candidate_1/study-evaluation.json`;
    yield* fs.writeFileString(file, (yield* fs.readFileString(file)).replace('"b"', '"z"'));
    const result = yield* manager.compareStudy(input);
    assert.equal(result.n, 1);
    assert.equal(result.interval, null);
    assert.equal(result.excludedRuns[0]?.reason, "invalid-evaluation");
    assert.equal(result.dispersion, null);
  }).pipe(Effect.provide(layer), Effect.scoped),
);

it.effect("omits unfinished and legacy runs and fails strict missing-pair comparisons", () =>
  Effect.gen(function* () {
    yield* seedStudy({ legacyRun: "baseline_1", nonterminalRun: "candidate_2" });
    const manager = yield* RlManager;
    const result = yield* manager.compareStudy({
      ...input,
      estimator: { ...estimator, missingPairPolicy: "fail" },
    });
    assert.equal(result.conclusion, "missing-pairs");
    assert.equal(result.interval, null);
    assert.deepEqual(result.excludedRuns.map((run) => run.reason).sort(), [
      "missing-evaluation",
      "not-completed",
    ]);
  }).pipe(Effect.provide(layer), Effect.scoped),
);

it.effect("reports protocol mismatch instead of comparing incompatible evaluations", () =>
  Effect.gen(function* () {
    yield* seedStudy({
      evaluation: (runId, result) =>
        runId === "candidate_1" ? { ...result, protocolSha256: "f".repeat(64) } : result,
    });
    const result = yield* (yield* RlManager).compareStudy(input);
    assert.equal(result.conclusion, "incompatible-protocol");
    assert.equal(result.interval, null);
    assert.equal(result.pairedDelta, null);
  }).pipe(Effect.provide(layer), Effect.scoped),
);

it.effect(
  "counts failed members and rejects unknown variants without silently returning an estimate",
  () =>
    Effect.gen(function* () {
      yield* seedStudy({ failedRun: "baseline_1" });
      const manager = yield* RlManager;
      const result = yield* manager.compareStudy(input);
      assert.equal(result.failedRuns, 1);
      assert.equal(result.n, 1);
      assert.equal(
        (yield* manager.compareStudy({ ...input, baselineLabel: "missing" })).conclusion,
        "invalid-variants",
      );
      assert.equal(
        (yield* manager.compareStudy({ ...input, baselineLabel: "candidate" })).conclusion,
        "invalid-variants",
      );
    }).pipe(Effect.provide(layer), Effect.scoped),
);

it.effect("reports missing metrics, including names inherited from Object.prototype", () =>
  Effect.gen(function* () {
    yield* seedStudy();
    const manager = yield* RlManager;
    for (const metricKey of ["eval_after/unobserved", "toString"]) {
      const result = yield* manager.compareStudy({ ...input, metricKey });
      assert.equal(result.n, 0);
      assert.equal(result.interval, null);
      assert.equal(result.excludedRuns.length, 4);
      assert.isTrue(result.excludedRuns.every((entry) => entry.reason === "missing-metric"));
    }
  }).pipe(Effect.provide(layer), Effect.scoped),
);
