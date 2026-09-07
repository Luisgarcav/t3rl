// @effect-diagnostics nodeBuiltinImport:off - resolves the checked-in Python fixture.
import * as NodeURL from "node:url";

import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import type { RlEvaluationProtocol } from "@t3tools/contracts";
import * as Deferred from "effect/Deferred";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import { describe } from "vite-plus/test";

import * as ServerConfig from "../config.ts";
import { SqlitePersistenceMemory } from "../persistence/Layers/Sqlite.ts";
import * as ProcessRunner from "../processRunner.ts";
import * as ArtifactIdentity from "./ArtifactIdentity.ts";
import * as Artifacts from "./Artifacts.ts";
import { CapabilitiesLive } from "./Capabilities.ts";
import * as Experiments from "./Experiments.ts";
import * as RlManager from "./Manager.ts";
import { RunStoreLive } from "./RunStore.ts";
import * as SourceEvidence from "./SourceEvidence.ts";
import { WorkerSpawnerLive } from "./WorkerSpawner.ts";

const repoRoot = NodeURL.fileURLToPath(new URL("../../../../", import.meta.url));
const worker = `${repoRoot}python/t3rl_worker/fake_worker.py`;

const definition: Experiments.RlExperimentDefinition = {
  experimentId: "fake-checkpoint-smoke",
  displayName: "Checkpoint fixture",
  description: "Standard-library protocol-v2 checkpoint fixture",
  runnerId: "fake",
  protocolVersion: 2,
  instrumentationLevel: "standard",
  defaultSeed: 7,
  entrypoint: worker,
  scenario: "checkpoint",
  maxRuntimeSeconds: 60,
  config: {
    modelId: "t3rl/tiny-linear-fixture",
    modelRevision: "f".repeat(64),
    tokenizerRevision: "f".repeat(64),
    quantization: "none",
    precision: "fp32",
    loraRank: 2,
    loraAlpha: 4,
    loraDropout: 0,
    loraBias: "none",
    loraTargetModules: ["linear"],
    loraModulesToSave: [],
    useRslora: false,
    checkpointCadenceSteps: 2,
    maxIntermediateCheckpoints: 2,
    keepBest: false,
    keepFinal: true,
    gracefulCheckpointDeadlineSeconds: 5,
    maxSteps: 8,
  },
};

const cancellationDefinition: Experiments.RlExperimentDefinition = {
  ...definition,
  experimentId: "fake-checkpoint-cancellation-smoke",
  displayName: "Graceful checkpoint fixture",
  config: {
    ...definition.config,
    maxSteps: 100,
    checkpointCadenceSteps: 10,
    maxIntermediateCheckpoints: 1,
    stepDelayMs: 25,
  },
};

const configLayer = Layer.effect(
  ServerConfig.ServerConfig,
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const stateDir = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-checkpoint-smoke-" });
    return ServerConfig.make({ baseDir: stateDir, rlRunsDir: `${stateDir}/rl` } as never);
  }),
);

const experimentLayer = Layer.unwrap(
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const directory = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-checkpoint-project-" });
    const dataset = `${directory}/dataset.json`;
    yield* fs.writeFileString(dataset, '[{"sampleId":"fixture","text":"checkpoint fixture"}]\n');
    const identity = yield* ArtifactIdentity.computeArtifactIdentity({
      artifactPath: dataset,
      maxBytes: 1024,
    });
    const projectDefinition: Experiments.ResolvedExperimentDefinition = {
      ...definition,
      config: { ...definition.config, projectDatasetPath: dataset },
      projectInputs: [
        {
          role: "dataset",
          sourcePath: dataset,
          snapshotName: "dataset.json",
          sha256: identity.sha256,
          bytes: identity.bytes,
        },
      ],
    };
    return Experiments.layerFromRecord({
      [definition.experimentId]: projectDefinition,
      [cancellationDefinition.experimentId]: cancellationDefinition,
      "fake-checkpoint-no-inputs": { ...definition, experimentId: "fake-checkpoint-no-inputs" },
    });
  }),
);

const smokeLayer = RlManager.RlManagerLive.pipe(
  Layer.provide(WorkerSpawnerLive),
  Layer.provide(CapabilitiesLive),
  Layer.provide(ProcessRunner.layer),
  Layer.provide(experimentLayer),
  Layer.provide(
    SourceEvidence.layerFromResolver(() =>
      Effect.succeed({ workspaceRoot: repoRoot, sourceRevision: "fixture", sourceDirty: false }),
    ),
  ),
  Layer.provideMerge(configLayer),
  Layer.provideMerge(RunStoreLive),
  Layer.provideMerge(SqlitePersistenceMemory),
  Layer.provideMerge(NodeServices.layer),
);

const terminalStates = new Set(["completed", "failed", "cancelled", "interrupted"]);

const awaitTerminal = (manager: RlManager.RlManagerShape, runId: string) =>
  Effect.gen(function* () {
    const terminal = yield* Deferred.make<string>();
    const unsubscribe = yield* manager.subscribe({ runId }, (event) => {
      if (
        (event._tag === "Snapshot" || event._tag === "Lifecycle") &&
        terminalStates.has(event.summary.state)
      ) {
        Deferred.doneUnsafe(terminal, Effect.succeed(event.summary.state));
      }
    });
    return yield* Deferred.await(terminal).pipe(Effect.ensuring(Effect.sync(unsubscribe)));
  });

describe("protocol-v2 checkpoint worker smoke", () => {
  it.live(
    "resumes deterministically, exposes lineage, and never mutates the parent",
    () =>
      Effect.gen(function* () {
        const manager = yield* RlManager.RlManager;
        const config = yield* ServerConfig.ServerConfig;
        const fs = yield* FileSystem.FileSystem;
        const seeds = { training: 7, data: 7, evaluationSample: 7, generation: 7 };
        const evaluationProtocol: RlEvaluationProtocol = {
          version: 1,
          protocolSha256: "a".repeat(64),
          datasetFingerprint: "checkpoint-fixture-v1",
          split: "test",
          sampleIds: ["held-out"],
          generationSeedPolicy: "fixed-per-sample",
          decoding: {},
          verifierSha256: "b".repeat(64),
        };
        const parent = yield* manager.start({
          projectId: "proj_checkpoint_smoke",
          experimentId: definition.experimentId,
          seed: 7,
          seeds,
          evaluationProtocol,
        });
        assert.strictEqual(yield* awaitTerminal(manager, parent.runId), "completed");
        const parentDetail = yield* manager.get({ runId: parent.runId });
        const checkpoint = parentDetail.artifacts.find(
          (artifact) =>
            artifact.evidence?._tag === "Checkpoint" && artifact.evidence.globalStep === 4,
        );
        const parentAdapter = parentDetail.artifacts.find(
          (artifact) => artifact.evidence?._tag === "Adapter",
        );
        assert.isDefined(checkpoint);
        assert.isDefined(parentAdapter);
        assert.strictEqual(checkpoint?.state, "ready");
        const intermediateCheckpoints = parentDetail.artifacts
          .filter(
            (artifact) =>
              artifact.evidence?._tag === "Checkpoint" &&
              artifact.evidence.checkpointClass === "intermediate",
          )
          .toSorted((left, right) => (right.checkpointStep ?? -1) - (left.checkpointStep ?? -1));
        assert.deepStrictEqual(
          intermediateCheckpoints.map((artifact) => [artifact.checkpointStep, artifact.state]),
          [
            [6, "ready"],
            [4, "ready"],
            [2, "trashed"],
          ],
        );
        if (
          checkpoint === undefined ||
          checkpoint.sha256 === null ||
          checkpoint.sha256 === undefined
        ) {
          return yield* Effect.die("Fixture checkpoint was not verified");
        }
        if (parentAdapter === undefined) {
          return yield* Effect.die("Fixture adapter was not published");
        }

        const adapterAsResume = yield* Effect.flip(
          manager.resume({
            projectId: "proj_checkpoint_smoke",
            parentRunId: parent.runId,
            sourceArtifactId: parentAdapter.artifactId,
            requestId: "invalid_resume_fixture_01",
          }),
        );
        assert.strictEqual(adapterAsResume.code, "ResumeIncompatible");
        const droppedInputs = yield* manager
          .resume({
            projectId: "proj_checkpoint_smoke",
            parentRunId: parent.runId,
            sourceArtifactId: checkpoint.artifactId,
            requestId: "removed_inputs_fixture_01",
            targetExperimentId: "fake-checkpoint-no-inputs",
          })
          .pipe(Effect.flip);
        assert.strictEqual(droppedInputs.code, "ResumeIncompatible");
        assert.include(droppedInputs.detail, "project input snapshots changed");

        const parentRoot = Artifacts.runDirectory({
          rlRunsDir: config.rlRunsDir,
          runId: parent.runId,
        });
        assert.isString(parentRoot);
        assert.isFalse(yield* fs.exists(`${parentRoot}/trainer`));
        assert.isFalse(yield* fs.exists(`${parentRoot}/checkpoints/checkpoint-2-intermediate`));
        const parentBefore = yield* ArtifactIdentity.computeArtifactIdentity({
          artifactPath: parentRoot!,
          maxBytes: RlManager.MAX_RUN_ARTIFACT_BYTES,
        });

        const child = yield* manager.resume({
          projectId: "proj_checkpoint_smoke",
          parentRunId: parent.runId,
          sourceArtifactId: checkpoint.artifactId,
          requestId: "resume_fixture_01",
        });
        assert.strictEqual(yield* awaitTerminal(manager, child.runId), "completed");
        const childDetail = yield* manager.get({ runId: child.runId });
        assert.deepStrictEqual(childDetail.manifest?.seeds, seeds);
        assert.deepStrictEqual(
          childDetail.manifest?.effectiveConfig.evaluationProtocol,
          evaluationProtocol,
        );
        const childAdapter = childDetail.artifacts.find(
          (artifact) => artifact.evidence?._tag === "Adapter",
        );
        assert.strictEqual(childAdapter?.sha256, parentAdapter?.sha256);
        assert.deepStrictEqual(
          childDetail.metrics.at(-1)?.values,
          parentDetail.metrics.at(-1)?.values,
        );
        assert.deepInclude(childDetail.lineage.edges[0], {
          childRunId: child.runId,
          parentRunId: parent.runId,
          sourceArtifactId: checkpoint.artifactId,
          sourceArtifactSha256: checkpoint.sha256,
          relation: "resume",
          sourceStep: 4,
        });
        const retriedChild = yield* manager.resume({
          projectId: "proj_checkpoint_smoke",
          parentRunId: parent.runId,
          sourceArtifactId: checkpoint.artifactId,
          requestId: "resume_fixture_01",
        });
        assert.strictEqual(retriedChild.runId, child.runId);
        const otherCheckpoint = parentDetail.artifacts.find(
          (artifact) =>
            artifact.state === "ready" &&
            artifact.evidence?._tag === "Checkpoint" &&
            artifact.evidence.globalStep === 6,
        );
        assert.isDefined(otherCheckpoint);
        if (otherCheckpoint === undefined) {
          return yield* Effect.die("Fixture did not retain checkpoint step 6");
        }
        const mismatchedRetry = yield* Effect.flip(
          manager.resume({
            projectId: "proj_checkpoint_smoke",
            parentRunId: parent.runId,
            sourceArtifactId: otherCheckpoint.artifactId,
            requestId: "resume_fixture_01",
          }),
        );
        assert.strictEqual(mismatchedRetry.code, "ResumeIncompatible");

        const warmed = yield* manager.warmStart({
          projectId: "proj_checkpoint_smoke",
          parentRunId: parent.runId,
          sourceArtifactId: parentAdapter.artifactId,
          requestId: "warm_fixture_01",
        });
        assert.strictEqual(yield* awaitTerminal(manager, warmed.runId), "completed");
        const warmedDetail = yield* manager.get({ runId: warmed.runId });
        assert.deepInclude(warmedDetail.lineage.edges[0], {
          childRunId: warmed.runId,
          parentRunId: parent.runId,
          sourceArtifactId: parentAdapter.artifactId,
          relation: "warm-start",
          sourceStep: 8,
        });

        const parentAfter = yield* ArtifactIdentity.computeArtifactIdentity({
          artifactPath: parentRoot!,
          maxBytes: RlManager.MAX_RUN_ARTIFACT_BYTES,
        });
        assert.strictEqual(parentAfter.sha256, parentBefore.sha256);
      }).pipe(Effect.provide(smokeLayer), Effect.scoped),
    { timeout: 30_000 },
  );

  it.live(
    "publishes a verified graceful checkpoint when cancellation interrupts training",
    () =>
      Effect.gen(function* () {
        const manager = yield* RlManager.RlManager;
        const run = yield* manager.start({
          projectId: "proj_checkpoint_cancel_smoke",
          experimentId: cancellationDefinition.experimentId,
          seed: 7,
        });
        const observedMetric = yield* Deferred.make<void>();
        const unsubscribe = yield* manager.subscribe({ runId: run.runId }, (event) => {
          if (event._tag === "Metrics") {
            Deferred.doneUnsafe(observedMetric, Effect.void);
          }
        });
        yield* Deferred.await(observedMetric);
        const cancellation = yield* manager.cancel({ runId: run.runId });
        assert.strictEqual(cancellation.state, "cancelling");
        assert.strictEqual(yield* awaitTerminal(manager, run.runId), "cancelled");
        unsubscribe();

        const detail = yield* manager.get({ runId: run.runId });
        const graceful = detail.artifacts.find(
          (artifact) =>
            artifact.state === "ready" &&
            artifact.sha256 !== null &&
            artifact.sha256 !== undefined &&
            artifact.evidence?._tag === "Checkpoint" &&
            artifact.evidence.checkpointClass === "graceful",
        );
        assert.isDefined(graceful);
      }).pipe(Effect.provide(smokeLayer), Effect.scoped),
    { timeout: 30_000 },
  );
});
