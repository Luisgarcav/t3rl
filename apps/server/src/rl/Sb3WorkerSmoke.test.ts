/** Opt-in real PPO smoke: T3RL_REAL_SMOKE=1 vp test run <this file>. */
// @effect-diagnostics nodeBuiltinImport:off - gated host capability probe.
import * as NodeChildProcess from "node:child_process";
import * as NodeURL from "node:url";

import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import * as Deferred from "effect/Deferred";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import { describe } from "vite-plus/test";

import * as ServerConfig from "../config.ts";
import { SqlitePersistenceMemory } from "../persistence/Layers/Sqlite.ts";
import * as ProcessRunner from "../processRunner.ts";
import { CapabilitiesLive } from "./Capabilities.ts";
import * as Experiments from "./Experiments.ts";
import * as RlManager from "./Manager.ts";
import { RunStoreLive } from "./RunStore.ts";
import * as SourceEvidence from "./SourceEvidence.ts";
import { WorkerSpawnerLive } from "./WorkerSpawner.ts";

const repoRoot = NodeURL.fileURLToPath(new URL("../../../../", import.meta.url));
const python =
  process.env["T3RL_PYTHON_STABLE_BASELINES3"]?.trim() ||
  process.env["T3RL_PYTHON"]?.trim() ||
  "python3";
const worker = `${repoRoot}python/t3rl_worker/sb3_worker.py`;
const realSmokeEnabled =
  process.env["T3RL_REAL_SMOKE"] === "1" &&
  NodeChildProcess.spawnSync(python, [worker, "--probe"]).status === 0;

const definition: Experiments.RlExperimentDefinition = {
  experimentId: "cartpole-ppo-smoke",
  displayName: "CartPole smoke",
  description: "Short gated integration run",
  runnerId: "stable-baselines3",
  instrumentationLevel: "standard",
  defaultSeed: 7,
  entrypoint: worker,
  maxRuntimeSeconds: 120,
  config: {
    algorithm: "PPO",
    environment: "CartPole-v1",
    policy: "MlpPolicy",
    device: "cpu",
    totalTimesteps: 512,
    nSteps: 128,
    batchSize: 64,
    evaluationEpisodes: 2,
  },
};

const configLayer = Layer.effect(
  ServerConfig.ServerConfig,
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const stateDir = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-sb3-smoke-" });
    return ServerConfig.make({ baseDir: stateDir, rlRunsDir: `${stateDir}/rl` } as never);
  }),
);

const smokeLayer = RlManager.RlManagerLive.pipe(
  Layer.provide(WorkerSpawnerLive),
  Layer.provide(CapabilitiesLive),
  Layer.provide(ProcessRunner.layer),
  Layer.provide(Experiments.layerFromRecord({ "cartpole-ppo-smoke": definition })),
  Layer.provide(
    SourceEvidence.layerFromResolver(() =>
      Effect.succeed({ workspaceRoot: repoRoot, sourceRevision: "smoke", sourceDirty: false }),
    ),
  ),
  Layer.provide(configLayer),
  Layer.provideMerge(RunStoreLive),
  Layer.provideMerge(SqlitePersistenceMemory),
  Layer.provideMerge(NodeServices.layer),
);

const TERMINAL = new Set(["completed", "failed", "cancelled", "interrupted"]);

describe.skipIf(!realSmokeEnabled)("Stable-Baselines3 worker smoke", () => {
  it.live(
    "trains, evaluates, and retains the Phase 1 artifacts",
    () =>
      Effect.gen(function* () {
        const manager = yield* RlManager.RlManager;
        const { runId } = yield* manager.start({
          projectId: "proj_sb3_smoke",
          experimentId: "cartpole-ppo-smoke",
          seed: 7,
        });
        const terminal = yield* Deferred.make<string>();
        const unsubscribe = yield* manager.subscribe({ runId }, (event) => {
          if (
            (event._tag === "Snapshot" || event._tag === "Lifecycle") &&
            TERMINAL.has(event.summary.state)
          ) {
            Deferred.doneUnsafe(terminal, Effect.succeed(event.summary.state));
          }
        });
        const state = yield* Deferred.await(terminal).pipe(
          Effect.ensuring(Effect.sync(unsubscribe)),
        );
        assert.strictEqual(state, "completed");

        const detail = yield* manager.get({ runId });
        const kinds = new Set(detail.artifacts.map((artifact) => artifact.kind));
        for (const kind of ["manifest", "summary", "model", "evaluation", "replay", "log"]) {
          assert.isTrue(kinds.has(kind as never));
        }
        assert.isAtLeast(detail.metrics.length, 1);
        assert.strictEqual(detail.manifest?.effectiveConfig["environment"], "CartPole-v1");
      }).pipe(Effect.provide(smokeLayer), Effect.scoped),
    { timeout: 120_000 },
  );
});
