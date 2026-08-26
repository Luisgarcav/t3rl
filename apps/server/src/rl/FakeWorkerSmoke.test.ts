/**
 * Proves the real spawn, Python discovery, and NDJSON path outside the fake
 * spawner. Skipped when the host has no Python, so a machine without one still
 * has a green suite.
 *
 * These use `it.live` rather than `it.effect`: a real child process runs on the
 * real clock, and the cancellation grace period would never elapse under the
 * TestClock that `it.effect` installs.
 */
// @effect-diagnostics nodeBuiltinImport:off - a sync availability probe, before any Effect runtime exists.
import * as NodeChildProcess from "node:child_process";

import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import * as Deferred from "effect/Deferred";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import { describe } from "vite-plus/test";

import * as ServerConfig from "../config.ts";
import { SqlitePersistenceMemory } from "../persistence/Layers/Sqlite.ts";
import { CapabilitiesLive } from "./Capabilities.ts";
import * as Experiments from "./Experiments.ts";
import * as RlManager from "./Manager.ts";
import * as ProcessRunner from "../processRunner.ts";
import { RunStoreLive } from "./RunStore.ts";
import { WorkerSpawnerLive } from "./WorkerSpawner.ts";
import * as SourceEvidence from "./SourceEvidence.ts";

const pythonAvailable = (() => {
  try {
    return NodeChildProcess.spawnSync("python3", ["--version"]).status === 0;
  } catch {
    return false;
  }
})();

/** Repository root, so the worker entrypoint path in the definition resolves. */
const repoRoot = new URL("../../../../", import.meta.url).pathname;

const configLayer = Layer.effect(
  ServerConfig.ServerConfig,
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const stateDir = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-smoke-" });
    return ServerConfig.make({
      baseDir: repoRoot,
      rlRunsDir: `${stateDir}/rl`,
    } as never);
  }),
);

const smokeLayer = RlManager.RlManagerLive.pipe(
  Layer.provide(WorkerSpawnerLive),
  Layer.provide(CapabilitiesLive),
  Layer.provide(
    SourceEvidence.layerFromResolver(() =>
      Effect.succeed({ workspaceRoot: repoRoot, sourceRevision: "smoke", sourceDirty: false }),
    ),
  ),
  Layer.provide(ProcessRunner.layer),
  Layer.provide(Experiments.layerFromDirectory(`${repoRoot}python/t3rl_worker/experiments`)),
  Layer.provide(configLayer),
  Layer.provideMerge(RunStoreLive),
  Layer.provideMerge(SqlitePersistenceMemory),
  Layer.provideMerge(NodeServices.layer),
);

const TERMINAL = new Set(["completed", "failed", "cancelled", "interrupted"]);

/**
 * Awaits the run's terminal lifecycle event. This test drives a real process so
 * it cannot use TestClock, but it still must not poll: the manager's own
 * subscription is the signal.
 */
const awaitTerminal = (manager: RlManager.RlManagerShape, runId: string) =>
  Effect.gen(function* () {
    const reached = yield* Deferred.make<string>();
    const unsubscribe = yield* manager.subscribe({ runId }, (event) => {
      if (event._tag !== "Snapshot" && event._tag !== "Lifecycle") return;
      if (TERMINAL.has(event.summary.state)) {
        Deferred.doneUnsafe(reached, Effect.succeed(event.summary.state));
      }
    });
    return yield* Deferred.await(reached).pipe(Effect.ensuring(Effect.sync(unsubscribe)));
  });

describe.skipIf(!pythonAvailable)("fake worker smoke", () => {
  it.live(
    "runs the checked-in fake experiment to completion",
    () =>
      Effect.gen(function* () {
        const manager = yield* RlManager.RlManager;
        const { runId } = yield* manager.start({
          projectId: "proj_smoke",
          experimentId: "fake",
          seed: 7,
        });

        const state = yield* awaitTerminal(manager, runId);
        assert.strictEqual(state, "completed");

        const detail = yield* manager.get({ runId });
        assert.isTrue(detail.artifacts.some((artifact) => artifact.kind === "summary"));
        assert.isNotNull(detail.manifest);
        assert.strictEqual(detail.manifest?.seed, 7);
      }).pipe(Effect.provide(smokeLayer), Effect.scoped),
    { timeout: 30_000 },
  );

  it.live(
    "stops an unresponsive worker, escalating past the ignored SIGTERM",
    () =>
      Effect.gen(function* () {
        const manager = yield* RlManager.RlManager;
        const { runId } = yield* manager.start({
          projectId: "proj_smoke",
          experimentId: "fake-ignore-cancel",
          seed: 7,
        });

        // Wait until the worker is actually running before cancelling, so the
        // test exercises signal delivery rather than a race with spawn.
        yield* Effect.gen(function* () {
          const running = yield* Deferred.make<void>();
          const unsubscribe = yield* manager.subscribe({ runId }, (event) => {
            if (
              (event._tag === "Snapshot" || event._tag === "Lifecycle") &&
              event.summary.state === "running"
            ) {
              Deferred.doneUnsafe(running, Effect.void);
            }
          });
          yield* Deferred.await(running).pipe(Effect.ensuring(Effect.sync(unsubscribe)));
        });

        yield* manager.cancel({ runId });
        const state = yield* awaitTerminal(manager, runId);
        assert.strictEqual(state, "cancelled");
      }).pipe(Effect.provide(smokeLayer), Effect.scoped),
    { timeout: 30_000 },
  );
});
