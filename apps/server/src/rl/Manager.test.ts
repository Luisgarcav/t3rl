// @effect-diagnostics preferSchemaOverJson:off - hand-written worker stdout fixtures.
import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import * as Deferred from "effect/Deferred";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as TestClock from "effect/testing/TestClock";
import { describe } from "vite-plus/test";

import * as ServerConfig from "../config.ts";
import { SqlitePersistenceMemory } from "../persistence/Layers/Sqlite.ts";
import { Capabilities, type CapabilitiesShape } from "./Capabilities.ts";
import * as Experiments from "./Experiments.ts";
import * as RlManager from "./Manager.ts";
import { RunStoreLive } from "./RunStore.ts";
import { FakeWorkerProcess, fakeWorkerSpawnerLayer } from "./testing/FakeWorkerSpawner.ts";

const hello = JSON.stringify({
  type: "hello",
  protocol: 1,
  runner: "fake",
  runnerVersion: "0.1.0",
});

const metrics = (step: number, value: number) =>
  JSON.stringify({
    type: "metrics",
    step,
    wallClockMs: step * 10,
    values: { "train/return": value },
  });

const fakeDefinition: Experiments.RlExperimentDefinition = {
  experimentId: "fake",
  runnerId: "fake",
  instrumentationLevel: "minimal",
  defaultSeed: 7,
  entrypoint: "python/t3rl_worker/fake_worker.py",
  scenario: "success",
  config: {},
};

const capabilitiesLayer = Layer.succeed(
  Capabilities,
  Capabilities.of({
    report: () =>
      Effect.succeed({
        runners: [
          {
            runnerId: "fake",
            available: true,
            version: "3.12.4",
            failureCode: null,
            remedy: null,
          },
        ],
      }),
    resolvePython: () => Effect.succeed({ executable: "python3", version: "3.12.4" }),
  } satisfies CapabilitiesShape),
);

/**
 * Each test gets a scoped temp state directory, removed when the scope closes.
 * Artifacts are real files, so the run root has to be real too.
 */
const configLayer = Layer.effect(
  ServerConfig.ServerConfig,
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const stateDir = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-manager-test-" });
    // Shaped like the real flat ServerConfig, so a wrong access path fails
    // here instead of only in production.
    return ServerConfig.make({
      baseDir: stateDir,
      rlRunsDir: `${stateDir}/rl`,
    } as never);
  }),
);

const makeLayer = (worker: FakeWorkerProcess) =>
  RlManager.RlManagerLive.pipe(
    Layer.provide(fakeWorkerSpawnerLayer(worker)),
    Layer.provide(capabilitiesLayer),
    Layer.provide(Experiments.layerFromRecord({ fake: fakeDefinition })),
    Layer.provide(configLayer),
    Layer.provideMerge(RunStoreLive),
    Layer.provideMerge(SqlitePersistenceMemory),
    Layer.provideMerge(NodeServices.layer),
  );

const withWorker = <A, E>(
  worker: FakeWorkerProcess,
  effect: Effect.Effect<A, E, RlManager.RlManager>,
) => effect.pipe(Effect.provide(makeLayer(worker)), Effect.scoped);

/**
 * Lets the forked stdout/stderr pumps make progress. Adequate for intermediate
 * assertions, but never for waiting on an outcome: yielding a fixed number of
 * times is a guess about scheduling, not a synchronisation point. Anything that
 * must have happened is awaited with `awaitState` instead.
 */
const settle = Effect.gen(function* () {
  for (let index = 0; index < 50; index += 1) {
    yield* Effect.yieldNow;
  }
});

/**
 * Resolves when the run reaches a state the predicate accepts. Subscribes
 * first, so the snapshot covers a state that was already reached before this
 * call, and no event between subscribe and check can be missed.
 */
const awaitState = (
  manager: RlManager.RlManagerShape,
  runId: string,
  predicate: (state: string) => boolean,
) =>
  Effect.gen(function* () {
    const reached = yield* Deferred.make<string>();
    const unsubscribe = yield* manager.subscribe({ runId }, (event) => {
      const state = event._tag === "Metrics" ? null : event.summary.state;
      if (state !== null && predicate(state)) {
        Deferred.doneUnsafe(reached, Effect.succeed(state));
      }
    });
    return yield* Deferred.await(reached).pipe(Effect.ensuring(Effect.sync(unsubscribe)));
  });

const TERMINAL = new Set(["completed", "failed", "cancelled", "interrupted"]);
const awaitTerminal = (manager: RlManager.RlManagerShape, runId: string) =>
  awaitState(manager, runId, (state) => TERMINAL.has(state));

describe("RlManager", () => {
  it.effect("reaches running only after hello, and completed only after done", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });

          const preparing = yield* manager.get({ runId });
          assert.strictEqual(preparing.summary.state, "preparing");

          worker.emitStdout(hello);
          yield* awaitState(manager, runId, (state) => state === "running");
          assert.strictEqual((yield* manager.get({ runId })).summary.state, "running");

          worker.emitStdout(JSON.stringify({ type: "done", status: "completed" }));
          yield* awaitTerminal(manager, runId);
          assert.strictEqual((yield* manager.get({ runId })).summary.state, "completed");
        }),
      );
    }),
  );

  it.effect("never reports completion from a quiet metric stream", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          worker.emitStdout(hello);
          worker.emitStdout(metrics(1, 1));
          yield* awaitState(manager, runId, (state) => state === "running");

          // A long silence is not an ending. Nothing to await here — the point
          // is that no transition happens — so time is advanced and the state
          // re-read.
          yield* TestClock.adjust("10 minutes");
          yield* settle;

          assert.strictEqual((yield* manager.get({ runId })).summary.state, "running");
        }),
      );
    }),
  );

  it.effect("fails the run when the worker exits without done", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          worker.emitStdout(hello);
          yield* awaitState(manager, runId, (state) => state === "running");

          // Exit code 0 is still a failure: the worker never claimed to finish.
          worker.exit(0);
          yield* awaitTerminal(manager, runId);

          const detail = yield* manager.get({ runId });
          assert.strictEqual(detail.summary.state, "failed");
          assert.strictEqual(detail.summary.errorCode, "WorkerExited");
        }),
      );
    }),
  );

  it.effect("fails the run with ProtocolIncompatible on an unsupported hello", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          worker.emitStdout(
            JSON.stringify({ type: "hello", protocol: 99, runner: "fake", runnerVersion: "9" }),
          );
          yield* awaitTerminal(manager, runId);

          const detail = yield* manager.get({ runId });
          assert.strictEqual(detail.summary.state, "failed");
          assert.strictEqual(detail.summary.errorCode, "ProtocolIncompatible");
        }),
      );
    }),
  );

  it.effect("escalates SIGTERM to SIGKILL when the worker ignores cancellation", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          worker.emitStdout(hello);
          yield* awaitState(manager, runId, (state) => state === "running");

          yield* manager.cancel({ runId });
          yield* worker.awaitKills(1);
          assert.deepStrictEqual(worker.killSignals, ["SIGTERM"]);

          yield* TestClock.adjust(RlManager.CANCEL_GRACE);
          yield* worker.awaitKills(2);
          assert.deepStrictEqual(worker.killSignals, ["SIGTERM", "SIGKILL"]);

          worker.exit(null);
          yield* awaitTerminal(manager, runId);
          assert.strictEqual((yield* manager.get({ runId })).summary.state, "cancelled");
        }),
      );
    }),
  );

  it.effect("records completed, not cancelled, when done arrives during cancellation", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          worker.emitStdout(hello);
          yield* awaitState(manager, runId, (state) => state === "running");

          yield* manager.cancel({ runId });
          yield* worker.awaitKills(1);

          // The worker finished on its own before the signal landed. That is a
          // real result and must not be discarded as a cancellation.
          worker.emitStdout(JSON.stringify({ type: "done", status: "completed" }));
          yield* awaitTerminal(manager, runId);
          worker.exit(0);

          assert.strictEqual((yield* manager.get({ runId })).summary.state, "completed");
        }),
      );
    }),
  );

  it.effect("coalesces a metric burst instead of queueing it", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          worker.emitStdout(hello);
          yield* awaitState(manager, runId, (state) => state === "running");

          const received: number[] = [];
          const flushed = yield* Deferred.make<void>();
          yield* manager.subscribe({ runId }, (event) => {
            if (event._tag === "Metrics") {
              received.push(event.batch.step);
              Deferred.doneUnsafe(flushed, Effect.void);
            }
          });

          for (let step = 1; step <= 50; step += 1) {
            worker.emitStdout(metrics(step, step));
          }
          yield* settle;
          yield* TestClock.adjust("1 second");
          yield* Deferred.await(flushed);

          // One flush window collapses the burst to its latest value.
          assert.strictEqual(received.length, 1);
          assert.strictEqual(received[0], 50);
        }),
      );
    }),
  );

  it.effect("replays a snapshot then live events to a late subscriber", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          worker.emitStdout(hello);
          yield* awaitState(manager, runId, (state) => state === "running");

          const tags: string[] = [];
          yield* manager.subscribe({ runId }, (event) => tags.push(event._tag));

          assert.strictEqual(tags[0], "Snapshot");

          worker.emitStdout(JSON.stringify({ type: "done", status: "completed" }));
          yield* awaitTerminal(manager, runId);

          assert.isTrue(tags.includes("Lifecycle"));
        }),
      );
    }),
  );

  it.effect("bounds the retained stderr buffer", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          worker.emitStdout(hello);
          yield* awaitState(manager, runId, (state) => state === "running");

          // Far more than the cap; the manager must drop oldest, not grow.
          const line = "x".repeat(1024);
          for (let index = 0; index < 200; index += 1) {
            worker.emitStderr(line);
          }
          yield* settle;

          worker.exit(1);
          yield* awaitTerminal(manager, runId);

          const detail = yield* manager.get({ runId });
          const log = detail.artifacts.find((artifact) => artifact.kind === "log");
          assert.isDefined(log);
          assert.isTrue((log?.bytes ?? 0) <= RlManager.MAX_STDERR_BYTES);
        }),
      );
    }),
  );

  it.effect("fails the start and the run when the experiment is unknown", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const exit = yield* Effect.exit(
            manager.start({ projectId: "proj_01", experimentId: "missing", seed: 7 }),
          );
          assert.isTrue(exit._tag === "Failure");

          // The capability failure is visible before any process existed.
          const { runs } = yield* manager.list({ projectId: "proj_01" });
          assert.strictEqual(runs[0]?.state, "failed");
          assert.strictEqual(runs[0]?.errorCode, "RunnerUnavailable");
          assert.strictEqual(worker.spawnInputs.length, 0);
        }),
      );
    }),
  );
});
