// @effect-diagnostics preferSchemaOverJson:off - hand-written worker stdout fixtures.
import type { RlArtifactMetadata, RlSubscriptionEvent } from "@t3tools/contracts";
import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import * as Deferred from "effect/Deferred";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Path from "effect/Path";
import * as TestClock from "effect/testing/TestClock";
import { describe } from "vite-plus/test";

import * as ServerConfig from "../config.ts";
import { SqlitePersistenceMemory } from "../persistence/Layers/Sqlite.ts";
import { Capabilities, type CapabilitiesShape } from "./Capabilities.ts";
import * as Experiments from "./Experiments.ts";
import * as RlManager from "./Manager.ts";
import { RunStoreLive } from "./RunStore.ts";
import * as SourceEvidence from "./SourceEvidence.ts";
import { FakeWorkerProcess, fakeWorkerSpawnerLayer } from "./testing/FakeWorkerSpawner.ts";

const hello = JSON.stringify({
  type: "hello",
  protocol: 1,
  runner: "fake",
  runnerVersion: "0.1.0",
});

const workerManifest = JSON.stringify({ type: "manifest", values: {} });

const emitReady = (worker: FakeWorkerProcess) => {
  worker.emitStdout(hello);
  worker.emitStdout(workerManifest);
};

const metrics = (step: number, value: number) =>
  JSON.stringify({
    type: "metrics",
    step,
    wallClockMs: step * 10,
    values: { "train/return": value },
  });

const fakeDefinition: Experiments.RlExperimentDefinition = {
  experimentId: "fake",
  displayName: "Fake",
  description: "Test worker",
  runnerId: "fake",
  instrumentationLevel: "minimal",
  defaultSeed: 7,
  entrypoint: "python/t3rl_worker/fake_worker.py",
  scenario: "success",
  maxRuntimeSeconds: 3600,
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
        experiments: [],
      }),
    resolvePython: () => Effect.succeed({ executable: "python3", version: "3.12.4" }),
    resolveRunner: () =>
      Effect.succeed({
        executable: "python3",
        version: "3.12.4",
        runnerId: "fake",
        runnerVersion: "0.1.0",
        environmentFingerprint: "fake-fingerprint",
        environmentLock: null,
      }),
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
    Layer.provide(
      SourceEvidence.layerFromResolver(() =>
        Effect.succeed({ workspaceRoot: "/tmp", sourceRevision: "abc123", sourceDirty: false }),
      ),
    ),
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
      const state =
        event._tag === "Snapshot" || event._tag === "Lifecycle" ? event.summary.state : null;
      if (state !== null && predicate(state)) {
        Deferred.doneUnsafe(reached, Effect.succeed(state));
      }
    });
    return yield* Deferred.await(reached).pipe(Effect.ensuring(Effect.sync(unsubscribe)));
  });

const TERMINAL = new Set(["completed", "failed", "cancelled", "interrupted"]);
const awaitTerminal = (manager: RlManager.RlManagerShape, runId: string) =>
  awaitState(manager, runId, (state) => TERMINAL.has(state));

const awaitManifest = (manager: RlManager.RlManagerShape, runId: string) =>
  Effect.gen(function* () {
    const received = yield* Deferred.make<void>();
    const unsubscribe = yield* manager.subscribe({ runId }, (event) => {
      if (event._tag === "Manifest" || (event._tag === "Snapshot" && event.manifest !== null)) {
        Deferred.doneUnsafe(received, Effect.void);
      }
    });
    yield* Deferred.await(received).pipe(Effect.ensuring(Effect.sync(unsubscribe)));
  });

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

          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "running");
          yield* awaitManifest(manager, runId);
          const running = yield* manager.get({ runId });
          assert.strictEqual(running.summary.state, "running");
          assert.strictEqual(running.manifest?.sourceRevision, "abc123");
          assert.strictEqual(running.manifest?.sourceDirty, false);
          assert.strictEqual(worker.spawnInputs[0]?.cwd, "/tmp");

          worker.emitStdout(JSON.stringify({ type: "done", status: "completed" }));
          worker.exit(0);
          yield* awaitTerminal(manager, runId);
          assert.strictEqual((yield* manager.get({ runId })).summary.state, "completed");
        }),
      );
    }),
  );

  it.effect("fails a quiet metric stream as stalled instead of reporting completion", () =>
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
          emitReady(worker);
          worker.emitStdout(metrics(1, 1));
          yield* awaitState(manager, runId, (state) => state === "running");

          // A quiet stream is never success; after the liveness deadline it is
          // an explicit worker stall and the captured process is terminated.
          yield* TestClock.adjust("10 minutes");
          yield* awaitTerminal(manager, runId);

          const detail = yield* manager.get({ runId });
          assert.strictEqual(detail.summary.state, "failed");
          assert.strictEqual(detail.summary.errorCode, "WorkerStalled");
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
          emitReady(worker);
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

  it.effect("requires exit zero after done before recording completion", () =>
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
          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "running");
          worker.emitStdout(JSON.stringify({ type: "done", status: "completed" }));
          worker.exit(9);
          yield* awaitTerminal(manager, runId);

          const detail = yield* manager.get({ runId });
          assert.strictEqual(detail.summary.state, "failed");
          assert.strictEqual(detail.summary.errorCode, "RunnerException");
        }),
      );
    }),
  );

  it.effect("fails and stops a worker that emits metrics before hello", () =>
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
          worker.emitStdout(metrics(1, 1));
          yield* awaitTerminal(manager, runId);
          yield* worker.awaitKills(1);

          const detail = yield* manager.get({ runId });
          assert.strictEqual(detail.summary.errorCode, "MalformedWorkerMessage");
          assert.deepStrictEqual(worker.killSignals, ["SIGTERM"]);
        }),
      );
    }),
  );

  it.effect("times out a worker that never sends hello", () =>
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
          yield* TestClock.adjust(RlManager.STARTUP_TIMEOUT);
          yield* awaitTerminal(manager, runId);
          yield* worker.awaitKills(1);
          assert.strictEqual(
            (yield* manager.get({ runId })).summary.errorCode,
            "WorkerHelloTimeout",
          );
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
          emitReady(worker);
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
          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "running");

          yield* manager.cancel({ runId });
          yield* worker.awaitKills(1);

          // The worker finished on its own before the signal landed. That is a
          // real result and must not be discarded as a cancellation.
          worker.emitStdout(JSON.stringify({ type: "done", status: "completed" }));
          worker.exit(0);
          yield* awaitTerminal(manager, runId);

          assert.strictEqual((yield* manager.get({ runId })).summary.state, "completed");
        }),
      );
    }),
  );

  it.effect("rate-limits a metric burst without dropping durable steps", () =>
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
          emitReady(worker);
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
          // `done` is the protocol receipt that the sequential stdout pump has
          // consumed the whole burst; it also flushes the one pending live batch.
          worker.emitStdout(JSON.stringify({ type: "done", status: "completed" }));
          yield* Deferred.await(flushed);

          // Live transport stays bounded, while a reconnect can replay every
          // semantic step from durable storage.
          assert.strictEqual(received.length, 1);
          assert.strictEqual(received[0], 50);
          const detail = yield* manager.get({ runId });
          assert.deepStrictEqual(
            detail.metrics.map((batch) => batch.step),
            Array.from({ length: 50 }, (_, index) => index + 1),
          );
        }),
      );
    }),
  );

  it.effect("coalesces adjacent metric namespaces without dropping either", () =>
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
          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "running");

          worker.emitStdout(
            JSON.stringify({
              type: "metrics",
              step: 8,
              wallClockMs: 100,
              values: { "train/reward": 0.5, "system/num_tokens": 588 },
            }),
          );
          worker.emitStdout(
            JSON.stringify({
              type: "metrics",
              step: 8,
              wallClockMs: 120,
              values: { "eval/reward": 0.875, "eval/verifier_pass_rate": 0.875 },
            }),
          );
          yield* settle;
          yield* TestClock.adjust("1 second");
          yield* settle;

          const detail = yield* manager.get({ runId });
          assert.deepStrictEqual(detail.metrics, [
            {
              step: 8,
              wallClockMs: 100,
              values: { "train/reward": 0.5, "system/num_tokens": 588 },
            },
            {
              step: 8,
              wallClockMs: 120,
              values: { "eval/reward": 0.875, "eval/verifier_pass_rate": 0.875 },
            },
          ]);
        }),
      );
    }),
  );

  it.effect("does not merge metric values from different steps", () =>
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
          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "running");

          worker.emitStdout(
            JSON.stringify({
              type: "metrics",
              step: 7,
              wallClockMs: 100,
              values: { "train/reward": 0.5 },
            }),
          );
          worker.emitStdout(
            JSON.stringify({
              type: "metrics",
              step: 8,
              wallClockMs: 120,
              values: { "eval/reward": 0.875 },
            }),
          );
          yield* settle;
          yield* TestClock.adjust("1 second");
          yield* settle;

          const detail = yield* manager.get({ runId });
          assert.deepStrictEqual(detail.metrics, [
            {
              step: 7,
              wallClockMs: 100,
              values: { "train/reward": 0.5 },
            },
            {
              step: 8,
              wallClockMs: 120,
              values: { "eval/reward": 0.875 },
            },
          ]);
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
          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "running");

          worker.emitStdout(metrics(1, 42));
          yield* settle;
          yield* TestClock.adjust("1 second");
          yield* settle;

          const events: RlSubscriptionEvent[] = [];
          yield* manager.subscribe({ runId }, (event) => events.push(event));

          assert.strictEqual(events[0]?._tag, "Snapshot");
          const snapshot = events[0];
          assert.isTrue(snapshot?._tag === "Snapshot" && snapshot.metrics.length === 1);

          worker.emitStdout(JSON.stringify({ type: "done", status: "completed" }));
          worker.exit(0);
          yield* awaitTerminal(manager, runId);

          assert.isTrue(events.some((event) => event._tag === "Lifecycle"));
        }),
      );
    }),
  );

  it.effect("keeps a long-lived subscriber through more than 256 metric events", () =>
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
          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "running");

          let metricEvents = 0;
          let completed = false;
          yield* manager.subscribe({ runId }, (event) => {
            if (event._tag === "Metrics") metricEvents += 1;
            if (event._tag === "Lifecycle" && event.summary.state === "completed") completed = true;
          });
          for (let step = 1; step <= 260; step += 1) {
            worker.emitStdout(metrics(step, step));
            yield* TestClock.adjust("500 millis");
            yield* settle;
          }
          worker.emitStdout(JSON.stringify({ type: "done", status: "completed" }));
          worker.exit(0);
          yield* awaitTerminal(manager, runId);

          assert.isAtLeast(metricEvents, 257);
          assert.isTrue(completed);
        }),
      );
    }),
  );

  it.effect("deduplicates a retried start request", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const input = {
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
            requestId: "request_01",
          };
          const first = yield* manager.start(input);
          const retry = yield* manager.start(input);
          assert.strictEqual(retry.runId, first.runId);
          assert.strictEqual(worker.spawnInputs.length, 1);
        }),
      );
    }),
  );

  it.effect("rejects an artifact symlink that resolves outside the run", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const fs = yield* FileSystem.FileSystem;
          const path = yield* Path.Path;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          const spawn = worker.spawnInputs[0];
          const runDirIndex = spawn?.args.indexOf("--run-dir") ?? -1;
          const runDir = spawn?.args[runDirIndex + 1];
          assert.isString(runDir);
          const outside = path.join(path.dirname(runDir!), "outside.json");
          yield* fs.writeFileString(outside, "secret");
          yield* fs.symlink(outside, path.join(runDir!, "escape.json"));

          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "running");
          worker.emitStdout(
            JSON.stringify({ type: "artifact", kind: "summary", path: "escape.json" }),
          );
          yield* awaitTerminal(manager, runId);
          assert.strictEqual(
            (yield* manager.get({ runId })).summary.errorCode,
            "MalformedWorkerMessage",
          );
        }).pipe(Effect.provide(NodeServices.layer)),
      );
    }),
  );

  it.effect("publishes a directory artifact with a server-computed identity", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const fs = yield* FileSystem.FileSystem;
          const path = yield* Path.Path;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          const spawn = worker.spawnInputs[0];
          const runDirIndex = spawn?.args.indexOf("--run-dir") ?? -1;
          const runDir = spawn?.args[runDirIndex + 1];
          assert.isString(runDir);
          yield* fs.makeDirectory(path.join(runDir!, "adapter", "nested"), { recursive: true });
          yield* fs.writeFileString(path.join(runDir!, "adapter", "weights.bin"), "weights");
          yield* fs.writeFileString(path.join(runDir!, "adapter", "nested", "config.json"), "{}");

          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "running");
          const published = yield* Deferred.make<RlArtifactMetadata>();
          yield* manager.subscribe({ runId }, (event) => {
            if (event._tag === "Artifact" && event.artifact.kind === "adapter") {
              Deferred.doneUnsafe(published, Effect.succeed(event.artifact));
            }
          });
          worker.emitStdout(
            JSON.stringify({
              type: "artifact",
              kind: "adapter",
              path: "adapter",
              sha256: "0".repeat(64),
            }),
          );
          const artifact = yield* Deferred.await(published);

          assert.strictEqual(artifact.state, "ready");
          assert.strictEqual(artifact.format, "directory-v1");
          assert.strictEqual(artifact.fileCount, 2);
          assert.strictEqual(artifact.logicalName, "adapter");
          assert.match(artifact.sha256 ?? "", /^[0-9a-f]{64}$/);
          assert.notStrictEqual(artifact.sha256, "0".repeat(64));
          const defaultedPage = yield* manager.listArtifacts({ runId, limit: 0 });
          assert.isAtLeast(defaultedPage.artifacts.length, 1);
        }).pipe(Effect.provide(NodeServices.layer)),
      );
    }),
  );

  it.effect("keeps terminal state recording when the final log cannot be published", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const fs = yield* FileSystem.FileSystem;
          const path = yield* Path.Path;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          const spawn = worker.spawnInputs[0];
          const runDirIndex = spawn?.args.indexOf("--run-dir") ?? -1;
          const runDir = spawn?.args[runDirIndex + 1];
          assert.isString(runDir);
          yield* fs.makeDirectory(path.join(runDir!, "worker.log"));

          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "running");
          worker.emitStdout(JSON.stringify({ type: "done", status: "completed" }));
          worker.exit(0);

          assert.strictEqual(yield* awaitTerminal(manager, runId), "completed");
          assert.strictEqual((yield* manager.get({ runId })).summary.state, "completed");
        }).pipe(Effect.provide(NodeServices.layer)),
      );
    }),
  );

  it.effect("fails explicitly when the resolved manifest artifact cannot be published", () =>
    Effect.gen(function* () {
      const worker = new FakeWorkerProcess();
      yield* withWorker(
        worker,
        Effect.gen(function* () {
          const manager = yield* RlManager.RlManager;
          const fs = yield* FileSystem.FileSystem;
          const path = yield* Path.Path;
          const { runId } = yield* manager.start({
            projectId: "proj_01",
            experimentId: "fake",
            seed: 7,
          });
          const spawn = worker.spawnInputs[0];
          const runDirIndex = spawn?.args.indexOf("--run-dir") ?? -1;
          const runDir = spawn?.args[runDirIndex + 1];
          assert.isString(runDir);
          yield* fs.makeDirectory(path.join(runDir!, "manifest.json"));

          emitReady(worker);
          yield* awaitState(manager, runId, (state) => state === "failed");
          worker.exit(1);
          yield* settle;

          const detail = yield* manager.get({ runId });
          assert.strictEqual(detail.summary.state, "failed");
          assert.strictEqual(detail.summary.errorCode, "RunnerException");
          assert.include(detail.summary.errorMessage ?? "", "resolved manifest artifact");
        }).pipe(Effect.provide(NodeServices.layer)),
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
          emitReady(worker);
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

          // The invalid definition is visible before any process existed.
          const { runs } = yield* manager.list({ projectId: "proj_01" });
          assert.strictEqual(runs[0]?.state, "failed");
          assert.strictEqual(runs[0]?.errorCode, "InvalidExperiment");
          assert.strictEqual(worker.spawnInputs.length, 0);
        }),
      );
    }),
  );
});
