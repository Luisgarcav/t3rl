/**
 * RlManager - supervises T3RL runs.
 *
 * Owns the worker process, translates its output into lifecycle facts through
 * the pure state machine, fans events out to subscribers, and bounds every
 * buffer it holds. Modeled on terminal/Manager.ts.
 *
 * @module RlManager
 */
import {
  RlCapabilityReport,
  RlRunNotFoundError,
  RlRunStartError,
  type RlArtifactMetadata,
  type RlErrorCode,
  type RlMetricBatch,
  type RlResolvedManifest,
  type RlRunState,
  type RlRunSummary,
  type RlSubscriptionEvent,
  RL_WORKER_PROTOCOL_VERSION,
} from "@t3tools/contracts";
import * as Context from "effect/Context";
import * as Crypto from "effect/Crypto";
import * as DateTime from "effect/DateTime";
import * as Duration from "effect/Duration";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Fiber from "effect/Fiber";
import * as Layer from "effect/Layer";
import * as Path from "effect/Path";
import * as Scope from "effect/Scope";
import * as Stream from "effect/Stream";
import * as SynchronizedRef from "effect/SynchronizedRef";

import { HostProcessArchitecture, HostProcessPlatform } from "@t3tools/shared/hostProcess";

import * as ServerConfig from "../config.ts";
import * as Artifacts from "./Artifacts.ts";
import { Capabilities } from "./Capabilities.ts";
import { Experiments } from "./Experiments.ts";
import * as Lifecycle from "./Lifecycle.ts";
import { RunStore } from "./RunStore.ts";
import { WorkerSpawner, type WorkerProcess } from "./WorkerSpawner.ts";
import * as WorkerProtocol from "./WorkerProtocol.ts";

/** How long a cancelled worker gets to exit before SIGKILL. */
export const CANCEL_GRACE = Duration.seconds(5);
/** Retained stderr, flushed to the run log on exit. Oldest bytes are dropped. */
export const MAX_STDERR_BYTES = 64 * 1024;
/** Metric batches that reach subscribers and the store, per second. */
export const MAX_METRIC_BATCHES_PER_SECOND = 2;
const METRIC_FLUSH_INTERVAL = Duration.millis(1000 / MAX_METRIC_BATCHES_PER_SECOND);
/** A subscriber that cannot keep up is dropped rather than buffered forever. */
const MAX_SUBSCRIBER_BACKLOG = 256;

export interface RlRunDetail {
  readonly summary: RlRunSummary;
  readonly manifest: RlResolvedManifest | null;
  readonly artifacts: ReadonlyArray<RlArtifactMetadata>;
}

export interface StartRunInput {
  readonly projectId: string;
  readonly experimentId: string;
  readonly seed: number;
}

type Subscriber = (event: RlSubscriptionEvent) => void;

interface RunRecord {
  readonly runId: string;
  readonly projectId: string;
  readonly experimentId: string;
  state: RlRunState;
  worker: WorkerProcess | null;
  /** Set once `done` arrives, so process exit is not re-reported as a failure. */
  doneSeen: boolean;
  cancelRequested: boolean;
  metricSeq: number;
  pendingBatch: RlMetricBatch | null;
  flushScheduled: boolean;
  stderrBytes: number;
  stderrChunks: string[];
  subscribers: Set<Subscriber>;
  fibers: Fiber.Fiber<unknown, unknown>[];
}

export interface RlManagerShape {
  readonly capabilities: () => Effect.Effect<RlCapabilityReport>;
  readonly start: (
    input: StartRunInput,
  ) => Effect.Effect<{ readonly runId: string }, RlRunStartError>;
  readonly cancel: (input: {
    readonly runId: string;
  }) => Effect.Effect<{ readonly state: RlRunState }, RlRunNotFoundError>;
  readonly list: (input: {
    readonly projectId: string;
    readonly limit?: number | undefined;
  }) => Effect.Effect<{ readonly runs: ReadonlyArray<RlRunSummary> }>;
  readonly get: (input: {
    readonly runId: string;
  }) => Effect.Effect<RlRunDetail, RlRunNotFoundError>;
  /** Delivers a snapshot, then live events, until the returned unsubscribe runs. */
  readonly subscribe: (
    input: { readonly runId: string },
    onEvent: Subscriber,
  ) => Effect.Effect<() => void, RlRunNotFoundError>;
  readonly sweepInterruptedRuns: () => Effect.Effect<number>;
}

export class RlManager extends Context.Service<RlManager, RlManagerShape>()(
  "t3/rl/Manager/RlManager",
) {}

const DEFAULT_LIST_LIMIT = 50;

const makeManager = Effect.gen(function* () {
  const store = yield* RunStore;
  const spawner = yield* WorkerSpawner;
  const capabilities = yield* Capabilities;
  const experiments = yield* Experiments;
  const config = yield* ServerConfig.ServerConfig;
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const crypto = yield* Crypto.Crypto;
  const hostPlatform = yield* HostProcessPlatform;
  const hostArchitecture = yield* HostProcessArchitecture;
  const scope = yield* Effect.scope;

  const runs = yield* SynchronizedRef.make(new Map<string, RunRecord>());

  const now = DateTime.now.pipe(Effect.map((instant) => DateTime.formatIso(instant)));

  const getRecord = (runId: string) =>
    SynchronizedRef.get(runs).pipe(Effect.map((map) => map.get(runId) ?? null));

  const publish = (record: RunRecord, event: RlSubscriptionEvent): void => {
    for (const subscriber of record.subscribers) {
      subscriber(event);
    }
  };

  const summaryOf = (runId: string) =>
    store.getRun({ runId }).pipe(
      Effect.map((detail) => detail.summary),
      Effect.orDie,
    );

  /**
   * The single writer of run state. Every observation goes through the pure
   * machine first; a rejected transition is logged and dropped rather than
   * written, so the store can never hold a state the machine forbids.
   */
  const applyEvent = (
    runId: string,
    event: Lifecycle.RlLifecycleEvent,
    detail?: { readonly code?: RlErrorCode; readonly message?: string },
  ) =>
    Effect.gen(function* () {
      const record = yield* getRecord(runId);
      if (record === null) return;

      const result = Lifecycle.transition(record.state, event);
      if (result._tag === "Rejected") {
        yield* Effect.logDebug("rl lifecycle transition rejected", {
          runId,
          from: record.state,
          event: event._tag,
          reason: result.reason,
        });
        return;
      }

      record.state = result.state;
      const at = yield* now;
      yield* store
        .updateState({
          runId,
          state: result.state,
          at,
          ...(detail?.code === undefined ? {} : { errorCode: detail.code }),
          ...(detail?.message === undefined ? {} : { errorMessage: detail.message.slice(0, 2048) }),
          ...(record.worker === null ? {} : { workerPid: record.worker.pid }),
        })
        .pipe(Effect.orDie);

      const summary = yield* summaryOf(runId);
      publish(record, { _tag: "Lifecycle", summary });
    });

  // ---- metrics -----------------------------------------------------------

  const flushMetrics = (runId: string) =>
    Effect.gen(function* () {
      const record = yield* getRecord(runId);
      if (record === null) return;
      const batch = record.pendingBatch;
      record.pendingBatch = null;
      record.flushScheduled = false;
      if (batch === null) return;

      record.metricSeq += 1;
      const at = yield* now;
      yield* store.appendMetrics({ runId, seq: record.metricSeq, batch, at }).pipe(Effect.orDie);
      publish(record, { _tag: "Metrics", batch });
    });

  /**
   * Holds one pending batch and a single scheduled flush. A burst replaces the
   * pending value instead of queueing, so an over-eager worker cannot grow the
   * server's memory.
   */
  const offerMetrics = (runId: string, batch: RlMetricBatch) =>
    Effect.gen(function* () {
      const record = yield* getRecord(runId);
      if (record === null) return;
      record.pendingBatch = batch;
      if (record.flushScheduled) return;
      record.flushScheduled = true;
      const fiber = yield* Effect.forkIn(
        Effect.sleep(METRIC_FLUSH_INTERVAL).pipe(Effect.andThen(flushMetrics(runId))),
        scope,
      );
      record.fibers.push(fiber);
    });

  // ---- stderr ------------------------------------------------------------

  const appendStderr = (record: RunRecord, line: string): void => {
    const chunk = `${line}\n`;
    record.stderrChunks.push(chunk);
    record.stderrBytes += Buffer.byteLength(chunk, "utf8");
    while (record.stderrBytes > MAX_STDERR_BYTES && record.stderrChunks.length > 1) {
      const dropped = record.stderrChunks.shift();
      if (dropped === undefined) break;
      record.stderrBytes -= Buffer.byteLength(dropped, "utf8");
    }
  };

  const writeStderrArtifact = (record: RunRecord) =>
    Effect.gen(function* () {
      if (record.stderrChunks.length === 0) return;
      const runRoot = Artifacts.runDirectory({
        rlRunsDir: config.rlRunsDir,
        runId: record.runId,
      });
      if (runRoot === null) return;
      const relativePath = "worker.log";
      const target = path.join(runRoot, relativePath);
      const contents = record.stderrChunks.join("");
      yield* fs.writeFileString(target, contents).pipe(Effect.orDie);
      const at = yield* now;
      yield* store
        .recordArtifact({
          runId: record.runId,
          kind: "log",
          relativePath,
          bytes: Buffer.byteLength(contents, "utf8"),
          contentType: "text/plain",
          producedAt: at,
        })
        .pipe(Effect.orDie);
    });

  // ---- worker output -----------------------------------------------------

  const handleArtifact = (record: RunRecord, kind: RlArtifactMetadata["kind"], relative: string) =>
    Effect.gen(function* () {
      const resolved = Artifacts.resolveArtifactPath({
        rlRunsDir: config.rlRunsDir,
        runId: record.runId,
        relativePath: relative,
      });
      if (resolved === null) {
        yield* applyEvent(
          record.runId,
          { _tag: "WorkerFailed", code: "MalformedWorkerMessage", message: "artifact escaped run" },
          { code: "MalformedWorkerMessage", message: `artifact path rejected: ${relative}` },
        );
        return;
      }
      const info = yield* fs.stat(resolved).pipe(Effect.option);
      const bytes = info._tag === "Some" ? Number(info.value.size) : 0;
      const at = yield* now;
      yield* store
        .recordArtifact({
          runId: record.runId,
          kind,
          relativePath: relative,
          bytes,
          contentType: "application/octet-stream",
          producedAt: at,
        })
        .pipe(Effect.orDie);
    });

  const handleLine = (runId: string, line: string) =>
    Effect.gen(function* () {
      const record = yield* getRecord(runId);
      if (record === null) return;

      const decoded = WorkerProtocol.decodeWorkerLine(line);
      if (decoded._tag === "Ignored") return;
      if (decoded._tag === "Failure") {
        yield* applyEvent(
          runId,
          { _tag: "WorkerFailed", code: decoded.code, message: decoded.detail },
          { code: decoded.code, message: decoded.detail },
        );
        return;
      }

      switch (decoded.message._tag) {
        case "Hello":
          yield* applyEvent(runId, { _tag: "WorkerReady" });
          return;
        case "Manifest":
          return;
        case "Metrics":
          yield* offerMetrics(runId, decoded.message.batch);
          return;
        case "Artifact":
          yield* handleArtifact(record, decoded.message.kind, decoded.message.path);
          return;
        case "Error":
          yield* applyEvent(
            runId,
            { _tag: "WorkerFailed", code: decoded.message.code, message: decoded.message.detail },
            { code: decoded.message.code, message: decoded.message.detail },
          );
          return;
        case "Done":
          record.doneSeen = true;
          yield* flushMetrics(runId);
          yield* applyEvent(runId, { _tag: "WorkerDone", success: decoded.message.success });
          return;
      }
    });

  const watchExit = (record: RunRecord, worker: WorkerProcess) =>
    Effect.gen(function* () {
      const code = yield* worker.exitCode.pipe(Effect.orElseSucceed(() => null));
      yield* flushMetrics(record.runId);
      yield* writeStderrArtifact(record);

      if (record.doneSeen) return;

      if (record.cancelRequested) {
        yield* applyEvent(record.runId, { _tag: "WorkerStopped" });
        return;
      }

      // Silence is never success: without `done`, an exit is a failure even at
      // code 0, because the worker never claimed to have finished.
      yield* applyEvent(
        record.runId,
        { _tag: "WorkerFailed", code: "WorkerExited", message: `worker exited with ${code}` },
        { code: "WorkerExited", message: `worker exited without done (code ${String(code)})` },
      );
    });

  // ---- public surface ----------------------------------------------------

  const start: RlManagerShape["start"] = (input) =>
    Effect.gen(function* () {
      const uuid = yield* crypto.randomUUIDv4.pipe(Effect.orDie);
      const runId = `run_${uuid.replace(/-/g, "").slice(0, 24)}`;
      const requestedAt = yield* now;

      yield* store
        .insertRequested({
          runId,
          projectId: input.projectId,
          experimentId: input.experimentId,
          requestedAt,
        })
        .pipe(Effect.orDie);

      const record: RunRecord = {
        runId,
        projectId: input.projectId,
        experimentId: input.experimentId,
        state: "requested",
        worker: null,
        doneSeen: false,
        cancelRequested: false,
        metricSeq: 0,
        pendingBatch: null,
        flushScheduled: false,
        stderrBytes: 0,
        stderrChunks: [],
        subscribers: new Set(),
        fibers: [],
      };
      yield* SynchronizedRef.update(runs, (map) => new Map(map).set(runId, record));

      yield* applyEvent(runId, { _tag: "PreparationStarted" });

      // Everything that can refuse the run happens before a process exists.
      const failStart = (error: RlRunStartError) =>
        applyEvent(
          runId,
          { _tag: "WorkerFailed", code: error.code, message: error.detail },
          { code: error.code, message: error.detail },
        ).pipe(Effect.andThen(Effect.fail(error)));

      const definition = yield* experiments
        .resolve({ experimentId: input.experimentId })
        .pipe(Effect.catchTag("RlRunStartError", failStart));

      const python = yield* capabilities
        .resolvePython()
        .pipe(Effect.catchTag("RlRunStartError", failStart));

      const runRoot = Artifacts.runDirectory({
        rlRunsDir: config.rlRunsDir,
        runId,
      });
      if (runRoot === null) {
        return yield* failStart(
          new RlRunStartError({ code: "RunnerUnavailable", detail: "invalid run directory" }),
        );
      }
      yield* fs.makeDirectory(runRoot, { recursive: true }).pipe(Effect.orDie);

      const manifest: RlResolvedManifest = {
        experimentId: definition.experimentId,
        runnerId: definition.runnerId,
        runnerVersion: "0.1.0",
        protocolVersion: RL_WORKER_PROTOCOL_VERSION,
        seed: input.seed,
        effectiveConfig: definition.config,
        sourceRevision: null,
        sourceDirty: false,
        pythonExecutable: python.executable,
        pythonVersion: python.version,
        environmentFingerprint: `${python.executable}@${python.version}`,
        instrumentationLevel: definition.instrumentationLevel,
        hardwareSummary: `${hostPlatform}/${hostArchitecture}`,
      };
      yield* store.setManifest({ runId, manifest }).pipe(Effect.orDie);

      const worker = yield* spawner
        .spawn({
          command: python.executable,
          args: [
            definition.entrypoint,
            "--scenario",
            definition.scenario,
            "--run-dir",
            runRoot,
            "--seed",
            String(input.seed),
          ],
          cwd: config.baseDir,
          env: { T3RL_RUN_ID: runId, PATH: process.env["PATH"] ?? "" },
        })
        .pipe(
          Effect.provideService(Scope.Scope, scope),
          Effect.catchTag("RlWorkerSpawnError", (cause) =>
            failStart(new RlRunStartError({ code: "RunnerUnavailable", detail: cause.message })),
          ),
        );

      record.worker = worker;

      const stdoutFiber = yield* Effect.forkIn(
        worker.stdoutLines.pipe(
          Stream.runForEach((line) => handleLine(runId, line)),
          Effect.catchCause(() => Effect.void),
        ),
        scope,
      );
      const stderrFiber = yield* Effect.forkIn(
        worker.stderrLines.pipe(
          Stream.runForEach((line) => Effect.sync(() => appendStderr(record, line))),
          Effect.catchCause(() => Effect.void),
        ),
        scope,
      );
      const exitFiber = yield* Effect.forkIn(watchExit(record, worker), scope);
      record.fibers.push(stdoutFiber, stderrFiber, exitFiber);

      return { runId };
    });

  const cancel: RlManagerShape["cancel"] = (input) =>
    Effect.gen(function* () {
      const record = yield* getRecord(input.runId);
      if (record === null) {
        const stored = yield* store
          .getRun({ runId: input.runId })
          .pipe(Effect.orDie, Effect.option);
        if (stored._tag === "None") {
          return yield* new RlRunNotFoundError({ runId: input.runId });
        }
        return { state: stored.value.summary.state };
      }

      if (
        Lifecycle.transition(record.state, { _tag: "CancellationRequested" })._tag === "Rejected"
      ) {
        // Already terminal: cancelling is a no-op that reports where it landed.
        return { state: record.state };
      }

      record.cancelRequested = true;
      yield* applyEvent(input.runId, { _tag: "CancellationRequested" });

      const worker = record.worker;
      if (worker === null) {
        yield* applyEvent(input.runId, { _tag: "WorkerStopped" });
        return { state: record.state };
      }

      yield* worker.kill("SIGTERM");
      const escalation = yield* Effect.forkIn(
        Effect.sleep(CANCEL_GRACE).pipe(
          Effect.andThen(
            Effect.gen(function* () {
              const current = yield* getRecord(input.runId);
              // Only escalate if the worker still has not gone away.
              if (current === null || current.state !== "cancelling") return;
              yield* worker.kill("SIGKILL");
            }),
          ),
        ),
        scope,
      );
      record.fibers.push(escalation);

      return { state: record.state };
    });

  const list: RlManagerShape["list"] = (input) =>
    store.listRuns({ projectId: input.projectId, limit: input.limit ?? DEFAULT_LIST_LIMIT }).pipe(
      Effect.map((runsList) => ({ runs: runsList })),
      Effect.orDie,
    );

  const get: RlManagerShape["get"] = (input) =>
    Effect.gen(function* () {
      const detail = yield* store
        .getRun({ runId: input.runId })
        .pipe(Effect.catchTag("PersistenceSqlError", Effect.orDie));
      const artifacts = yield* store.listArtifacts({ runId: input.runId }).pipe(Effect.orDie);
      return { summary: detail.summary, manifest: detail.manifest, artifacts };
    });

  const subscribe: RlManagerShape["subscribe"] = (input, onEvent) =>
    Effect.gen(function* () {
      const detail = yield* get(input);
      let delivered = 0;
      const bounded: Subscriber = (event) => {
        if (delivered >= MAX_SUBSCRIBER_BACKLOG) return;
        delivered += 1;
        onEvent(event);
      };

      // Snapshot first, then live: a reconnecting client rebuilds current state
      // without replaying the run and without creating a second one.
      onEvent({
        _tag: "Snapshot",
        summary: detail.summary,
        manifest: detail.manifest,
        artifacts: detail.artifacts,
      });

      const record = yield* getRecord(input.runId);
      if (record === null) {
        return () => {};
      }
      record.subscribers.add(bounded);
      return () => {
        record.subscribers.delete(bounded);
      };
    });

  const sweepInterruptedRuns: RlManagerShape["sweepInterruptedRuns"] = () =>
    Effect.gen(function* () {
      const at = yield* now;
      const count = yield* store.markActiveAsInterrupted({ at }).pipe(Effect.orDie);
      if (count > 0) {
        yield* Effect.logInfo("marked interrupted RL runs after restart", { count });
      }
      return count;
    });

  yield* sweepInterruptedRuns();

  return RlManager.of({
    capabilities: () => capabilities.report(),
    start,
    cancel,
    list,
    get,
    subscribe,
    sweepInterruptedRuns,
  });
});

export const RlManagerLive = Layer.effect(RlManager, makeManager);
