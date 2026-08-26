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
  RL_MAX_RUN_ARTIFACTS,
  RL_MAX_SNAPSHOT_METRIC_BATCHES,
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
import { SourceEvidence } from "./SourceEvidence.ts";
import { WorkerSpawner, type WorkerProcess } from "./WorkerSpawner.ts";
import * as WorkerProtocol from "./WorkerProtocol.ts";

/** How long a cancelled worker gets to exit before SIGKILL. */
export const CANCEL_GRACE = Duration.seconds(5);
/** The worker must announce its protocol promptly after spawn. */
export const STARTUP_TIMEOUT = Duration.seconds(15);
/** Any valid protocol message resets this liveness deadline. */
export const WORKER_STALL_TIMEOUT = Duration.seconds(60);
/** Retained stderr, flushed to the run log on exit. Oldest bytes are dropped. */
export const MAX_STDERR_BYTES = 64 * 1024;
/** Metric batches that reach subscribers and the store, per second. */
export const MAX_METRIC_BATCHES_PER_SECOND = 2;
const METRIC_FLUSH_INTERVAL = Duration.millis(1000 / MAX_METRIC_BATCHES_PER_SECOND);
export const MAX_ARTIFACT_BYTES = 256 * 1024 * 1024;
export const MAX_RUN_ARTIFACT_BYTES = 512 * 1024 * 1024;

export interface RlRunDetail {
  readonly summary: RlRunSummary;
  readonly manifest: RlResolvedManifest | null;
  readonly artifacts: ReadonlyArray<RlArtifactMetadata>;
  readonly metrics: ReadonlyArray<RlMetricBatch>;
}

export interface StartRunInput {
  readonly projectId: string;
  readonly experimentId: string;
  readonly seed: number;
  readonly requestId?: string | undefined;
}

type Subscriber = (event: RlSubscriptionEvent) => void;

interface RunRecord {
  readonly runId: string;
  readonly projectId: string;
  readonly experimentId: string;
  state: RlRunState;
  worker: WorkerProcess | null;
  protocolPhase: "awaiting-hello" | "awaiting-manifest" | "ready" | "done";
  doneResult: boolean | null;
  cancelRequested: boolean;
  metricSeq: number;
  pendingBatch: RlMetricBatch | null;
  flushScheduled: boolean;
  stderr: Buffer;
  manifestBase: RlResolvedManifest | null;
  artifactBytes: number;
  artifactPaths: Set<string>;
  watchdog: Fiber.Fiber<unknown, unknown> | null;
  deadline: Fiber.Fiber<unknown, unknown> | null;
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
  const sourceEvidence = yield* SourceEvidence;
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
      yield* Effect.forkIn(
        Effect.sleep(METRIC_FLUSH_INTERVAL).pipe(Effect.andThen(flushMetrics(runId))),
        scope,
      );
    });

  // ---- stderr ------------------------------------------------------------

  const appendStderr = (record: RunRecord, line: string): void => {
    const chunk = Buffer.from(`${line}\n`, "utf8");
    record.stderr = Buffer.concat([record.stderr, chunk]).subarray(-MAX_STDERR_BYTES);
  };

  const writeStderrArtifact = (record: RunRecord) =>
    Effect.gen(function* () {
      const runRoot = Artifacts.runDirectory({
        rlRunsDir: config.rlRunsDir,
        runId: record.runId,
      });
      if (runRoot === null) return;
      const relativePath = "worker.log";
      const target = path.join(runRoot, relativePath);
      yield* fs.writeFile(target, record.stderr).pipe(Effect.orDie);
      const at = yield* now;
      const artifact = yield* store
        .recordArtifact({
          runId: record.runId,
          kind: "log",
          relativePath,
          bytes: record.stderr.byteLength,
          contentType: "text/plain",
          producedAt: at,
        })
        .pipe(Effect.orDie);
      record.artifactPaths.add(relativePath);
      record.artifactBytes += artifact.bytes;
      publish(record, { _tag: "Artifact", artifact });
    });

  const writeManifestArtifact = (record: RunRecord, manifest: RlResolvedManifest) =>
    Effect.gen(function* () {
      const runRoot = Artifacts.runDirectory({
        rlRunsDir: config.rlRunsDir,
        runId: record.runId,
      });
      if (runRoot === null) return;
      const relativePath = "manifest.json";
      // @effect-diagnostics-next-line preferSchemaOverJson:off - durable human-readable artifact.
      const contents = `${JSON.stringify(manifest, null, 2)}\n`;
      yield* fs.writeFileString(path.join(runRoot, relativePath), contents).pipe(Effect.orDie);
      const at = yield* now;
      const artifact = yield* store
        .recordArtifact({
          runId: record.runId,
          kind: "manifest",
          relativePath,
          bytes: Buffer.byteLength(contents, "utf8"),
          contentType: "application/json",
          producedAt: at,
        })
        .pipe(Effect.orDie);
      record.artifactPaths.add(relativePath);
      record.artifactBytes += artifact.bytes;
      publish(record, { _tag: "Artifact", artifact });
    });

  const stopWorker = (record: RunRecord) =>
    Effect.gen(function* () {
      const worker = record.worker;
      if (worker === null) return;
      yield* worker.kill("SIGTERM");
      yield* Effect.forkIn(
        Effect.sleep(CANCEL_GRACE).pipe(
          Effect.andThen(
            Effect.gen(function* () {
              const current = yield* getRecord(record.runId);
              if (current === null || current.worker !== worker) return;
              yield* worker.kill("SIGKILL");
            }),
          ),
        ),
        scope,
      );
    });

  const failRun = (record: RunRecord, code: RlErrorCode, message: string): Effect.Effect<void> =>
    Effect.gen(function* () {
      if (
        Lifecycle.transition(record.state, { _tag: "WorkerFailed", code, message })._tag ===
        "Rejected"
      ) {
        return;
      }
      yield* applyEvent(record.runId, { _tag: "WorkerFailed", code, message }, { code, message });
      yield* stopWorker(record);
    });

  const armWatchdog = (
    record: RunRecord,
    timeout: Duration.Input,
    code: RlErrorCode,
    message: string,
  ) =>
    Effect.gen(function* () {
      if (record.watchdog !== null) {
        yield* Fiber.interrupt(record.watchdog).pipe(Effect.ignore);
      }
      record.watchdog = yield* Effect.forkIn(
        Effect.sleep(timeout).pipe(Effect.andThen(failRun(record, code, message))),
        scope,
      );
    });

  const clearTimers = (record: RunRecord) =>
    Effect.all(
      [record.watchdog, record.deadline]
        .filter((fiber): fiber is Fiber.Fiber<unknown, unknown> => fiber !== null)
        .map((fiber) => Fiber.interrupt(fiber).pipe(Effect.ignore)),
      { discard: true },
    );

  // ---- worker output -----------------------------------------------------

  const handleArtifact = (record: RunRecord, kind: RlArtifactMetadata["kind"], relative: string) =>
    Effect.gen(function* () {
      if (
        record.artifactPaths.has(relative) ||
        record.artifactPaths.size >= RL_MAX_RUN_ARTIFACTS - 1
      ) {
        yield* failRun(
          record,
          "MalformedWorkerMessage",
          "worker announced too many or duplicate artifacts",
        );
        return;
      }
      const resolved = Artifacts.resolveArtifactPath({
        rlRunsDir: config.rlRunsDir,
        runId: record.runId,
        relativePath: relative,
      });
      if (resolved === null) {
        yield* failRun(record, "MalformedWorkerMessage", `artifact path rejected: ${relative}`);
        return;
      }

      const runRoot = Artifacts.runDirectory({ rlRunsDir: config.rlRunsDir, runId: record.runId });
      if (runRoot === null) return;
      const canonical = yield* Effect.all({
        root: fs.realPath(runRoot).pipe(Effect.option),
        file: fs.realPath(resolved).pipe(Effect.option),
      });
      if (canonical.root._tag === "None" || canonical.file._tag === "None") {
        yield* failRun(record, "MalformedWorkerMessage", `artifact does not exist: ${relative}`);
        return;
      }
      const relativeCanonical = path.relative(canonical.root.value, canonical.file.value);
      if (
        relativeCanonical === "" ||
        relativeCanonical.startsWith("..") ||
        path.isAbsolute(relativeCanonical)
      ) {
        yield* failRun(record, "MalformedWorkerMessage", `artifact escaped its run: ${relative}`);
        return;
      }
      const info = yield* fs.stat(canonical.file.value).pipe(Effect.option);
      if (info._tag === "None" || info.value.type !== "File") {
        yield* failRun(
          record,
          "MalformedWorkerMessage",
          `artifact is not a regular file: ${relative}`,
        );
        return;
      }
      const bytes = Number(info.value.size);
      if (
        !Number.isSafeInteger(bytes) ||
        bytes < 0 ||
        bytes > MAX_ARTIFACT_BYTES ||
        record.artifactBytes + bytes > MAX_RUN_ARTIFACT_BYTES
      ) {
        yield* failRun(
          record,
          "MalformedWorkerMessage",
          `artifact exceeded the run budget: ${relative}`,
        );
        return;
      }
      const at = yield* now;
      const contentType =
        kind === "model" ? "application/zip" : kind === "log" ? "text/plain" : "application/json";
      const artifact = yield* store
        .recordArtifact({
          runId: record.runId,
          kind,
          relativePath: relative,
          bytes,
          contentType,
          producedAt: at,
        })
        .pipe(Effect.orDie);
      record.artifactPaths.add(relative);
      record.artifactBytes += bytes;
      publish(record, { _tag: "Artifact", artifact });
    });

  const handleLine = (runId: string, line: string) =>
    Effect.gen(function* () {
      const record = yield* getRecord(runId);
      if (record === null) return;

      const decoded = WorkerProtocol.decodeWorkerLine(line);
      if (decoded._tag === "Ignored") return;
      if (decoded._tag === "Failure") {
        yield* failRun(record, decoded.code, decoded.detail);
        return;
      }

      switch (decoded.message._tag) {
        case "Hello": {
          if (
            record.protocolPhase !== "awaiting-hello" ||
            record.manifestBase === null ||
            decoded.message.runner !== record.manifestBase.runnerId ||
            decoded.message.runnerVersion !== record.manifestBase.runnerVersion
          ) {
            yield* failRun(
              record,
              "ProtocolIncompatible",
              "worker hello did not match the resolved runner",
            );
            return;
          }
          record.protocolPhase = "awaiting-manifest";
          yield* applyEvent(runId, { _tag: "WorkerReady" });
          yield* armWatchdog(
            record,
            WORKER_STALL_TIMEOUT,
            "WorkerStalled",
            "worker stalled before its manifest",
          );
          return;
        }
        case "Manifest": {
          if (record.protocolPhase !== "awaiting-manifest" || record.manifestBase === null) {
            yield* failRun(record, "MalformedWorkerMessage", "worker manifest was out of order");
            return;
          }
          const manifest: RlResolvedManifest = {
            ...record.manifestBase,
            effectiveConfig: {
              ...record.manifestBase.effectiveConfig,
              ...decoded.message.values,
            },
          };
          yield* store.setManifest({ runId, manifest }).pipe(Effect.orDie);
          record.protocolPhase = "ready";
          yield* writeManifestArtifact(record, manifest);
          publish(record, { _tag: "Manifest", manifest });
          yield* armWatchdog(
            record,
            WORKER_STALL_TIMEOUT,
            "WorkerStalled",
            "worker stopped producing output",
          );
          return;
        }
        case "Metrics": {
          if (record.protocolPhase !== "ready") {
            yield* failRun(record, "MalformedWorkerMessage", "worker metrics were out of order");
            return;
          }
          yield* offerMetrics(runId, decoded.message.batch);
          yield* armWatchdog(
            record,
            WORKER_STALL_TIMEOUT,
            "WorkerStalled",
            "worker stopped producing output",
          );
          return;
        }
        case "Artifact": {
          if (record.protocolPhase !== "ready") {
            yield* failRun(record, "MalformedWorkerMessage", "worker artifact was out of order");
            return;
          }
          yield* handleArtifact(record, decoded.message.kind, decoded.message.path);
          if (record.state === "running" || record.state === "cancelling") {
            yield* armWatchdog(
              record,
              WORKER_STALL_TIMEOUT,
              "WorkerStalled",
              "worker stopped producing output",
            );
          }
          return;
        }
        case "Error": {
          if (record.protocolPhase === "awaiting-hello" || record.protocolPhase === "done") {
            yield* failRun(record, "MalformedWorkerMessage", "worker error was out of order");
            return;
          }
          yield* failRun(record, decoded.message.code, decoded.message.detail);
          return;
        }
        case "Done": {
          if (record.protocolPhase !== "ready") {
            yield* failRun(record, "MalformedWorkerMessage", "worker done was out of order");
            return;
          }
          record.protocolPhase = "done";
          record.doneResult = decoded.message.success;
          yield* flushMetrics(runId);
          yield* armWatchdog(
            record,
            WORKER_STALL_TIMEOUT,
            "WorkerStalled",
            "worker did not exit after done",
          );
          return;
        }
      }
    });

  const watchExit = (
    record: RunRecord,
    worker: WorkerProcess,
    stdoutFiber: Fiber.Fiber<unknown, unknown>,
    stderrFiber: Fiber.Fiber<unknown, unknown>,
  ) =>
    Effect.gen(function* () {
      const code = yield* worker.exitCode.pipe(Effect.orElseSucceed(() => null));
      // Child exit can race the stream pumps. Drain both before interpreting
      // `done` or writing the final log so terminal state reflects all output.
      yield* Effect.all([Fiber.await(stdoutFiber), Fiber.await(stderrFiber)], {
        discard: true,
      });
      yield* clearTimers(record);
      yield* flushMetrics(record.runId);
      yield* writeStderrArtifact(record);

      if (record.doneResult === true && code === 0) {
        yield* applyEvent(record.runId, { _tag: "WorkerDone", success: true });
      } else if (record.doneResult !== null) {
        const message =
          record.doneResult === false
            ? "worker reported a failed run"
            : `worker reported completion but exited with ${String(code)}`;
        yield* applyEvent(
          record.runId,
          { _tag: "WorkerFailed", code: "RunnerException", message },
          { code: "RunnerException", message },
        );
      } else if (record.cancelRequested) {
        yield* applyEvent(record.runId, { _tag: "WorkerStopped" });
      } else {
        // Silence is never success: without `done`, an exit is a failure even
        // at code 0, because the worker never claimed to have finished.
        yield* applyEvent(
          record.runId,
          { _tag: "WorkerFailed", code: "WorkerExited", message: `worker exited with ${code}` },
          { code: "WorkerExited", message: `worker exited without done (code ${String(code)})` },
        );
      }
      yield* SynchronizedRef.update(runs, (map) => {
        const next = new Map(map);
        next.delete(record.runId);
        return next;
      });
    });

  // ---- public surface ----------------------------------------------------

  const start: RlManagerShape["start"] = (input) =>
    Effect.gen(function* () {
      const uuid = yield* crypto.randomUUIDv4.pipe(Effect.orDie);
      const proposedRunId = `run_${uuid.replace(/-/g, "").slice(0, 24)}`;
      const requestedAt = yield* now;

      const requested = yield* store
        .insertRequested({
          runId: proposedRunId,
          projectId: input.projectId,
          experimentId: input.experimentId,
          requestedAt,
          ...(input.requestId === undefined ? {} : { requestId: input.requestId }),
        })
        .pipe(Effect.orDie);
      if (!requested.inserted) return { runId: requested.runId };
      const runId = requested.runId;

      const record: RunRecord = {
        runId,
        projectId: input.projectId,
        experimentId: input.experimentId,
        state: "requested",
        worker: null,
        protocolPhase: "awaiting-hello",
        doneResult: null,
        cancelRequested: false,
        metricSeq: 0,
        pendingBatch: null,
        flushScheduled: false,
        stderr: Buffer.alloc(0),
        manifestBase: null,
        artifactBytes: 0,
        artifactPaths: new Set(),
        watchdog: null,
        deadline: null,
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
        ).pipe(
          Effect.andThen(
            SynchronizedRef.update(runs, (map) => {
              const next = new Map(map);
              next.delete(runId);
              return next;
            }),
          ),
          Effect.andThen(Effect.fail(error)),
        );

      const definition = yield* experiments
        .resolve({ experimentId: input.experimentId })
        .pipe(Effect.catchTag("RlRunStartError", failStart));

      const source = yield* sourceEvidence
        .resolve(input.projectId)
        .pipe(Effect.catchTag("RlRunStartError", failStart));

      const resolvedRunner = yield* capabilities
        .resolveRunner({ runnerId: definition.runnerId })
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

      record.manifestBase = {
        experimentId: definition.experimentId,
        runnerId: definition.runnerId,
        runnerVersion: resolvedRunner.runnerVersion,
        protocolVersion: RL_WORKER_PROTOCOL_VERSION,
        seed: input.seed,
        effectiveConfig: definition.config,
        sourceRevision: source.sourceRevision,
        sourceDirty: source.sourceDirty,
        pythonExecutable: resolvedRunner.executable,
        pythonVersion: resolvedRunner.version,
        environmentFingerprint: resolvedRunner.environmentFingerprint,
        instrumentationLevel: definition.instrumentationLevel,
        hardwareSummary: `${hostPlatform}/${hostArchitecture}`,
      };

      const workerArgs = [
        definition.entrypoint,
        "--run-dir",
        runRoot,
        "--seed",
        String(input.seed),
        "--config-json",
        // @effect-diagnostics-next-line preferSchemaOverJson:off - bounded worker argv payload.
        JSON.stringify(definition.config),
        ...(definition.scenario === undefined ? [] : ["--scenario", definition.scenario]),
      ];
      const workerEnv = Object.fromEntries(
        ["PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR"]
          .map((name) => [name, process.env[name]] as const)
          .filter((entry): entry is readonly [string, string] => entry[1] !== undefined),
      );

      const worker = yield* spawner
        .spawn({
          command: resolvedRunner.executable,
          args: workerArgs,
          cwd: source.workspaceRoot,
          env: {
            ...workerEnv,
            T3RL_RUN_ID: runId,
            PYTHONHASHSEED: String(input.seed),
          },
        })
        .pipe(
          Effect.provideService(Scope.Scope, scope),
          Effect.catchTag("RlWorkerSpawnError", (cause) =>
            failStart(new RlRunStartError({ code: "RunnerUnavailable", detail: cause.message })),
          ),
        );

      record.worker = worker;
      yield* armWatchdog(
        record,
        STARTUP_TIMEOUT,
        "WorkerHelloTimeout",
        "worker did not send hello in time",
      );
      record.deadline = yield* Effect.forkIn(
        Effect.sleep(Duration.seconds(definition.maxRuntimeSeconds)).pipe(
          Effect.andThen(
            failRun(record, "WorkerStalled", "worker exceeded the experiment runtime limit"),
          ),
        ),
        scope,
      );

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
      const exitFiber = yield* Effect.forkIn(
        watchExit(record, worker, stdoutFiber, stderrFiber),
        scope,
      );
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
        yield* SynchronizedRef.update(runs, (map) => {
          const next = new Map(map);
          next.delete(input.runId);
          return next;
        });
        return { state: record.state };
      }

      yield* stopWorker(record);

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
      const metrics = yield* store
        .listMetrics({ runId: input.runId, limit: RL_MAX_SNAPSHOT_METRIC_BATCHES })
        .pipe(Effect.orDie);
      return { summary: detail.summary, manifest: detail.manifest, artifacts, metrics };
    });

  const subscribe: RlManagerShape["subscribe"] = (input, onEvent) =>
    Effect.gen(function* () {
      const record = yield* getRecord(input.runId);
      if (record === null) {
        const detail = yield* get(input);
        onEvent({ _tag: "Snapshot", ...detail });
        return () => {};
      }

      // Register first and buffer while reading the durable snapshot. Events
      // already reflected in that snapshot are deduplicated before delivery;
      // events that raced after the read are replayed immediately after it.
      let snapshotDelivered = false;
      const buffered: RlSubscriptionEvent[] = [];
      const subscriber: Subscriber = (event) => {
        if (!snapshotDelivered) buffered.push(event);
        else onEvent(event);
      };
      record.subscribers.add(subscriber);
      const detail = yield* get(input).pipe(
        Effect.onError(() => Effect.sync(() => record.subscribers.delete(subscriber))),
      );
      onEvent({ _tag: "Snapshot", ...detail });
      snapshotDelivered = true;

      const metricKeys = new Set(
        detail.metrics.map((batch) => `${batch.step}:${batch.wallClockMs}`),
      );
      const artifactIds = new Set(detail.artifacts.map((artifact) => artifact.artifactId));
      for (const event of buffered) {
        if (event._tag === "Lifecycle" && event.summary.state === detail.summary.state) continue;
        if (event._tag === "Manifest" && detail.manifest !== null) continue;
        if (event._tag === "Artifact" && artifactIds.has(event.artifact.artifactId)) continue;
        if (
          event._tag === "Metrics" &&
          metricKeys.has(`${event.batch.step}:${event.batch.wallClockMs}`)
        ) {
          continue;
        }
        onEvent(event);
      }
      return () => {
        record.subscribers.delete(subscriber);
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
    capabilities: () =>
      Effect.gen(function* () {
        const [report, availableExperiments] = yield* Effect.all([
          capabilities.report(),
          experiments.list().pipe(Effect.orDie),
        ]);
        return { ...report, experiments: availableExperiments };
      }),
    start,
    cancel,
    list,
    get,
    subscribe,
    sweepInterruptedRuns,
  });
});

export const RlManagerLive = Layer.effect(RlManager, makeManager);
