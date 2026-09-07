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
  RlStudyNotFoundError,
  isTerminalRlRunState,
  type RlArtifactEvidence,
  type RlArtifactPage,
  type RlArtifactMetadata,
  type RlCheckpointArtifactEvidence,
  type RlErrorCode,
  RlEvaluationProtocol,
  type RlLineageEdge,
  type RlLineageRelation,
  type RlMetricBatch,
  type RlResolvedManifest,
  type RlRunLineage,
  type RlRunState,
  type RlRunSummary,
  type RlStudy,
  type RlStudyComparison,
  type RlStudyDefinition,
  type RlStudyEstimator,
  type RlExperimentValidationReport,
  type RlSubscriptionEvent,
  RL_MAX_ARTIFACT_PAGE_SIZE,
  RL_MAX_RUN_ARTIFACTS,
  RL_MAX_SNAPSHOT_ARTIFACTS,
  RL_MAX_SNAPSHOT_METRIC_BATCHES,
} from "@t3tools/contracts";
import * as Context from "effect/Context";
import * as Crypto from "effect/Crypto";
import * as DateTime from "effect/DateTime";
import * as Deferred from "effect/Deferred";
import * as Duration from "effect/Duration";
import * as Effect from "effect/Effect";
import * as Encoding from "effect/Encoding";
import * as FileSystem from "effect/FileSystem";
import * as Fiber from "effect/Fiber";
import * as Layer from "effect/Layer";
import * as Path from "effect/Path";
import * as Result from "effect/Result";
import * as Scope from "effect/Scope";
import * as Schema from "effect/Schema";
import * as Stream from "effect/Stream";
import * as SynchronizedRef from "effect/SynchronizedRef";

import { HostProcessArchitecture, HostProcessPlatform } from "@t3tools/shared/hostProcess";

import * as ServerConfig from "../config.ts";
import * as ArtifactIdentity from "./ArtifactIdentity.ts";
import * as Artifacts from "./Artifacts.ts";
import { Capabilities } from "./Capabilities.ts";
import { Experiments } from "./Experiments.ts";
import { snapshotEnvironment } from "./EnvironmentEvidence.ts";
import * as Lifecycle from "./Lifecycle.ts";
import { RunStore } from "./RunStore.ts";
import { SourceEvidence } from "./SourceEvidence.ts";
import { WorkerSpawner, type WorkerProcess } from "./WorkerSpawner.ts";
import * as WorkerProtocol from "./WorkerProtocol.ts";
import { comparePairedStudyEffect } from "./StudyStatistics.ts";
import { collectStudyObservation } from "./StudyEvidence.ts";

/** How long a cancelled worker gets to exit before SIGKILL. */
export const CANCEL_GRACE = Duration.seconds(5);
/** The worker must announce its protocol promptly after spawn. */
export const STARTUP_TIMEOUT = Duration.seconds(15);
/** Any valid protocol message resets this liveness deadline. */
export const WORKER_STALL_TIMEOUT = Duration.seconds(60);
/** Retained stderr, flushed to the run log on exit. Oldest bytes are dropped. */
export const MAX_STDERR_BYTES = 64 * 1024;
/** Metric batches that reach live subscribers per second. Durable evidence is never rate-limited. */
export const MAX_METRIC_BATCHES_PER_SECOND = 2;
const METRIC_FLUSH_INTERVAL = Duration.millis(1000 / MAX_METRIC_BATCHES_PER_SECOND);
export const MAX_ARTIFACT_BYTES = 256 * 1024 * 1024;
export const MAX_RUN_ARTIFACT_BYTES = 512 * 1024 * 1024;

export interface RlRunDetail {
  readonly summary: RlRunSummary;
  readonly manifest: RlResolvedManifest | null;
  readonly lineage: RlRunLineage;
  readonly artifacts: ReadonlyArray<RlArtifactMetadata>;
  readonly metrics: ReadonlyArray<RlMetricBatch>;
}

export interface StartRunInput {
  readonly projectId: string;
  readonly experimentId: string;
  readonly seed: number;
  readonly requestId?: string | undefined;
  readonly seeds?: RlResolvedManifest["seeds"] | undefined;
  readonly evaluationProtocol?: RlEvaluationProtocol | undefined;
}

export interface ContinueRunInput {
  readonly projectId: string;
  readonly parentRunId: string;
  readonly sourceArtifactId: string;
  readonly requestId: string;
  readonly targetExperimentId?: string | undefined;
}

type Subscriber = (event: RlSubscriptionEvent) => void;
const decodeEvaluationProtocol = Schema.decodeUnknownEffect(RlEvaluationProtocol);

interface ContinuationSource {
  readonly relation: RlLineageRelation;
  readonly parentRunId: string;
  readonly experimentId: string;
  readonly seed: number;
  readonly seeds: RlResolvedManifest["seeds"];
  readonly evaluationProtocol: RlEvaluationProtocol | undefined;
  readonly sourceArtifactId: string;
  readonly sourceArtifactSha256: string;
  readonly sourceStep: number;
  readonly sourcePath: string;
  readonly compatibility: RlArtifactEvidence["compatibility"];
}

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
  resolvedManifest: RlResolvedManifest | null;
  expectedProtocolVersion: number;
  continuation: ContinuationSource | null;
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
  readonly resume: (
    input: ContinueRunInput,
  ) => Effect.Effect<{ readonly runId: string }, RlRunStartError>;
  readonly warmStart: (
    input: ContinueRunInput,
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
  readonly listArtifacts: (input: {
    readonly runId: string;
    readonly cursor?: string | undefined;
    readonly limit?: number | undefined;
  }) => Effect.Effect<RlArtifactPage, RlRunNotFoundError>;
  /** Delivers a snapshot, then live events, until the returned unsubscribe runs. */
  readonly subscribe: (
    input: { readonly runId: string },
    onEvent: Subscriber,
  ) => Effect.Effect<() => void, RlRunNotFoundError>;
  readonly sweepInterruptedRuns: () => Effect.Effect<number>;
  readonly createStudy: (input: {
    readonly projectId: string;
    readonly definition: RlStudyDefinition;
  }) => Effect.Effect<RlStudy, RlRunStartError>;
  readonly getStudy: (input: {
    readonly studyId: string;
  }) => Effect.Effect<RlStudy, RlStudyNotFoundError>;
  readonly compareStudy: (input: {
    readonly studyId: string;
    readonly baselineLabel: string;
    readonly candidateLabel: string;
    readonly metricKey: string;
    readonly estimator: RlStudyEstimator;
  }) => Effect.Effect<RlStudyComparison, RlStudyNotFoundError>;
  readonly validateExperiment: (input: {
    readonly projectId: string;
    readonly experimentId: string;
  }) => Effect.Effect<RlExperimentValidationReport>;
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

  const canonicalJson = (value: unknown): string => {
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
    if (typeof value === "object" && value !== null) {
      return `{${Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
        .join(",")}}`;
    }
    return JSON.stringify(value);
  };

  const sha256Text = (value: string) =>
    crypto
      .digest("SHA-256", new TextEncoder().encode(value))
      .pipe(Effect.orDie, Effect.map(Encoding.encodeHex));

  const manifestCompatibility = (manifest: RlResolvedManifest) =>
    manifest.model === undefined
      ? null
      : {
          model: manifest.model,
          framework: { id: manifest.runnerId, version: manifest.runnerVersion },
          environmentFingerprint: manifest.environmentFingerprint,
          environmentLockSha256: manifest.environmentLock?.lockfileSha256 ?? null,
        };

  const configuredModelMismatch = (
    config: Readonly<Record<string, unknown>>,
    model: NonNullable<RlResolvedManifest["model"]>,
  ): string | null => {
    const equalWhenDeclared = (key: string, actual: unknown): boolean =>
      !(key in config) || canonicalJson(config[key]) === canonicalJson(actual);
    if (!equalWhenDeclared("modelId", model.baseModelId)) return "base model ID";
    if (
      typeof config.modelRevision === "string" &&
      /^[0-9a-f]{40,64}$/.test(config.modelRevision) &&
      config.modelRevision !== model.baseModelRevision
    ) {
      return "base model revision";
    }
    if (
      typeof config.tokenizerRevision === "string" &&
      /^[0-9a-f]{40,64}$/.test(config.tokenizerRevision) &&
      config.tokenizerRevision !== model.tokenizerRevision
    ) {
      return "tokenizer revision";
    }
    for (const [key, actual] of [
      ["quantization", model.quantization],
      ["precision", model.precision],
      ["loraRank", model.peftConfig.rank],
      ["loraAlpha", model.peftConfig.alpha],
      ["loraDropout", model.peftConfig.dropout],
      ["loraBias", model.peftConfig.bias],
      ["loraTargetModules", model.peftConfig.targetModules],
      ["loraModulesToSave", model.peftConfig.modulesToSave],
      ["useRslora", model.peftConfig.useRslora],
    ] as const) {
      if (!equalWhenDeclared(key, actual)) return key;
    }
    const configuredTrainableModules = [
      ...(Array.isArray(config.loraTargetModules) ? config.loraTargetModules : []),
      ...(Array.isArray(config.loraModulesToSave) ? config.loraModulesToSave : []),
    ];
    if (
      configuredTrainableModules.length > 0 &&
      canonicalJson(configuredTrainableModules) !== canonicalJson(model.trainableModules)
    ) {
      return "trainable modules";
    }
    return null;
  };

  const compatibilityMismatch = (
    manifest: RlResolvedManifest,
    compatibility: RlArtifactEvidence["compatibility"],
  ): string | null => {
    const expected = manifestCompatibility(manifest);
    if (expected === null) return "the resolved manifest has no model identity";
    if (canonicalJson(expected.model) !== canonicalJson(compatibility.model)) {
      return "model, tokenizer, precision, quantization, or PEFT configuration changed";
    }
    if (
      expected.framework.id !== compatibility.framework.id ||
      expected.framework.version !== compatibility.framework.version
    ) {
      return "framework identity changed";
    }
    if (expected.environmentFingerprint !== compatibility.environmentFingerprint) {
      return "environment fingerprint changed";
    }
    if (expected.environmentLockSha256 !== compatibility.environmentLockSha256) {
      return "environment lock changed";
    }
    return null;
  };

  const missingPublishedState = (
    evidence: RlCheckpointArtifactEvidence,
    contentManifest: ReadonlyArray<ArtifactIdentity.ArtifactContentEntry>,
  ): string | null => {
    const state = evidence.resumeState;
    if (
      !state.trainerState ||
      !state.optimizerState ||
      !state.schedulerState ||
      !state.rngState ||
      !state.datasetCursorState
    ) {
      return "checkpoint evidence declares incomplete trainer state";
    }
    if (
      evidence.compatibility.model.precision === "fp16" &&
      state.gradientScalerState !== "captured"
    ) {
      return "fp16 checkpoint did not capture gradient scaler state";
    }
    const files = new Set(contentManifest.map((entry) => entry.path));
    const required = [
      "adapter_config.json",
      "adapter_model.safetensors",
      "trainer_state.json",
      "optimizer.pt",
      "scheduler.pt",
      "rng_state.pth",
      "t3rl-dataset-cursor.json",
      ...(state.gradientScalerState === "captured" ? ["scaler.pt"] : []),
    ];
    for (const name of required) {
      if (!files.has(name)) return `checkpoint is missing ${name}`;
    }
    for (const name of state.stateFiles) {
      if (!files.has(name)) return `checkpoint evidence names a missing state file: ${name}`;
    }
    return null;
  };

  const publishedEvidenceError = (
    record: RunRecord,
    kind: RlArtifactMetadata["kind"],
    evidence: RlArtifactEvidence | undefined,
    identity: ArtifactIdentity.ArtifactIdentity,
  ): string | null => {
    if (kind !== "checkpoint" && kind !== "adapter") {
      return evidence === undefined ? null : `${kind} artifacts cannot carry checkpoint evidence`;
    }
    if (record.expectedProtocolVersion < 2) {
      return evidence === undefined
        ? null
        : "protocol-v1 artifacts cannot carry checkpoint evidence";
    }
    if (evidence === undefined) return `${kind} artifacts require protocol-v2 evidence`;
    if (record.resolvedManifest === null) return "artifact arrived before a resolved manifest";
    if (kind === "checkpoint" && evidence._tag !== "Checkpoint") {
      return "checkpoint artifact carried adapter evidence";
    }
    if (kind === "adapter" && evidence._tag !== "Adapter") {
      return "adapter artifact carried checkpoint evidence";
    }
    const mismatch = compatibilityMismatch(record.resolvedManifest, evidence.compatibility);
    if (mismatch !== null) return `artifact compatibility mismatch: ${mismatch}`;
    const files = new Set(identity.contentManifest.map((entry) => entry.path));
    if (!files.has("adapter_config.json") || !files.has("adapter_model.safetensors")) {
      return `${kind} must contain adapter_config.json and adapter_model.safetensors`;
    }
    return evidence._tag === "Checkpoint"
      ? missingPublishedState(evidence, identity.contentManifest)
      : null;
  };

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
      publish(record, { _tag: "Metrics", batch });
    });

  /**
   * Persists every worker observation before applying the live transport rate
   * limit. The one pending publication may be replaced by a newer step, but
   * values from different steps are never combined and reconnect snapshots
   * replay every durable semantic step. Values from the same step may coalesce
   * for presentation because they describe one observation point.
   */
  const offerMetrics = (runId: string, batch: RlMetricBatch) =>
    Effect.gen(function* () {
      const record = yield* getRecord(runId);
      if (record === null) return;
      record.metricSeq += 1;
      const at = yield* now;
      yield* store.appendMetrics({ runId, seq: record.metricSeq, batch, at }).pipe(Effect.orDie);
      record.pendingBatch =
        record.pendingBatch === null || record.pendingBatch.step !== batch.step
          ? batch
          : {
              step: batch.step,
              wallClockMs: batch.wallClockMs,
              values: { ...record.pendingBatch.values, ...batch.values },
            };
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

  const writeFileAtomically = (target: string, contents: string | Uint8Array) =>
    Effect.gen(function* () {
      const uuid = yield* crypto.randomUUIDv4.pipe(Effect.orDie);
      const temporary = `${target}.${uuid}.tmp`;
      yield* (
        typeof contents === "string"
          ? fs.writeFileString(temporary, contents)
          : fs.writeFile(temporary, contents)
      ).pipe(
        Effect.andThen(fs.rename(temporary, target)),
        Effect.ensuring(fs.remove(temporary, { force: true }).pipe(Effect.ignore)),
      );
    });

  const artifactContentType = (
    kind: RlArtifactMetadata["kind"],
    relativePath: string,
    directory: boolean,
  ): string => {
    if (directory) return "application/vnd.t3rl.directory.v1";
    const extension = path.extname(relativePath).toLowerCase();
    if (extension === ".json") return "application/json";
    if (extension === ".jsonl" || extension === ".ndjson") return "application/x-ndjson";
    if (extension === ".txt" || extension === ".log" || kind === "log") return "text/plain";
    if (extension === ".zip") return "application/zip";
    if (extension === ".safetensors") return "application/octet-stream";
    return "application/octet-stream";
  };

  const recordReadyArtifact = (
    record: RunRecord,
    kind: RlArtifactMetadata["kind"],
    relativePath: string,
    artifactPath: string,
    evidence?: RlArtifactEvidence | undefined,
  ) =>
    Effect.gen(function* () {
      const remainingBytes = MAX_RUN_ARTIFACT_BYTES - record.artifactBytes;
      const identity = yield* ArtifactIdentity.computeArtifactIdentity({
        artifactPath,
        maxBytes: Math.min(MAX_ARTIFACT_BYTES, remainingBytes),
      }).pipe(
        Effect.provideService(FileSystem.FileSystem, fs),
        Effect.provideService(Path.Path, path),
      );
      const evidenceError = publishedEvidenceError(record, kind, evidence, identity);
      if (evidenceError !== null) {
        return yield* new ArtifactIdentity.ArtifactIdentityError({ detail: evidenceError });
      }
      const at = yield* now;
      const artifact = yield* store.recordArtifact({
        runId: record.runId,
        kind,
        relativePath,
        bytes: identity.bytes,
        contentType: artifactContentType(kind, relativePath, identity.directory),
        producedAt: at,
        sha256: identity.sha256,
        logicalName: relativePath,
        format: ArtifactIdentity.inferArtifactFormat(relativePath, identity.directory),
        fileCount: identity.fileCount,
        contentManifest: identity.contentManifest,
        ...(evidence === undefined ? {} : { evidence }),
        ...(evidence?._tag === "Checkpoint" ? { checkpointStep: evidence.globalStep } : {}),
      });
      record.artifactPaths.add(relativePath);
      record.artifactBytes += artifact.bytes;
      publish(record, { _tag: "Artifact", artifact });
      return artifact;
    });

  const enforceCheckpointRetention = (record: RunRecord) =>
    Effect.gen(function* () {
      const limit = record.resolvedManifest?.checkpointPolicy?.maxIntermediateCheckpoints;
      if (limit === undefined) return;
      const checkpoints = yield* store.listReadyIntermediateCheckpoints({ runId: record.runId });
      for (const checkpoint of checkpoints.slice(limit)) {
        const resolved = Artifacts.resolveArtifactPath({
          rlRunsDir: config.rlRunsDir,
          runId: record.runId,
          relativePath: checkpoint.relativePath,
        });
        if (resolved === null) continue;
        yield* fs.remove(resolved, { force: true, recursive: true });
        const trashed = yield* store.setArtifactState({
          artifactId: checkpoint.metadata.artifactId,
          state: "trashed",
        });
        if (trashed === null) continue;
        record.artifactBytes = Math.max(0, record.artifactBytes - checkpoint.metadata.bytes);
        publish(record, { _tag: "Artifact", artifact: trashed });
      }
    });

  const writeStderrArtifact = (record: RunRecord) =>
    Effect.gen(function* () {
      const runRoot = Artifacts.runDirectory({
        rlRunsDir: config.rlRunsDir,
        runId: record.runId,
      });
      if (runRoot === null) return;
      const relativePath = "worker.log";
      const target = path.join(runRoot, relativePath);
      yield* writeFileAtomically(target, record.stderr);
      yield* recordReadyArtifact(record, "log", relativePath, target);
    }).pipe(
      Effect.catchCause((cause) =>
        Effect.logWarning("rl final log artifact was not recorded", {
          runId: record.runId,
          cause,
        }),
      ),
    );

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
      const target = path.join(runRoot, relativePath);
      yield* writeFileAtomically(target, contents);
      yield* recordReadyArtifact(record, "manifest", relativePath, target);
    });

  const stopWorker = (record: RunRecord) =>
    Effect.gen(function* () {
      const worker = record.worker;
      if (worker === null) return;
      yield* worker.kill("SIGTERM");
      const grace =
        record.resolvedManifest?.checkpointPolicy === undefined
          ? CANCEL_GRACE
          : Duration.seconds(record.resolvedManifest.checkpointPolicy.gracefulDeadlineSeconds);
      yield* Effect.forkIn(
        Effect.sleep(grace).pipe(
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

  const handleArtifact = (
    record: RunRecord,
    kind: RlArtifactMetadata["kind"],
    relative: string,
    evidence?: RlArtifactEvidence | undefined,
  ) =>
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
      const artifact = yield* recordReadyArtifact(
        record,
        kind,
        relative,
        canonical.file.value,
        evidence,
      ).pipe(
        Effect.catch((error) =>
          failRun(
            record,
            "MalformedWorkerMessage",
            `artifact could not be verified: ${relative} (${"detail" in error ? String(error.detail) : "identity or evidence rejected"})`,
          ).pipe(Effect.as(null)),
        ),
      );
      if (artifact === null) return;
      if (evidence?._tag === "Checkpoint" && evidence.checkpointClass === "intermediate") {
        yield* enforceCheckpointRetention(record).pipe(
          Effect.catch(() =>
            failRun(record, "RunnerException", "server could not enforce checkpoint retention"),
          ),
        );
      }
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
            decoded.message.protocol !== record.expectedProtocolVersion ||
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
          if (
            record.expectedProtocolVersion >= 2 &&
            (decoded.message.model === undefined || decoded.message.checkpointPolicy === undefined)
          ) {
            yield* failRun(
              record,
              "MalformedWorkerMessage",
              "protocol-v2 manifest omitted model identity or checkpoint policy",
            );
            return;
          }
          if (decoded.message.model !== undefined) {
            const peftConfigSha256 = yield* sha256Text(
              canonicalJson(decoded.message.model.peftConfig),
            );
            if (peftConfigSha256 !== decoded.message.model.peftConfigSha256) {
              yield* failRun(
                record,
                "MalformedWorkerMessage",
                "manifest PEFT configuration hash did not match its canonical configuration",
              );
              return;
            }
            const configuredMismatch = configuredModelMismatch(
              record.manifestBase.effectiveConfig,
              decoded.message.model,
            );
            if (configuredMismatch !== null) {
              yield* failRun(
                record,
                "MalformedWorkerMessage",
                `manifest model identity disagreed with configured ${configuredMismatch}`,
              );
              return;
            }
          }
          const manifest: RlResolvedManifest = {
            ...record.manifestBase,
            effectiveConfig: {
              ...record.manifestBase.effectiveConfig,
              ...decoded.message.values,
            },
            ...(decoded.message.model === undefined ? {} : { model: decoded.message.model }),
            ...(decoded.message.checkpointPolicy === undefined
              ? {}
              : { checkpointPolicy: decoded.message.checkpointPolicy }),
          };
          if (record.continuation !== null) {
            const mismatch = compatibilityMismatch(manifest, record.continuation.compatibility);
            if (mismatch !== null) {
              yield* failRun(
                record,
                "ResumeIncompatible",
                `child manifest is incompatible with its ${record.continuation.relation} source: ${mismatch}`,
              );
              return;
            }
          }
          yield* store.setManifest({ runId, manifest }).pipe(Effect.orDie);
          record.resolvedManifest = manifest;
          record.protocolPhase = "ready";
          const manifestArtifactRecorded = yield* writeManifestArtifact(record, manifest).pipe(
            Effect.as(true),
            Effect.catchCause((cause) =>
              Effect.logWarning("rl manifest artifact was not recorded", {
                runId: record.runId,
                cause,
              }).pipe(
                Effect.andThen(
                  failRun(
                    record,
                    "RunnerException",
                    "server could not persist the resolved manifest artifact",
                  ),
                ),
                Effect.as(false),
              ),
            ),
          );
          if (!manifestArtifactRecorded) return;
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
          yield* handleArtifact(
            record,
            decoded.message.kind,
            decoded.message.path,
            decoded.message.evidence,
          );
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
        case "Heartbeat": {
          if (record.expectedProtocolVersion < 2 || record.protocolPhase !== "ready") {
            yield* failRun(record, "MalformedWorkerMessage", "worker heartbeat was out of order");
            return;
          }
          yield* armWatchdog(
            record,
            WORKER_STALL_TIMEOUT,
            "WorkerStalled",
            "worker stopped producing output",
          );
          return;
        }
        case "Resource": {
          if (record.expectedProtocolVersion < 2 || record.protocolPhase !== "ready") {
            yield* failRun(
              record,
              "MalformedWorkerMessage",
              "worker resource sample was out of order",
            );
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

  const continuationFailure = (detail: string) =>
    new RlRunStartError({ code: "ResumeIncompatible", detail });

  const prepareContinuation = (
    input: ContinueRunInput,
    relation: RlLineageRelation,
  ): Effect.Effect<ContinuationSource, RlRunStartError> =>
    Effect.gen(function* () {
      const parent = yield* store
        .getRun({ runId: input.parentRunId })
        .pipe(
          Effect.mapError(() => continuationFailure("parent run was not found in this project")),
        );
      if (parent.summary.projectId !== input.projectId) {
        return yield* continuationFailure("parent run was not found in this project");
      }
      if (!isTerminalRlRunState(parent.summary.state)) {
        return yield* continuationFailure("parent run must be terminal before creating a child");
      }
      if (parent.manifest === null || parent.manifest.protocolVersion < 2) {
        return yield* continuationFailure("parent run has no protocol-v2 compatibility manifest");
      }

      const source = yield* store
        .findArtifact({
          runId: input.parentRunId,
          artifactId: input.sourceArtifactId,
        })
        .pipe(
          Effect.mapError(() => continuationFailure("source artifact metadata is unavailable")),
        );
      if (source === null) return yield* continuationFailure("source artifact was not found");
      if (source.metadata.state !== "ready" || source.metadata.sha256 == null) {
        return yield* continuationFailure("source artifact is not ready with a verified hash");
      }
      const evidence = source.metadata.evidence;
      if (relation === "resume") {
        if (source.metadata.kind !== "checkpoint" || evidence?._tag !== "Checkpoint") {
          return yield* continuationFailure("exact resume requires a verified checkpoint artifact");
        }
        if (source.contentManifest === null) {
          return yield* continuationFailure("checkpoint has no verified content manifest");
        }
        const incomplete = missingPublishedState(evidence, source.contentManifest);
        if (incomplete !== null) return yield* continuationFailure(incomplete);
      } else if (source.metadata.kind !== "adapter" || evidence?._tag !== "Adapter") {
        return yield* continuationFailure("warm start requires a verified adapter artifact");
      }
      if (evidence === undefined || evidence === null) {
        return yield* continuationFailure("source artifact has no compatibility evidence");
      }
      if (source.contentManifest === null) {
        return yield* continuationFailure("source artifact has no verified content manifest");
      }
      const contentFiles = new Set(source.contentManifest.map((entry) => entry.path));
      if (
        !contentFiles.has("adapter_config.json") ||
        !contentFiles.has("adapter_model.safetensors")
      ) {
        return yield* continuationFailure(
          "source artifact is not an independently loadable PEFT adapter",
        );
      }
      const parentMismatch = compatibilityMismatch(parent.manifest, evidence.compatibility);
      if (parentMismatch !== null) {
        return yield* continuationFailure(`source disagrees with its parent: ${parentMismatch}`);
      }

      const projectSource = yield* sourceEvidence.resolve(input.projectId);
      const definition = yield* experiments.resolve({
        experimentId: input.targetExperimentId ?? parent.summary.experimentId,
        workspaceRoot: projectSource.workspaceRoot,
      });
      if ((definition.protocolVersion ?? 1) < 2) {
        return yield* continuationFailure("experiment is not configured for protocol-v2 resume");
      }
      if (relation === "resume") {
        const resolvedInputs = (definition.projectInputs ?? []).map((entry) => ({
          role: entry.role,
          logicalName: entry.snapshotName,
          sha256: entry.sha256,
          bytes: entry.bytes,
        }));
        if (
          canonicalJson(parent.manifest.effectiveConfig.resolvedProjectInputs ?? []) !==
          canonicalJson(resolvedInputs)
        ) {
          return yield* continuationFailure("project input snapshots changed since the parent run");
        }
        for (const [key, value] of Object.entries(definition.config)) {
          const inputRole =
            key === "projectDatasetPath"
              ? "dataset"
              : key === "projectVerifierPath"
                ? "verifier"
                : null;
          if (
            inputRole !== null &&
            definition.projectInputs?.some((entry) => entry.role === inputRole)
          )
            continue;
          if (canonicalJson(parent.manifest.effectiveConfig[key]) !== canonicalJson(value)) {
            return yield* continuationFailure(`experiment configuration changed at ${key}`);
          }
        }
      }
      const runner = yield* capabilities.resolveRunner({
        runnerId: definition.runnerId,
        ...(definition.method === undefined ? {} : { method: definition.method }),
      });
      if (
        runner.runnerId !== evidence.compatibility.framework.id ||
        runner.runnerVersion !== evidence.compatibility.framework.version
      ) {
        return yield* continuationFailure("installed framework does not match the source artifact");
      }
      if (runner.environmentFingerprint !== evidence.compatibility.environmentFingerprint) {
        return yield* continuationFailure("installed environment fingerprint changed");
      }
      if (
        (runner.environmentLock?.lockfileSha256 ?? null) !==
        evidence.compatibility.environmentLockSha256
      ) {
        return yield* continuationFailure("installed environment lock changed");
      }

      const parentRoot = Artifacts.runDirectory({
        rlRunsDir: config.rlRunsDir,
        runId: input.parentRunId,
      });
      const artifactPath = Artifacts.resolveArtifactPath({
        rlRunsDir: config.rlRunsDir,
        runId: input.parentRunId,
        relativePath: source.relativePath,
      });
      if (parentRoot === null || artifactPath === null) {
        return yield* continuationFailure("source artifact path is unavailable");
      }
      const canonical = yield* Effect.all({
        root: fs.realPath(parentRoot),
        artifact: fs.realPath(artifactPath),
      }).pipe(Effect.mapError(() => continuationFailure("source artifact file is unavailable")));
      const relative = path.relative(canonical.root, canonical.artifact);
      if (relative === "" || relative.startsWith("..") || path.isAbsolute(relative)) {
        return yield* continuationFailure("source artifact escaped its parent run directory");
      }
      const verified = yield* ArtifactIdentity.verifyArtifactIdentity({
        artifactPath: canonical.artifact,
        expectedSha256: source.metadata.sha256,
        maxBytes: MAX_ARTIFACT_BYTES,
      }).pipe(
        Effect.provideService(FileSystem.FileSystem, fs),
        Effect.provideService(Path.Path, path),
        Effect.mapError(() => continuationFailure("source artifact could not be re-verified")),
      );
      if (!verified) {
        return yield* continuationFailure(
          "source artifact bytes no longer match its recorded hash",
        );
      }

      return {
        relation,
        parentRunId: input.parentRunId,
        experimentId: definition.experimentId,
        seed: parent.manifest.seed,
        seeds: parent.manifest.seeds,
        evaluationProtocol:
          relation === "resume" && parent.manifest.effectiveConfig.evaluationProtocol !== undefined
            ? yield* decodeEvaluationProtocol(
                parent.manifest.effectiveConfig.evaluationProtocol,
              ).pipe(
                Effect.mapError(() => continuationFailure("parent evaluation protocol is invalid")),
              )
            : undefined,
        sourceArtifactId: input.sourceArtifactId,
        sourceArtifactSha256: source.metadata.sha256,
        sourceStep: evidence.globalStep,
        sourcePath: canonical.artifact,
        compatibility: evidence.compatibility,
      };
    });

  // ---- public surface ----------------------------------------------------

  const startWithContinuation = (
    input: StartRunInput,
    continuation: ContinuationSource | null,
  ): Effect.Effect<{ readonly runId: string }, RlRunStartError> =>
    Effect.gen(function* () {
      const uuid = yield* crypto.randomUUIDv4.pipe(Effect.orDie);
      const proposedRunId = `run_${uuid.replace(/-/g, "").slice(0, 24)}`;
      const requestedAt = yield* now;
      const lineage =
        continuation === null
          ? undefined
          : ({
              childRunId: proposedRunId,
              parentRunId: continuation.parentRunId,
              sourceArtifactId: continuation.sourceArtifactId,
              sourceArtifactSha256: continuation.sourceArtifactSha256,
              relation: continuation.relation,
              sourceStep: continuation.sourceStep,
              createdAt: requestedAt,
            } satisfies RlLineageEdge);

      const requested = yield* store
        .insertRequested({
          runId: proposedRunId,
          projectId: input.projectId,
          experimentId: input.experimentId,
          requestedAt,
          ...(input.requestId === undefined ? {} : { requestId: input.requestId }),
          ...(lineage === undefined ? {} : { lineage }),
        })
        .pipe(Effect.orDie);
      if (!requested.inserted) {
        if (continuation === null) return { runId: requested.runId };
        const existingLineage = yield* store
          .getLineage({ runId: requested.runId })
          .pipe(Effect.orDie);
        const direct = existingLineage.edges[0];
        if (
          direct === undefined ||
          direct.parentRunId !== continuation.parentRunId ||
          direct.sourceArtifactId !== continuation.sourceArtifactId ||
          direct.sourceArtifactSha256 !== continuation.sourceArtifactSha256 ||
          direct.relation !== continuation.relation
        ) {
          return yield* continuationFailure(
            "request ID already belongs to a different child-run continuation",
          );
        }
        return { runId: requested.runId };
      }
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
        resolvedManifest: null,
        expectedProtocolVersion: 1,
        continuation,
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

      const source = yield* sourceEvidence
        .resolve(input.projectId)
        .pipe(Effect.catchTag("RlRunStartError", failStart));

      const definition = yield* experiments
        .resolve({ experimentId: input.experimentId, workspaceRoot: source.workspaceRoot })
        .pipe(Effect.catchTag("RlRunStartError", failStart));
      if (input.seeds !== undefined) {
        const independentSeedsSupported =
          (definition.runnerId === "trl" || definition.runnerId === "axolotl") &&
          (definition.method === "sft" || definition.method === "dpo");
        if (
          input.seeds.training !== input.seed ||
          (!independentSeedsSupported &&
            Object.values(input.seeds).some((seed) => seed !== input.seed))
        ) {
          return yield* failStart(
            new RlRunStartError({
              code: "InvalidExperiment",
              detail:
                "This runner cannot apply the declared independent seed set. Match all seeds to the training seed or use an SFT/DPO runner that supports independent data seeds.",
            }),
          );
        }
      }
      const protocolVersion = definition.protocolVersion ?? 1;
      record.expectedProtocolVersion = protocolVersion;

      const resolvedRunner = yield* capabilities
        .resolveRunner({
          runnerId: definition.runnerId,
          ...(definition.method === undefined ? {} : { method: definition.method }),
        })
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

      let effectiveConfig = definition.config;
      if (definition.projectInputs !== undefined) {
        const inputsRoot = path.join(runRoot, "inputs", "project");
        yield* fs.makeDirectory(inputsRoot, { recursive: true }).pipe(Effect.orDie);
        const snapshotPaths = new Map<string, string>();
        for (const projectInput of definition.projectInputs) {
          const target = path.join(inputsRoot, projectInput.snapshotName);
          yield* fs.copy(projectInput.sourcePath, target, { overwrite: false }).pipe(
            Effect.mapError(
              () =>
                new RlRunStartError({
                  code: "InvalidExperiment",
                  detail: `Unable to snapshot project ${projectInput.role}`,
                }),
            ),
            Effect.catchTag("RlRunStartError", failStart),
          );
          const verified = yield* ArtifactIdentity.verifyArtifactIdentity({
            artifactPath: target,
            expectedSha256: projectInput.sha256,
            maxBytes: MAX_ARTIFACT_BYTES,
          }).pipe(
            Effect.provideService(FileSystem.FileSystem, fs),
            Effect.provideService(Path.Path, path),
            Effect.mapError(
              () =>
                new RlRunStartError({
                  code: "InvalidExperiment",
                  detail: `Project ${projectInput.role} changed while it was being snapshotted`,
                }),
            ),
            Effect.catchTag("RlRunStartError", failStart),
          );
          if (!verified) {
            return yield* failStart(
              new RlRunStartError({
                code: "InvalidExperiment",
                detail: `Project ${projectInput.role} changed while it was being snapshotted`,
              }),
            );
          }
          snapshotPaths.set(projectInput.role, target);
        }
        effectiveConfig = {
          ...definition.config,
          ...(snapshotPaths.has("dataset")
            ? { projectDatasetPath: snapshotPaths.get("dataset") }
            : {}),
          ...(snapshotPaths.has("verifier")
            ? { projectVerifierPath: snapshotPaths.get("verifier") }
            : {}),
        };
      }
      const workerConfig = effectiveConfig;
      if (definition.projectInputs !== undefined) {
        effectiveConfig = {
          ...workerConfig,
          resolvedProjectInputs: definition.projectInputs.map((entry) => ({
            role: entry.role,
            logicalName: entry.snapshotName,
            sha256: entry.sha256,
            bytes: entry.bytes,
          })),
        };
      }

      let continuationPath: string | null = null;
      if (continuation !== null) {
        continuationPath = path.join(
          runRoot,
          "inputs",
          continuation.relation === "resume" ? "checkpoint" : "adapter",
        );
        yield* fs.makeDirectory(path.dirname(continuationPath), { recursive: true }).pipe(
          Effect.mapError(() => continuationFailure("child input directory could not be created")),
          Effect.catchTag("RlRunStartError", failStart),
        );
        const copied = yield* fs
          .copy(continuation.sourcePath, continuationPath, {
            overwrite: false,
          })
          .pipe(
            Effect.andThen(
              ArtifactIdentity.verifyArtifactIdentity({
                artifactPath: continuationPath,
                expectedSha256: continuation.sourceArtifactSha256,
                maxBytes: MAX_ARTIFACT_BYTES,
              }).pipe(
                Effect.provideService(FileSystem.FileSystem, fs),
                Effect.provideService(Path.Path, path),
              ),
            ),
            Effect.mapError(() =>
              continuationFailure("source artifact could not be copied into the child run"),
            ),
            Effect.catchTag("RlRunStartError", failStart),
          );
        if (!copied) {
          return yield* failStart(
            continuationFailure("child source copy did not match the parent artifact hash"),
          );
        }
      }

      const environmentLock = yield* snapshotEnvironment(
        resolvedRunner.environmentLock,
        runRoot,
      ).pipe(
        Effect.provideService(FileSystem.FileSystem, fs),
        Effect.provideService(Path.Path, path),
        Effect.catchTag("RlRunStartError", failStart),
      );
      record.manifestBase = {
        experimentId: definition.experimentId,
        runnerId: definition.runnerId,
        runnerVersion: resolvedRunner.runnerVersion,
        protocolVersion,
        seed: input.seed,
        ...(input.seeds === undefined ? {} : { seeds: input.seeds }),
        effectiveConfig: {
          ...effectiveConfig,
          ...(input.evaluationProtocol === undefined
            ? {}
            : { evaluationProtocol: input.evaluationProtocol }),
        },
        sourceRevision: source.sourceRevision,
        sourceDirty: source.sourceDirty,
        pythonExecutable: resolvedRunner.executable,
        pythonVersion: resolvedRunner.version,
        environmentFingerprint: resolvedRunner.environmentFingerprint,
        environmentLock,
        instrumentationLevel: definition.instrumentationLevel,
        hardwareSummary: `${hostPlatform}/${hostArchitecture}`,
        ...(lineage === undefined ? {} : { lineage }),
      };

      const workerArgs = [
        definition.entrypoint,
        "--run-dir",
        runRoot,
        "--seed",
        String(input.seed),
        "--config-json",
        // @effect-diagnostics-next-line preferSchemaOverJson:off - bounded worker argv payload.
        JSON.stringify(workerConfig),
        ...(definition.scenario === undefined ? [] : ["--scenario", definition.scenario]),
        ...(continuationPath === null
          ? []
          : continuation?.relation === "resume"
            ? ["--resume-checkpoint", continuationPath]
            : ["--warm-start-adapter", continuationPath]),
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
            T3RL_ENVIRONMENT_FINGERPRINT: resolvedRunner.environmentFingerprint,
            T3RL_ENVIRONMENT_LOCK_SHA256: resolvedRunner.environmentLock?.lockfileSha256 ?? "",
            ...(input.seeds === undefined
              ? {}
              : {
                  // @effect-diagnostics-next-line preferSchemaOverJson:off - schema-validated study seeds.
                  T3RL_SEED_SET_JSON: JSON.stringify(input.seeds),
                }),
            ...(input.evaluationProtocol === undefined
              ? {}
              : {
                  // @effect-diagnostics-next-line preferSchemaOverJson:off - schema-validated study protocol.
                  T3RL_EVALUATION_PROTOCOL_JSON: JSON.stringify(input.evaluationProtocol),
                }),
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

  const start: RlManagerShape["start"] = (input) => startWithContinuation(input, null);

  const continueRun = (
    input: ContinueRunInput,
    relation: RlLineageRelation,
  ): Effect.Effect<{ readonly runId: string }, RlRunStartError> =>
    Effect.gen(function* () {
      const continuation = yield* prepareContinuation(input, relation);
      return yield* startWithContinuation(
        {
          projectId: input.projectId,
          experimentId: continuation.experimentId,
          seed: continuation.seed,
          ...(continuation.seeds === undefined ? {} : { seeds: continuation.seeds }),
          ...(continuation.evaluationProtocol === undefined
            ? {}
            : { evaluationProtocol: continuation.evaluationProtocol }),
          requestId: input.requestId,
        },
        continuation,
      );
    });

  const resume: RlManagerShape["resume"] = (input) => continueRun(input, "resume");
  const warmStart: RlManagerShape["warmStart"] = (input) => continueRun(input, "warm-start");

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
      const artifacts = yield* store
        .listArtifacts({ runId: input.runId, limit: RL_MAX_SNAPSHOT_ARTIFACTS })
        .pipe(
          Effect.map((page) => page.artifacts),
          Effect.orDie,
        );
      const metrics = yield* store
        .listMetrics({ runId: input.runId, limit: RL_MAX_SNAPSHOT_METRIC_BATCHES })
        .pipe(Effect.orDie);
      const lineage = yield* store.getLineage({ runId: input.runId }).pipe(Effect.orDie);
      return { summary: detail.summary, manifest: detail.manifest, lineage, artifacts, metrics };
    });

  const listArtifacts: RlManagerShape["listArtifacts"] = (input) =>
    Effect.gen(function* () {
      yield* store
        .getRun({ runId: input.runId })
        .pipe(Effect.catchTag("PersistenceSqlError", Effect.orDie));
      const requestedLimit = input.limit ?? DEFAULT_LIST_LIMIT;
      const limit =
        requestedLimit > 0
          ? Math.min(requestedLimit, RL_MAX_ARTIFACT_PAGE_SIZE)
          : DEFAULT_LIST_LIMIT;
      return yield* store
        .listArtifacts({
          runId: input.runId,
          limit,
          ...(input.cursor === undefined ? {} : { cursor: input.cursor }),
        })
        .pipe(Effect.orDie);
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

  const waitForTerminal = (runId: string) =>
    Effect.gen(function* () {
      const done = yield* Deferred.make<RlRunState>();
      const unsubscribe = yield* subscribe({ runId }, (event) => {
        if (
          (event._tag === "Snapshot" || event._tag === "Lifecycle") &&
          isTerminalRlRunState(event.summary.state)
        ) {
          Deferred.doneUnsafe(done, Effect.succeed(event.summary.state));
        }
      });
      return yield* Deferred.await(done).pipe(Effect.ensuring(Effect.sync(unsubscribe)));
    });

  const createStudy: RlManagerShape["createStudy"] = (input) =>
    Effect.gen(function* () {
      const runCount = input.definition.variants.length * input.definition.seeds.length;
      if (runCount > input.definition.maxRuns) {
        return yield* new RlRunStartError({
          code: "InvalidExperiment",
          detail: `Study declares ${runCount} runs but its budget permits ${input.definition.maxRuns}.`,
        });
      }
      const labels = input.definition.variants.map((variant) => variant.label);
      const trainingSeeds = input.definition.seeds.map((seeds) => seeds.training);
      if (
        new Set(labels).size !== labels.length ||
        new Set(trainingSeeds).size !== trainingSeeds.length
      ) {
        return yield* new RlRunStartError({
          code: "InvalidExperiment",
          detail: "Study variant labels and training seeds must be unique.",
        });
      }
      const { protocolSha256, ...protocolBody } = input.definition.evaluationProtocol;
      if (new Set(protocolBody.sampleIds).size !== protocolBody.sampleIds.length) {
        return yield* new RlRunStartError({
          code: "InvalidExperiment",
          detail: "Evaluation protocol sample IDs must be unique.",
        });
      }
      const authoritativeProtocolSha256 = yield* sha256Text(canonicalJson(protocolBody));
      if (protocolSha256 !== authoritativeProtocolSha256) {
        return yield* new RlRunStartError({
          code: "InvalidExperiment",
          detail: `Evaluation protocol hash mismatch; expected ${authoritativeProtocolSha256}.`,
        });
      }
      const uuid = yield* crypto.randomUUIDv4.pipe(Effect.orDie);
      const at = yield* now;
      const study: RlStudy = {
        studyId: `study_${uuid.replace(/-/g, "").slice(0, 22)}`,
        projectId: input.projectId,
        state: "requested",
        definition: input.definition,
        protocolSha256: input.definition.evaluationProtocol.protocolSha256,
        createdAt: at,
        updatedAt: at,
        runs: input.definition.seeds.flatMap((seeds) =>
          input.definition.variants.map((variant) => ({
            variantLabel: variant.label,
            seeds,
            runId: null,
            state: "queued" as const,
          })),
        ),
      };
      yield* store.createStudy(study).pipe(Effect.orDie);
      const statuses = new Map(
        study.runs.map((entry) => [`${entry.variantLabel}:${entry.seeds.training}`, entry.state]),
      );
      const schedule = Effect.forEach(
        study.runs,
        (member) =>
          Effect.gen(function* () {
            const variant = input.definition.variants.find(
              (entry) => entry.label === member.variantLabel,
            )!;
            const key = `${member.variantLabel}:${member.seeds.training}`;
            const started = yield* Effect.result(
              start({
                projectId: input.projectId,
                experimentId: variant.experimentId,
                seed: member.seeds.training,
                seeds: member.seeds,
                evaluationProtocol: input.definition.evaluationProtocol,
                requestId: `${study.studyId}_${member.variantLabel}_${member.seeds.training}`.slice(
                  0,
                  64,
                ),
              }),
            );
            if (Result.isFailure(started)) {
              statuses.set(key, "failed");
              const values = [...statuses.values()];
              const settled = values.every(
                (state) => state === "completed" || state === "failed" || state === "cancelled",
              );
              const successes = values.filter((state) => state === "completed").length;
              yield* store
                .updateStudyRun({
                  studyId: study.studyId,
                  variantLabel: member.variantLabel,
                  trainingSeed: member.seeds.training,
                  state: "failed",
                  studyState: !settled ? "running" : successes === 0 ? "failed" : "partial",
                  at: yield* now,
                })
                .pipe(Effect.orDie);
              return;
            }
            statuses.set(key, "running");
            yield* store
              .updateStudyRun({
                studyId: study.studyId,
                variantLabel: member.variantLabel,
                trainingSeed: member.seeds.training,
                runId: started.success.runId,
                state: "running",
                studyState: "running",
                at: yield* now,
              })
              .pipe(Effect.orDie);
            const terminal = yield* waitForTerminal(started.success.runId);
            const memberState =
              terminal === "completed"
                ? "completed"
                : terminal === "cancelled"
                  ? "cancelled"
                  : "failed";
            statuses.set(key, memberState);
            const values = [...statuses.values()];
            const settled = values.every(
              (state) => state === "completed" || state === "failed" || state === "cancelled",
            );
            const successes = values.filter((state) => state === "completed").length;
            const studyState = !settled
              ? "running"
              : successes === values.length
                ? "completed"
                : successes === 0
                  ? "failed"
                  : "partial";
            yield* store
              .updateStudyRun({
                studyId: study.studyId,
                variantLabel: member.variantLabel,
                trainingSeed: member.seeds.training,
                state: memberState,
                studyState,
                at: yield* now,
              })
              .pipe(Effect.orDie);
          }),
        { concurrency: input.definition.maxConcurrency, discard: true },
      );
      yield* Effect.forkIn(schedule, scope);
      return study;
    });

  const getStudy: RlManagerShape["getStudy"] = (input) =>
    store.getStudy(input).pipe(Effect.catchTag("PersistenceSqlError", Effect.orDie));

  const compareStudy: RlManagerShape["compareStudy"] = (input) =>
    Effect.gen(function* () {
      const study = yield* getStudy({ studyId: input.studyId });
      const { protocolSha256, ...protocolBody } = study.definition.evaluationProtocol;
      const protocolMatches =
        protocolSha256 === study.protocolSha256 &&
        protocolSha256 === (yield* sha256Text(canonicalJson(protocolBody))) &&
        new Set(protocolBody.sampleIds).size === protocolBody.sampleIds.length;
      const validVariants =
        input.baselineLabel !== input.candidateLabel &&
        [input.baselineLabel, input.candidateLabel].every((label) =>
          study.definition.variants.some((variant) => variant.label === label),
        );
      const members = study.runs.filter(
        (member) =>
          member.variantLabel === input.baselineLabel ||
          member.variantLabel === input.candidateLabel,
      );
      const collected = validVariants
        ? yield* Effect.forEach(
            members,
            (member) =>
              collectStudyObservation({
                study,
                member,
                metricKey: input.metricKey,
                rlRunsDir: config.rlRunsDir,
              }).pipe(
                Effect.provideService(RunStore, store),
                Effect.provideService(FileSystem.FileSystem, fs),
                Effect.provideService(Path.Path, path),
                Effect.provideService(Crypto.Crypto, crypto),
              ),
            { concurrency: 4 },
          )
        : [];
      const excludedRuns = collected.flatMap((entry) =>
        entry._tag === "Excluded" ? [entry.excluded] : [],
      );
      const collect = (label: string) =>
        collected.flatMap((entry, index) =>
          entry._tag === "Observation" && members[index]?.variantLabel === label
            ? [entry.observation]
            : [],
        );
      const comparison = yield* comparePairedStudyEffect({
        studyId: study.studyId,
        protocolSha256: study.protocolSha256,
        baselineLabel: input.baselineLabel,
        candidateLabel: input.candidateLabel,
        metricKey: input.metricKey,
        baseline: collect(input.baselineLabel),
        candidate: collect(input.candidateLabel),
        failedRuns: excludedRuns.filter(
          (run) =>
            run.reason === "failed" || run.reason === "cancelled" || run.reason === "interrupted",
        ).length,
        excludedRuns,
        expectedSeeds: study.definition.seeds.map((seeds) => seeds.training),
        expectedSampleIds: study.definition.evaluationProtocol.sampleIds,
        generationSeedPolicy: study.definition.evaluationProtocol.generationSeedPolicy,
        estimator: input.estimator,
        compatibleProtocol:
          protocolMatches && !excludedRuns.some((run) => run.reason === "incompatible-protocol"),
      });
      return validVariants
        ? comparison
        : { ...comparison, conclusion: "invalid-variants" as const };
    });

  const validateExperiment: RlManagerShape["validateExperiment"] = (input) =>
    Effect.gen(function* () {
      const source = yield* Effect.result(sourceEvidence.resolve(input.projectId));
      if (Result.isFailure(source)) {
        return {
          experimentId: input.experimentId,
          namespace: "project",
          valid: false,
          issues: [
            {
              severity: "error",
              code: "project-unavailable",
              message: source.failure.detail,
              path: null,
            },
          ],
          resolvedInputs: [],
          supportedOperations: [],
        };
      }
      const report = yield* experiments.validate({
        experimentId: input.experimentId,
        workspaceRoot: source.success.workspaceRoot,
      });
      if (!report.valid) return report;
      const definition = yield* experiments
        .resolve({ experimentId: input.experimentId, workspaceRoot: source.success.workspaceRoot })
        .pipe(Effect.option);
      if (definition._tag === "None") return report;
      const runner = yield* Effect.result(
        capabilities.resolveRunner({
          runnerId: definition.value.runnerId,
          ...(definition.value.method === undefined ? {} : { method: definition.value.method }),
        }),
      );
      return Result.isSuccess(runner)
        ? report
        : {
            ...report,
            valid: false,
            issues: [
              ...report.issues,
              {
                severity: "error" as const,
                code: "runner-unavailable",
                message: runner.failure.detail,
                path: null,
              },
            ],
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
    resume,
    warmStart,
    cancel,
    list,
    get,
    listArtifacts,
    subscribe,
    sweepInterruptedRuns,
    createStudy,
    getStudy,
    compareStudy,
    validateExperiment,
  });
});

export const RlManagerLive = Layer.effect(RlManager, makeManager);
