import * as Schema from "effect/Schema";
import { TrimmedNonEmptyString } from "./baseSchemas.ts";

/** Worker protocol version this server speaks. A worker announcing anything else is rejected. */
export const RL_WORKER_PROTOCOL_VERSION = 1;

const IdentifierSchema = TrimmedNonEmptyString.check(Schema.isPattern(/^[A-Za-z0-9_-]+$/)).check(
  Schema.isMaxLength(64),
);

export const RlRunId = IdentifierSchema;
export type RlRunId = typeof RlRunId.Type;

export const RlExperimentId = IdentifierSchema;
export type RlExperimentId = typeof RlExperimentId.Type;

export const RlRunState = Schema.Literals([
  "requested",
  "preparing",
  "running",
  "cancelling",
  "completed",
  "failed",
  "cancelled",
  "interrupted",
]);
export type RlRunState = typeof RlRunState.Type;

/** States that never transition again. Kept beside the schema so the two cannot drift. */
export const RL_TERMINAL_RUN_STATES = [
  "completed",
  "failed",
  "cancelled",
  "interrupted",
] as const satisfies ReadonlyArray<RlRunState>;

export const isTerminalRlRunState = (state: RlRunState): boolean =>
  (RL_TERMINAL_RUN_STATES as ReadonlyArray<RlRunState>).includes(state);

export const RlErrorCode = Schema.Literals([
  "PythonNotFound",
  "RunnerUnavailable",
  "ProtocolIncompatible",
  "MalformedWorkerMessage",
  "WorkerStalled",
  "WorkerExited",
  "RunNotFound",
  "ArtifactNotFound",
]);
export type RlErrorCode = typeof RlErrorCode.Type;

const METRIC_KEY_PATTERN = /^[A-Za-z0-9_]+(\/[A-Za-z0-9_]+)*$/;
const MAX_METRIC_KEY_LENGTH = 64;

/**
 * `Schema.Record` validates values but not keys, so key shape is enforced with
 * an explicit filter over the whole record. Metric names reach chart axes and
 * grouping logic; an unchecked key is an injection surface, and a check placed
 * on the key schema alone would silently do nothing.
 */
const boundedMetricKeys = Schema.makeFilter((values: Record<string, unknown>) => {
  for (const key of Object.keys(values)) {
    if (key.length > MAX_METRIC_KEY_LENGTH) {
      return `Metric key exceeds ${MAX_METRIC_KEY_LENGTH} characters: ${key}`;
    }
    if (!METRIC_KEY_PATTERN.test(key)) {
      return `Metric key is not a valid metric name: ${key}`;
    }
  }
  return true;
});

/** Same reasoning as `boundedMetricKeys`: record keys need their own filter. */
const boundedConfigKeys = Schema.makeFilter((values: Record<string, unknown>) => {
  for (const key of Object.keys(values)) {
    if (key.length > 64) {
      return `Config key exceeds 64 characters: ${key}`;
    }
  }
  return true;
});

/**
 * A metric value is a finite number, `null` for "not captured", or an explicit
 * non-finite marker. JSON cannot carry NaN or Infinity, and encoding them as
 * `null` would make an unrecorded value indistinguishable from a diverged one.
 */
const NonFiniteMarkerSchema = Schema.Literals(["nan", "+inf", "-inf"]);
const MetricValueSchema = Schema.NullOr(Schema.Union([Schema.Number, NonFiniteMarkerSchema]));

export const RlMetricBatch = Schema.Struct({
  step: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  wallClockMs: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  values: Schema.Record(Schema.String, MetricValueSchema)
    .check(Schema.isMaxProperties(64))
    .check(boundedMetricKeys),
});
export type RlMetricBatch = typeof RlMetricBatch.Type;

export const RlArtifactKind = Schema.Literals([
  "manifest",
  "log",
  "summary",
  "model",
  "evaluation",
  "replay",
]);
export type RlArtifactKind = typeof RlArtifactKind.Type;

export const RlArtifactMetadata = Schema.Struct({
  artifactId: IdentifierSchema,
  kind: RlArtifactKind,
  bytes: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  contentType: Schema.String.check(Schema.isMaxLength(128)),
  producedAt: Schema.String,
});
export type RlArtifactMetadata = typeof RlArtifactMetadata.Type;

/** What actually ran. Never edited after the run leaves `preparing`. */
export const RlResolvedManifest = Schema.Struct({
  experimentId: RlExperimentId,
  runnerId: IdentifierSchema,
  runnerVersion: Schema.String.check(Schema.isMaxLength(64)),
  protocolVersion: Schema.Int,
  seed: Schema.Int,
  effectiveConfig: Schema.Record(Schema.String, Schema.Unknown)
    .check(Schema.isMaxProperties(128))
    .check(boundedConfigKeys),
  sourceRevision: Schema.NullOr(Schema.String.check(Schema.isMaxLength(64))),
  sourceDirty: Schema.Boolean,
  pythonExecutable: Schema.String.check(Schema.isMaxLength(1024)),
  pythonVersion: Schema.String.check(Schema.isMaxLength(64)),
  environmentFingerprint: Schema.String.check(Schema.isMaxLength(128)),
  instrumentationLevel: Schema.Literals(["minimal", "standard", "deep"]),
  hardwareSummary: Schema.String.check(Schema.isMaxLength(256)),
});
export type RlResolvedManifest = typeof RlResolvedManifest.Type;

export const RlRunSummary = Schema.Struct({
  runId: RlRunId,
  projectId: TrimmedNonEmptyString.check(Schema.isMaxLength(128)),
  experimentId: RlExperimentId,
  state: RlRunState,
  requestedAt: Schema.String,
  startedAt: Schema.NullOr(Schema.String),
  endedAt: Schema.NullOr(Schema.String),
  lastMessageAt: Schema.NullOr(Schema.String),
  errorCode: Schema.NullOr(RlErrorCode),
  errorMessage: Schema.NullOr(Schema.String.check(Schema.isMaxLength(2048))),
});
export type RlRunSummary = typeof RlRunSummary.Type;

export const RlRunnerCapability = Schema.Struct({
  runnerId: IdentifierSchema,
  available: Schema.Boolean,
  version: Schema.NullOr(Schema.String.check(Schema.isMaxLength(64))),
  failureCode: Schema.NullOr(RlErrorCode),
  /** Actionable text shown to the researcher. The lab never installs anything itself. */
  remedy: Schema.NullOr(Schema.String.check(Schema.isMaxLength(512))),
});
export type RlRunnerCapability = typeof RlRunnerCapability.Type;

export const RlCapabilityReport = Schema.Struct({
  runners: Schema.Array(RlRunnerCapability).check(Schema.isMaxLength(16)),
});
export type RlCapabilityReport = typeof RlCapabilityReport.Type;

/** What a subscriber receives: one snapshot, then live facts. */
export const RlSubscriptionEvent = Schema.Union([
  Schema.TaggedStruct("Snapshot", {
    summary: RlRunSummary,
    manifest: Schema.NullOr(RlResolvedManifest),
    artifacts: Schema.Array(RlArtifactMetadata),
  }),
  Schema.TaggedStruct("Lifecycle", { summary: RlRunSummary }),
  Schema.TaggedStruct("Metrics", { batch: RlMetricBatch }),
]);
export type RlSubscriptionEvent = typeof RlSubscriptionEvent.Type;

export class RlRunNotFoundError extends Schema.TaggedErrorClass<RlRunNotFoundError>()(
  "RlRunNotFoundError",
  { runId: Schema.String },
) {
  override get message() {
    return `RL run not found: ${this.runId}`;
  }
}

export class RlArtifactNotFoundError extends Schema.TaggedErrorClass<RlArtifactNotFoundError>()(
  "RlArtifactNotFoundError",
  { runId: Schema.String, artifactId: Schema.String },
) {
  override get message() {
    return `RL artifact not found: ${this.artifactId} in run ${this.runId}`;
  }
}

export class RlRunStartError extends Schema.TaggedErrorClass<RlRunStartError>()("RlRunStartError", {
  code: RlErrorCode,
  detail: Schema.String,
}) {
  override get message() {
    return `Failed to start RL run (${this.code}): ${this.detail}`;
  }
}
