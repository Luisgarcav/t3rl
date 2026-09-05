import * as Schema from "effect/Schema";
import { TrimmedNonEmptyString } from "./baseSchemas.ts";

/** Latest worker protocol emitted by post-training adapters. */
export const RL_WORKER_PROTOCOL_VERSION = 2;

/** Protocols the server can still supervise during a rolling client/worker upgrade. */
export const RL_SUPPORTED_WORKER_PROTOCOL_VERSIONS = [1, 2] as const;

const IdentifierSchema = TrimmedNonEmptyString.check(Schema.isPattern(/^[A-Za-z0-9_-]+$/)).check(
  Schema.isMaxLength(64),
);

export const RlRunId = IdentifierSchema;
export type RlRunId = typeof RlRunId.Type;

export const RlRunRequestId = IdentifierSchema;
export type RlRunRequestId = typeof RlRunRequestId.Type;

export const RlArtifactId = IdentifierSchema;
export type RlArtifactId = typeof RlArtifactId.Type;

export const RlExperimentId = IdentifierSchema;
export type RlExperimentId = typeof RlExperimentId.Type;

export const RlStudyId = IdentifierSchema;
export type RlStudyId = typeof RlStudyId.Type;

export const RlSeedSet = Schema.Struct({
  training: Schema.Int,
  data: Schema.Int,
  evaluationSample: Schema.Int,
  generation: Schema.Int,
});
export type RlSeedSet = typeof RlSeedSet.Type;

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
  "InvalidExperiment",
  "RunnerException",
  "ProtocolIncompatible",
  "MalformedWorkerMessage",
  "WorkerMessageTooLarge",
  "WorkerHelloTimeout",
  "WorkerStalled",
  "WorkerExited",
  "ServerInterrupted",
  "RunNotFound",
  "ArtifactNotFound",
  "ResumeIncompatible",
]);
export type RlErrorCode = typeof RlErrorCode.Type;

/** Wire and storage bounds shared by the server and every client surface. */
export const RL_MAX_RUN_ARTIFACTS = 4096;
export const RL_MAX_SNAPSHOT_ARTIFACTS = 64;
export const RL_MAX_ARTIFACT_PAGE_SIZE = 200;
export const RL_MAX_SNAPSHOT_METRIC_BATCHES = 4096;

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
  "config",
  "checkpoint",
  "adapter",
  "dataset",
  "source",
  "export",
]);
export type RlArtifactKind = typeof RlArtifactKind.Type;

export const RlArtifactState = Schema.Literals(["ready", "trashed", "purged"]);
export type RlArtifactState = typeof RlArtifactState.Type;

export const RlSha256 = Schema.String.check(Schema.isPattern(/^[0-9a-f]{64}$/));
export type RlSha256 = typeof RlSha256.Type;

export const RlEvaluationProtocol = Schema.Struct({
  version: Schema.Literal(1),
  protocolSha256: RlSha256,
  datasetFingerprint: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(256)),
  split: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(64)),
  sampleIds: Schema.Array(Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(256)))
    .check(Schema.isMinLength(1))
    .check(Schema.isMaxLength(10_000)),
  generationSeedPolicy: Schema.Literals(["fixed-per-sample", "per-run"]),
  decoding: Schema.Record(Schema.String, Schema.Unknown).check(Schema.isMaxProperties(32)),
  verifierSha256: RlSha256,
});
export type RlEvaluationProtocol = typeof RlEvaluationProtocol.Type;

const ImmutableModelRevision = Schema.String.check(Schema.isPattern(/^[0-9a-f]{40,64}$/)).check(
  Schema.isMaxLength(64),
);

const ModuleName = Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(128));

/** The LoRA configuration needed to interpret and independently load an adapter. */
export const RlPeftConfig = Schema.Struct({
  peftType: Schema.Literals(["LORA"]),
  taskType: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(64)),
  rank: Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 4096 })),
  alpha: Schema.Number.check(Schema.isBetween({ minimum: 0, maximum: 1_000_000 })),
  dropout: Schema.Number.check(Schema.isBetween({ minimum: 0, maximum: 1 })),
  bias: Schema.Literals(["none", "all", "lora_only"]),
  targetModules: Schema.Array(ModuleName)
    .check(Schema.isMinLength(1))
    .check(Schema.isMaxLength(128)),
  modulesToSave: Schema.Array(ModuleName).check(Schema.isMaxLength(128)),
  useRslora: Schema.Boolean,
});
export type RlPeftConfig = typeof RlPeftConfig.Type;

/** Immutable model and trainable-surface identity shared by manifests and artifacts. */
export const RlModelIdentity = Schema.Struct({
  baseModelId: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(256)),
  baseModelRevision: ImmutableModelRevision,
  tokenizerRevision: ImmutableModelRevision,
  peftConfig: RlPeftConfig,
  peftConfigSha256: RlSha256,
  quantization: Schema.Literals(["none", "4-bit", "8-bit"]),
  precision: Schema.Literals(["fp32", "fp16", "bf16"]),
  trainableModules: Schema.Array(ModuleName)
    .check(Schema.isMinLength(1))
    .check(Schema.isMaxLength(256)),
});
export type RlModelIdentity = typeof RlModelIdentity.Type;

/** Resolved checkpoint lifecycle, including the cancellation grace period. */
export const RlCheckpointPolicy = Schema.Struct({
  cadenceSteps: Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 1_000_000 })),
  maxIntermediateCheckpoints: Schema.Int.check(Schema.isBetween({ minimum: 0, maximum: 64 })),
  keepBest: Schema.Boolean,
  keepFinal: Schema.Boolean,
  gracefulDeadlineSeconds: Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 3600 })),
});
export type RlCheckpointPolicy = typeof RlCheckpointPolicy.Type;

export const RlCheckpointCompatibility = Schema.Struct({
  model: RlModelIdentity,
  framework: Schema.Struct({
    id: IdentifierSchema,
    version: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(64)),
  }),
  environmentFingerprint: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(128)),
  environmentLockSha256: Schema.NullOr(RlSha256),
});
export type RlCheckpointCompatibility = typeof RlCheckpointCompatibility.Type;

export const RlDatasetCursor = Schema.Struct({
  epoch: Schema.Number.check(Schema.isGreaterThanOrEqualTo(0)),
  batchInEpoch: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  sampleOffset: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
});
export type RlDatasetCursor = typeof RlDatasetCursor.Type;

/** Exact-resume state. Every boolean is independently checked before a child is created. */
export const RlResumeStateEvidence = Schema.Struct({
  trainerState: Schema.Boolean,
  optimizerState: Schema.Boolean,
  schedulerState: Schema.Boolean,
  rngState: Schema.Boolean,
  datasetCursorState: Schema.Boolean,
  gradientScalerState: Schema.Literals(["captured", "not-applicable"]),
  stateFiles: Schema.Array(Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(512)))
    .check(Schema.isMinLength(4))
    .check(Schema.isMaxLength(64)),
});
export type RlResumeStateEvidence = typeof RlResumeStateEvidence.Type;

export const RlCheckpointArtifactEvidence = Schema.TaggedStruct("Checkpoint", {
  checkpointClass: Schema.Literals(["intermediate", "best", "final", "graceful"]),
  compatibility: RlCheckpointCompatibility,
  globalStep: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  tokensSeen: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  datasetCursor: RlDatasetCursor,
  resumeState: RlResumeStateEvidence,
});
export type RlCheckpointArtifactEvidence = typeof RlCheckpointArtifactEvidence.Type;

export const RlAdapterArtifactEvidence = Schema.TaggedStruct("Adapter", {
  compatibility: RlCheckpointCompatibility,
  globalStep: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  tokensSeen: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  datasetCursor: RlDatasetCursor,
});
export type RlAdapterArtifactEvidence = typeof RlAdapterArtifactEvidence.Type;

export const RlArtifactEvidence = Schema.Union([
  RlCheckpointArtifactEvidence,
  RlAdapterArtifactEvidence,
]);
export type RlArtifactEvidence = typeof RlArtifactEvidence.Type;

export const RlLineageRelation = Schema.Literals(["resume", "warm-start"]);
export type RlLineageRelation = typeof RlLineageRelation.Type;

/** One immutable edge from a new child to the exact evidence it started from. */
export const RlLineageEdge = Schema.Struct({
  childRunId: RlRunId,
  parentRunId: RlRunId,
  sourceArtifactId: RlArtifactId,
  sourceArtifactSha256: RlSha256,
  relation: RlLineageRelation,
  sourceStep: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  createdAt: Schema.String,
});
export type RlLineageEdge = typeof RlLineageEdge.Type;

export const RlRunLineage = Schema.Struct({
  edges: Schema.Array(RlLineageEdge).check(Schema.isMaxLength(64)),
  truncated: Schema.Boolean,
});
export type RlRunLineage = typeof RlRunLineage.Type;

export const RlArtifactMetadata = Schema.Struct({
  artifactId: RlArtifactId,
  kind: RlArtifactKind,
  bytes: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  contentType: Schema.String.check(Schema.isMaxLength(128)),
  producedAt: Schema.String,
  /** Absent on older servers; null marks readable legacy evidence that was never verified. */
  sha256: Schema.optionalKey(Schema.NullOr(RlSha256)),
  logicalName: Schema.optionalKey(Schema.String.check(Schema.isMaxLength(512))),
  format: Schema.optionalKey(Schema.String.check(Schema.isMaxLength(64))),
  state: Schema.optionalKey(RlArtifactState),
  checkpointStep: Schema.optionalKey(
    Schema.NullOr(Schema.Int.check(Schema.isGreaterThanOrEqualTo(0))),
  ),
  fileCount: Schema.optionalKey(Schema.Int.check(Schema.isGreaterThanOrEqualTo(0))),
  /** Absent on legacy artifacts. Protocol-v2 checkpoints/adapters require it. */
  evidence: Schema.optionalKey(Schema.NullOr(RlArtifactEvidence)),
});
export type RlArtifactMetadata = typeof RlArtifactMetadata.Type;

export const RlArtifactPage = Schema.Struct({
  artifacts: Schema.Array(RlArtifactMetadata).check(Schema.isMaxLength(RL_MAX_ARTIFACT_PAGE_SIZE)),
  nextCursor: Schema.NullOr(RlArtifactId),
});
export type RlArtifactPage = typeof RlArtifactPage.Type;

export const RlEnvironmentLock = Schema.Struct({
  projectPath: Schema.String.check(Schema.isMaxLength(2048)),
  lockfilePath: Schema.String.check(Schema.isMaxLength(2048)),
  lockfileSha256: RlSha256,
  pythonExecutable: Schema.String.check(Schema.isMaxLength(1024)),
  pythonVersion: Schema.String.check(Schema.isMaxLength(64)),
  platform: Schema.String.check(Schema.isMaxLength(256)),
  framework: Schema.Struct({
    id: IdentifierSchema,
    version: Schema.String.check(Schema.isMaxLength(64)),
  }),
  pytorchVersion: Schema.NullOr(Schema.String.check(Schema.isMaxLength(64))),
  cudaAvailable: Schema.Boolean,
  cudaRuntime: Schema.NullOr(Schema.String.check(Schema.isMaxLength(64))),
  cudaDeviceCount: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  driverVersion: Schema.NullOr(Schema.String.check(Schema.isMaxLength(64))),
});
export type RlEnvironmentLock = typeof RlEnvironmentLock.Type;

/** What actually ran. Never edited after the run leaves `preparing`. */
export const RlResolvedManifest = Schema.Struct({
  experimentId: RlExperimentId,
  runnerId: IdentifierSchema,
  runnerVersion: Schema.String.check(Schema.isMaxLength(64)),
  protocolVersion: Schema.Int,
  seed: Schema.Int,
  /** Explicit independent seeds; absent on protocol-v1 and pre-study runs. */
  seeds: Schema.optionalKey(RlSeedSet),
  effectiveConfig: Schema.Record(Schema.String, Schema.Unknown)
    .check(Schema.isMaxProperties(128))
    .check(boundedConfigKeys),
  sourceRevision: Schema.NullOr(Schema.String.check(Schema.isMaxLength(64))),
  /** Null means Git evidence was unavailable; it must never be presented as a clean tree. */
  sourceDirty: Schema.NullOr(Schema.Boolean),
  pythonExecutable: Schema.String.check(Schema.isMaxLength(1024)),
  pythonVersion: Schema.String.check(Schema.isMaxLength(64)),
  environmentFingerprint: Schema.String.check(Schema.isMaxLength(128)),
  instrumentationLevel: Schema.Literals(["minimal", "standard", "deep"]),
  hardwareSummary: Schema.String.check(Schema.isMaxLength(256)),
  /** Optional so clients and storage continue to decode protocol-v1 manifests. */
  environmentLock: Schema.optionalKey(Schema.NullOr(RlEnvironmentLock)),
  /** Protocol-v2 post-training identity. */
  model: Schema.optionalKey(RlModelIdentity),
  checkpointPolicy: Schema.optionalKey(RlCheckpointPolicy),
  /** Direct parent edge; the full ancestor chain is returned alongside run detail. */
  lineage: Schema.optionalKey(RlLineageEdge),
});
export type RlResolvedManifest = typeof RlResolvedManifest.Type;

export const RlStudyState = Schema.Literals([
  "requested",
  "running",
  "partial",
  "completed",
  "failed",
  "cancelled",
]);
export type RlStudyState = typeof RlStudyState.Type;

export const RlStudyVariant = Schema.Struct({
  label: IdentifierSchema,
  experimentId: RlExperimentId,
});
export type RlStudyVariant = typeof RlStudyVariant.Type;

export const RlStudyDefinition = Schema.Struct({
  variants: Schema.Array(RlStudyVariant).check(Schema.isMinLength(2)).check(Schema.isMaxLength(8)),
  seeds: Schema.Array(RlSeedSet).check(Schema.isMinLength(2)).check(Schema.isMaxLength(64)),
  maxConcurrency: Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 16 })),
  maxRuns: Schema.Int.check(Schema.isBetween({ minimum: 2, maximum: 512 })),
  evaluationProtocol: RlEvaluationProtocol,
});
export type RlStudyDefinition = typeof RlStudyDefinition.Type;

export const RlStudyRun = Schema.Struct({
  variantLabel: IdentifierSchema,
  seeds: RlSeedSet,
  runId: Schema.NullOr(RlRunId),
  state: Schema.Literals(["queued", "running", "completed", "failed", "cancelled"]),
});
export type RlStudyRun = typeof RlStudyRun.Type;

export const RlStudy = Schema.Struct({
  studyId: RlStudyId,
  projectId: TrimmedNonEmptyString.check(Schema.isMaxLength(128)),
  state: RlStudyState,
  definition: RlStudyDefinition,
  protocolSha256: RlSha256,
  createdAt: Schema.String,
  updatedAt: Schema.String,
  runs: Schema.Array(RlStudyRun).check(Schema.isMaxLength(512)),
});
export type RlStudy = typeof RlStudy.Type;

export const RlStudyEstimator = Schema.Struct({
  version: Schema.Literal(1),
  statistic: Schema.Literals(["mean", "median", "paired-mean-delta"]),
  statisticalUnit: Schema.Literals(["run-seed", "paired-sample-within-run-seed"]),
  confidenceLevel: Schema.Number.check(Schema.isBetween({ minimum: 0.5, maximum: 0.999 })),
  resamplingSeed: Schema.Int,
  resampleCount: Schema.Int.check(Schema.isBetween({ minimum: 100, maximum: 100_000 })),
  missingPairPolicy: Schema.Literals(["exclude", "fail"]),
});
export type RlStudyEstimator = typeof RlStudyEstimator.Type;

export const RlStudyComparison = Schema.Struct({
  studyId: RlStudyId,
  protocolSha256: RlSha256,
  baselineLabel: IdentifierSchema,
  candidateLabel: IdentifierSchema,
  metricKey: Schema.String.check(Schema.isPattern(METRIC_KEY_PATTERN)),
  estimator: RlStudyEstimator,
  seedSet: Schema.Array(Schema.Int).check(Schema.isMaxLength(64)),
  n: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  unmatchedRuns: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  failedRuns: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  baselineMean: Schema.NullOr(Schema.Number),
  candidateMean: Schema.NullOr(Schema.Number),
  pairedDelta: Schema.NullOr(Schema.Number),
  dispersion: Schema.NullOr(Schema.Number),
  interval: Schema.NullOr(Schema.Tuple([Schema.Number, Schema.Number])),
  conclusion: Schema.Literals(["interval", "not-enough-evidence", "incompatible-protocol"]),
});
export type RlStudyComparison = typeof RlStudyComparison.Type;

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
  methods: Schema.optionalKey(
    Schema.Array(Schema.Literals(["sft", "dpo", "grpo", "rloo", "ppo"])).check(
      Schema.isMaxLength(5),
    ),
  ),
  methodCapabilities: Schema.optionalKey(
    Schema.Array(
      Schema.Struct({
        method: Schema.Literals(["sft", "dpo", "grpo", "rloo", "ppo"]),
        available: Schema.Boolean,
        remedy: Schema.NullOr(Schema.String.check(Schema.isMaxLength(512))),
      }),
    ).check(Schema.isMaxLength(5)),
  ),
});
export type RlRunnerCapability = typeof RlRunnerCapability.Type;

export const RlExperimentSummary = Schema.Struct({
  experimentId: RlExperimentId,
  displayName: Schema.String.check(Schema.isMaxLength(128)),
  description: Schema.String.check(Schema.isMaxLength(512)),
  runnerId: IdentifierSchema,
  defaultSeed: Schema.Int,
  instrumentationLevel: Schema.Literals(["minimal", "standard", "deep"]),
  config: Schema.Record(Schema.String, Schema.Unknown)
    .check(Schema.isMaxProperties(128))
    .check(boundedConfigKeys),
  method: Schema.optionalKey(Schema.Literals(["sft", "dpo", "grpo", "rloo", "ppo"])),
});
export type RlExperimentSummary = typeof RlExperimentSummary.Type;

export const RlProjectExperimentMode = Schema.Literals(["reproducible", "exploratory"]);
export const RlProjectInputReference = Schema.Union([
  Schema.TaggedStruct("ProjectFile", {
    path: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(512)),
  }),
  Schema.TaggedStruct("External", {
    id: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(256)),
    revision: Schema.NullOr(
      Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(128)),
    ),
  }),
]);
export type RlProjectInputReference = typeof RlProjectInputReference.Type;

export const RlProjectExperimentDefinition = Schema.Struct({
  version: Schema.Literal(1),
  experimentId: RlExperimentId,
  displayName: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(128)),
  description: Schema.String.check(Schema.isMaxLength(512)),
  mode: RlProjectExperimentMode,
  adapter: Schema.Literals(["trl", "axolotl"]),
  method: Schema.Literals(["sft", "dpo", "grpo"]),
  evaluationClaim: Schema.Literals(["held-out-loss", "preference-accuracy", "verifier-pass-rate"]),
  model: Schema.Struct({
    id: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(256)),
    revision: Schema.NullOr(
      Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(128)),
    ),
    tokenizerRevision: Schema.NullOr(
      Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(128)),
    ),
  }),
  dataset: RlProjectInputReference,
  datasetFormat: Schema.Literals([
    "sft-text",
    "sft-conversation",
    "dpo-preference",
    "rlvr-prompt-answer",
  ]),
  chatTemplate: Schema.Struct({
    source: Schema.Literals(["tokenizer", "project"]),
    sha256: Schema.NullOr(RlSha256),
  }),
  verifier: RlProjectInputReference,
  splitPolicy: Schema.Struct({
    train: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(64)),
    evaluation: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(64)),
  }),
  evaluationProtocol: RlEvaluationProtocol,
  budgets: Schema.Struct({
    maxRuntimeSeconds: Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 86_400 })),
    maxSteps: Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 10_000_000 })),
    maxArtifactBytes: Schema.Int.check(
      Schema.isBetween({ minimum: 1, maximum: 512 * 1024 * 1024 }),
    ),
  }),
  instrumentationLevel: Schema.Literals(["minimal", "standard", "deep"]),
  defaultSeed: Schema.Int,
  config: Schema.Record(Schema.String, Schema.Unknown)
    .check(Schema.isMaxProperties(128))
    .check(boundedConfigKeys),
});
export type RlProjectExperimentDefinition = typeof RlProjectExperimentDefinition.Type;

export const RlExperimentValidationIssue = Schema.Struct({
  severity: Schema.Literals(["warning", "error"]),
  code: IdentifierSchema,
  message: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(1024)),
  path: Schema.NullOr(Schema.String.check(Schema.isMaxLength(512))),
});
export const RlResolvedExperimentInput = Schema.Struct({
  role: Schema.Literals(["definition", "dataset", "verifier"]),
  logicalName: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(512)),
  sha256: Schema.NullOr(RlSha256),
  bytes: Schema.NullOr(Schema.Int.check(Schema.isGreaterThanOrEqualTo(0))),
  externalId: Schema.NullOr(Schema.String.check(Schema.isMaxLength(256))),
  revision: Schema.NullOr(Schema.String.check(Schema.isMaxLength(128))),
});
export const RlExperimentValidationReport = Schema.Struct({
  experimentId: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(128)),
  namespace: Schema.Literals(["bundled", "project"]),
  valid: Schema.Boolean,
  issues: Schema.Array(RlExperimentValidationIssue).check(Schema.isMaxLength(64)),
  resolvedInputs: Schema.Array(RlResolvedExperimentInput).check(Schema.isMaxLength(16)),
  supportedOperations: Schema.Array(
    Schema.Literals(["start", "resume", "warm-start", "study"]),
  ).check(Schema.isMaxLength(4)),
});
export type RlExperimentValidationReport = typeof RlExperimentValidationReport.Type;

export const RlCapabilityReport = Schema.Struct({
  runners: Schema.Array(RlRunnerCapability).check(Schema.isMaxLength(16)),
  experiments: Schema.Array(RlExperimentSummary).check(Schema.isMaxLength(64)),
});
export type RlCapabilityReport = typeof RlCapabilityReport.Type;

/** What a subscriber receives: one snapshot, then live facts. */
export const RlSubscriptionEvent = Schema.Union([
  Schema.TaggedStruct("Snapshot", {
    summary: RlRunSummary,
    manifest: Schema.NullOr(RlResolvedManifest),
    lineage: RlRunLineage,
    artifacts: Schema.Array(RlArtifactMetadata).check(
      Schema.isMaxLength(RL_MAX_SNAPSHOT_ARTIFACTS),
    ),
    metrics: Schema.Array(RlMetricBatch).check(Schema.isMaxLength(RL_MAX_SNAPSHOT_METRIC_BATCHES)),
  }),
  Schema.TaggedStruct("Lifecycle", { summary: RlRunSummary }),
  Schema.TaggedStruct("Manifest", { manifest: RlResolvedManifest }),
  Schema.TaggedStruct("Artifact", { artifact: RlArtifactMetadata }),
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

export class RlStudyNotFoundError extends Schema.TaggedErrorClass<RlStudyNotFoundError>()(
  "RlStudyNotFoundError",
  { studyId: Schema.String },
) {
  override get message() {
    return `RL study not found: ${this.studyId}`;
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
