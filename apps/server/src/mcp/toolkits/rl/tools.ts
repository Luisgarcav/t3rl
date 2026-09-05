import {
  RlArtifactMetadata,
  RlArtifactPage,
  RlCapabilityReport,
  RlExperimentValidationReport,
  RlMetricBatch,
  RlResolvedManifest,
  RlRunLineage,
  RlRunId,
  RlRunState,
  RlRunSummary,
  RlStudy,
  RlStudyComparison,
  RlStudyDefinition,
  RlStudyEstimator,
  RL_MAX_ARTIFACT_PAGE_SIZE,
} from "@t3tools/contracts";
import * as FileSystem from "effect/FileSystem";
import * as Path from "effect/Path";
import * as Schema from "effect/Schema";
import { Tool, Toolkit } from "effect/unstable/ai";

import * as ServerConfig from "../../../config.ts";
import * as ProjectionSnapshotQuery from "../../../orchestration/Services/ProjectionSnapshotQuery.ts";
import * as RlManager from "../../../rl/Manager.ts";
import * as RunStore from "../../../rl/RunStore.ts";
import * as McpInvocationContext from "../../McpInvocationContext.ts";

export const RL_AGENT_ARTIFACT_MAX_BYTES = 1024 * 1024;
export const RL_AGENT_METRIC_QUERY_MAX_BATCHES = 512;

const MetricKey = Schema.String.annotate({
  description: "Exact retained metric key, for example train/return.",
})
  .check(Schema.isTrimmed())
  .check(Schema.isNonEmpty())
  .check(Schema.isMaxLength(64))
  .check(Schema.isPattern(/^[A-Za-z0-9_]+(\/[A-Za-z0-9_]+)*$/));

const IdentifierParameter = (description: string) =>
  Schema.String.annotate({ description })
    .check(Schema.isTrimmed())
    .check(Schema.isNonEmpty())
    .check(Schema.isMaxLength(64))
    .check(Schema.isPattern(/^[A-Za-z0-9_-]+$/));

const RunIdParameter = IdentifierParameter(
  "Stable run ID returned by rl_list_runs, rl_start_run, or another RL tool.",
);

const PositiveLimit = (maximum: number, description: string) =>
  Schema.optional(
    Schema.Int.annotate({ description })
      .check(Schema.isGreaterThan(0))
      .check(Schema.isLessThanOrEqualTo(maximum)),
  );

const NonNegativeInt = Schema.Int.check(Schema.isGreaterThanOrEqualTo(0));

export const RlAgentToolErrorCode = Schema.Literals([
  "capability-unavailable",
  "thread-unavailable",
  "run-not-found",
  "artifact-not-found",
  "artifact-unsupported",
  "artifact-too-large",
  "invalid-range",
  "operation-failed",
]);

export class RlAgentToolError extends Schema.TaggedErrorClass<RlAgentToolError>()(
  "RlAgentToolError",
  {
    code: RlAgentToolErrorCode,
    detail: Schema.String.check(Schema.isMaxLength(2048)),
  },
) {
  override get message(): string {
    return this.detail;
  }
}

export const RlAgentMetricSummary = Schema.Struct({
  key: MetricKey,
  observations: NonNegativeInt,
  finiteCount: NonNegativeInt,
  nullCount: NonNegativeInt,
  nanCount: NonNegativeInt,
  positiveInfinityCount: NonNegativeInt,
  negativeInfinityCount: NonNegativeInt,
  firstStep: Schema.NullOr(NonNegativeInt),
  lastStep: Schema.NullOr(NonNegativeInt),
  firstFiniteValue: Schema.NullOr(Schema.Number),
  lastFiniteValue: Schema.NullOr(Schema.Number),
  minimum: Schema.NullOr(Schema.Number),
  maximum: Schema.NullOr(Schema.Number),
  mean: Schema.NullOr(Schema.Number),
});

const ProjectContext = Schema.Struct({
  projectId: Schema.String,
  threadId: Schema.String,
});

export const RlRunInspection = Schema.Struct({
  ...ProjectContext.fields,
  summary: RlRunSummary,
  manifest: Schema.NullOr(RlResolvedManifest),
  artifacts: Schema.Array(RlArtifactMetadata),
  lineage: RlRunLineage,
  metricBatchCount: NonNegativeInt,
  availableMetricKeys: Schema.Array(MetricKey),
  metricSummaries: Schema.Array(RlAgentMetricSummary),
});

const dependencies = [
  McpInvocationContext.McpInvocationContext,
  RlManager.RlManager,
  RunStore.RunStore,
  ProjectionSnapshotQuery.ProjectionSnapshotQuery,
  ServerConfig.ServerConfig,
  FileSystem.FileSystem,
  Path.Path,
];

const readonlyTool = <T extends Tool.Any>(tool: T): T =>
  tool
    .annotate(Tool.Readonly, true)
    .annotate(Tool.Destructive, false)
    .annotate(Tool.Idempotent, true)
    .annotate(Tool.OpenWorld, false) as T;

const actionTool = <T extends Tool.Any>(tool: T): T =>
  tool
    .annotate(Tool.Readonly, false)
    .annotate(Tool.Destructive, true)
    .annotate(Tool.Idempotent, true)
    .annotate(Tool.OpenWorld, false) as T;

export const RlCapabilitiesTool = readonlyTool(
  Tool.make("rl_capabilities", {
    description:
      "Inspect the current environment's RL Lab runners and version-controlled experiment catalog. Use this before proposing or starting a run; unavailable runners include an actionable remedy.",
    parameters: Schema.Record(Schema.String, Schema.Never).annotate({
      description: "No input is required.",
    }),
    success: Schema.Struct({ ...ProjectContext.fields, report: RlCapabilityReport }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Inspect RL Lab capabilities"),
);

export const RlListRunsTool = readonlyTool(
  Tool.make("rl_list_runs", {
    description:
      "List retained RL Lab runs for the current thread's project. Project scope is derived from the credential and cannot be overridden by tool input.",
    parameters: Schema.Struct({
      limit: PositiveLimit(100, "Maximum retained runs to return. Defaults to 50."),
    }),
    success: Schema.Struct({ ...ProjectContext.fields, runs: Schema.Array(RlRunSummary) }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "List project RL runs"),
);

export const RlGetRunTool = readonlyTool(
  Tool.make("rl_get_run", {
    description:
      "Inspect one project-scoped RL run: lifecycle and failure reason, immutable manifest and effective algorithm configuration, artifact inventory, observed metric keys, and bounded numerical summaries. This is the evidence behind the Overview, Algorithm, and Diagnostics views.",
    parameters: Schema.Struct({ runId: RunIdParameter }),
    success: RlRunInspection,
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Inspect RL run"),
);

export const RlListArtifactsTool = readonlyTool(
  Tool.make("rl_list_artifacts", {
    description:
      "List one bounded page of artifacts for a project-scoped RL run. Use nextCursor to continue through checkpoint-heavy runs without relying on the smaller rl_get_run snapshot.",
    parameters: Schema.Struct({
      runId: RunIdParameter,
      cursor: Schema.optional(
        Schema.String.annotate({ description: "Opaque artifact cursor from the previous page." })
          .check(Schema.isTrimmed())
          .check(Schema.isNonEmpty())
          .check(Schema.isMaxLength(64)),
      ),
      limit: PositiveLimit(
        RL_MAX_ARTIFACT_PAGE_SIZE,
        `Maximum artifacts to return. Defaults to 50; maximum ${RL_MAX_ARTIFACT_PAGE_SIZE}.`,
      ),
    }),
    success: Schema.Struct({ ...ProjectContext.fields, runId: RlRunId, page: RlArtifactPage }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "List RL artifacts"),
);

export const RlQueryMetricsTool = readonlyTool(
  Tool.make("rl_query_metrics", {
    description:
      "Query bounded raw observations for selected metrics in one project-scoped RL run. This is the agent interface to RL Lab's Data explorer. Results keep the latest matching batches when the requested range exceeds the limit.",
    parameters: Schema.Struct({
      runId: RunIdParameter,
      metricKeys: Schema.Array(MetricKey)
        .check(Schema.isMinLength(1))
        .check(Schema.isMaxLength(16))
        .annotate({ description: "One to sixteen exact metric keys to return." }),
      stepFrom: Schema.optional(
        Schema.Int.annotate({ description: "Optional inclusive first training step." }).check(
          Schema.isGreaterThanOrEqualTo(0),
        ),
      ),
      stepTo: Schema.optional(
        Schema.Int.annotate({ description: "Optional inclusive final training step." }).check(
          Schema.isGreaterThanOrEqualTo(0),
        ),
      ),
      limit: PositiveLimit(
        RL_AGENT_METRIC_QUERY_MAX_BATCHES,
        "Maximum metric batches to return. Defaults to 200.",
      ),
    }),
    success: Schema.Struct({
      ...ProjectContext.fields,
      runId: RlRunId,
      availableMetricKeys: Schema.Array(MetricKey),
      totalMatchingBatches: NonNegativeInt,
      returnedBatches: NonNegativeInt,
      truncated: Schema.Boolean,
      batches: Schema.Array(RlMetricBatch),
    }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Query RL metrics"),
);

export const RlCompareRunsTool = readonlyTool(
  Tool.make("rl_compare_runs", {
    description:
      "Compare two to eight runs from the current project using immutable manifests and bounded summaries for selected metrics. Use it to detect seed, configuration, runner, source, or environment drift before drawing conclusions.",
    parameters: Schema.Struct({
      runIds: Schema.Array(RunIdParameter)
        .check(Schema.isMinLength(2))
        .check(Schema.isMaxLength(8))
        .annotate({ description: "Two to eight distinct run IDs from the current project." }),
      metricKeys: Schema.Array(MetricKey)
        .check(Schema.isMinLength(1))
        .check(Schema.isMaxLength(8))
        .annotate({ description: "One to eight metric keys to summarize for every run." }),
    }),
    success: Schema.Struct({
      ...ProjectContext.fields,
      metricKeys: Schema.Array(MetricKey),
      runs: Schema.Array(
        Schema.Struct({
          summary: RlRunSummary,
          manifest: Schema.NullOr(RlResolvedManifest),
          metricSummaries: Schema.Array(RlAgentMetricSummary),
        }),
      ),
    }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Compare RL runs"),
);

export const RlReadArtifactTool = readonlyTool(
  Tool.make("rl_read_artifact", {
    description:
      "Read a bounded textual artifact from a project-scoped RL run, including logs, summaries, evaluations, manifests, environment trajectories, and prompt/completion verifier replays used by the Behavior view. Binary models and unsupported media types are refused.",
    parameters: Schema.Struct({
      runId: RunIdParameter,
      artifactId: Schema.String.annotate({ description: "Artifact ID returned by rl_get_run." })
        .check(Schema.isTrimmed())
        .check(Schema.isNonEmpty())
        .check(Schema.isMaxLength(64)),
      maxBytes: PositiveLimit(
        RL_AGENT_ARTIFACT_MAX_BYTES,
        `Maximum artifact bytes to return. Defaults to 262144; maximum ${RL_AGENT_ARTIFACT_MAX_BYTES}.`,
      ),
    }),
    success: Schema.Struct({
      ...ProjectContext.fields,
      runId: RlRunId,
      artifact: RlArtifactMetadata,
      content: Schema.String.check(Schema.isMaxLength(RL_AGENT_ARTIFACT_MAX_BYTES)),
    }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Read RL artifact"),
);

export const RlStartRunTool = actionTool(
  Tool.make("rl_start_run", {
    description:
      "Start one RL Lab experiment in the current thread's project. This executes a server-side training process and is permission-aware. Call rl_capabilities first, use an explicit integer seed, and reuse requestId only when retrying the same intended run.",
    parameters: Schema.Struct({
      experimentId: IdentifierParameter("Experiment ID from rl_capabilities."),
      seed: Schema.Int.annotate({ description: "Explicit integer random seed for this run." }),
      requestId: IdentifierParameter(
        "Stable idempotency key for this intended run. Reuse it only to retry the same request.",
      ),
    }),
    success: Schema.Struct({ ...ProjectContext.fields, runId: RlRunId }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Start RL run"),
);

export const RlCancelRunTool = actionTool(
  Tool.make("rl_cancel_run", {
    description:
      "Cancel one non-terminal RL Lab run in the current project. Cancellation is permission-aware and idempotent; terminal runs remain unchanged.",
    parameters: Schema.Struct({ runId: RunIdParameter }),
    success: Schema.Struct({ ...ProjectContext.fields, runId: RlRunId, state: RlRunState }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Cancel RL run"),
);

export const RlCreateStudyTool = actionTool(
  Tool.make("rl_create_study", {
    description:
      "Create and schedule a project-scoped multi-seed A/B study with bounded concurrency and one immutable evaluation protocol.",
    parameters: Schema.Struct({
      definition: RlStudyDefinition.annotate({
        description:
          "Immutable variants, four-role seed sets, run budget, concurrency, and evaluation protocol.",
      }),
    }),
    success: Schema.Struct({ ...ProjectContext.fields, study: RlStudy }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Create RL study"),
);

export const RlValidateExperimentTool = readonlyTool(
  Tool.make("rl_validate_experiment", {
    description:
      "Validate a project-owned experiment and its model, dataset, verifier, capacity, and budget without starting a worker or installing dependencies.",
    parameters: Schema.Struct({
      experimentId: IdentifierParameter(
        "Project experiment ID, normally namespaced as project__<id>.",
      ),
    }),
    success: Schema.Struct({ ...ProjectContext.fields, report: RlExperimentValidationReport }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Validate project RL experiment"),
);

export const RlGetStudyTool = readonlyTool(
  Tool.make("rl_get_study", {
    description:
      "Inspect a project-scoped study, including its seed set and queued, running, failed, unmatched, or completed members.",
    parameters: Schema.Struct({
      studyId: IdentifierParameter("Study ID returned by rl_create_study."),
    }),
    success: Schema.Struct({ ...ProjectContext.fields, study: RlStudy }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Inspect RL study"),
);

export const RlCompareStudyTool = readonlyTool(
  Tool.make("rl_compare_study", {
    description:
      "Compute the server-owned, versioned paired comparison for two variants in a study. Reports N, seeds, dispersion, interval, unmatched and failed runs, or not-enough-evidence.",
    parameters: Schema.Struct({
      studyId: IdentifierParameter("Study ID returned by rl_create_study."),
      baselineLabel: IdentifierParameter("Baseline variant label."),
      candidateLabel: IdentifierParameter("Candidate variant label."),
      metricKey: MetricKey,
      estimator: RlStudyEstimator.annotate({
        description:
          "Versioned statistic, confidence level, bootstrap seed/count, statistical unit, and missing-pair policy.",
      }),
    }),
    success: Schema.Struct({ ...ProjectContext.fields, comparison: RlStudyComparison }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Compare RL study"),
);

const ContinueRunParameters = Schema.Struct({
  parentRunId: IdentifierParameter("Terminal parent run ID from rl_get_run."),
  sourceArtifactId: IdentifierParameter(
    "Verified checkpoint or adapter artifact ID returned by rl_get_run or rl_list_artifacts.",
  ),
  requestId: IdentifierParameter(
    "Stable idempotency key for this intended child run. Reuse it only to retry the same request.",
  ),
});

const WarmStartRunParameters = Schema.Struct({
  ...ContinueRunParameters.fields,
  targetExperimentId: Schema.optional(
    IdentifierParameter(
      "Optional target experiment. Use this to make an SFT adapter the explicit input to a DPO run.",
    ),
  ),
});

export const RlResumeRunTool = actionTool(
  Tool.make("rl_resume_run", {
    description:
      "Create a new child run that exactly resumes a terminal parent from a verified, complete trainer checkpoint. The server refuses missing state or compatibility drift; it never mutates the parent.",
    parameters: ContinueRunParameters,
    success: Schema.Struct({
      ...ProjectContext.fields,
      parentRunId: RlRunId,
      runId: RlRunId,
      relation: Schema.Literals(["resume"]),
    }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Resume RL run from checkpoint"),
);

export const RlWarmStartRunTool = actionTool(
  Tool.make("rl_warm_start_run", {
    description:
      "Create a new child run initialized from a verified deployable PEFT adapter. This is an explicit warm start, not an exact trainer resume, and the parent remains immutable.",
    parameters: WarmStartRunParameters,
    success: Schema.Struct({
      ...ProjectContext.fields,
      parentRunId: RlRunId,
      runId: RlRunId,
      relation: Schema.Literals(["warm-start"]),
    }),
    failure: RlAgentToolError,
    dependencies,
  }).annotate(Tool.Title, "Warm-start RL run from adapter"),
);

export const RlToolkit = Toolkit.make(
  RlCapabilitiesTool,
  RlListRunsTool,
  RlGetRunTool,
  RlListArtifactsTool,
  RlQueryMetricsTool,
  RlCompareRunsTool,
  RlReadArtifactTool,
  RlStartRunTool,
  RlResumeRunTool,
  RlWarmStartRunTool,
  RlCancelRunTool,
  RlCreateStudyTool,
  RlGetStudyTool,
  RlCompareStudyTool,
  RlValidateExperimentTool,
);
