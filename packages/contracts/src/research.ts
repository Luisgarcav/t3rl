import * as Schema from "effect/Schema";

import { PositiveInt, TrimmedNonEmptyString, TrimmedString } from "./baseSchemas.ts";
import { RlRunId } from "./rl.ts";

/** Version-controlled project file used by the first research-workbench slice. */
export const RESEARCH_WORKSPACE_FILE = ".t3rl/research.json";
export const RESEARCH_WORKSPACE_VERSION = 2;
export const RESEARCH_MAX_PROFILES = 16;
export const RESEARCH_MAX_SELECTED_PROFILES = 8;
export const RESEARCH_MAX_RUN_BUDGET = 12;
export const RESEARCH_MAX_WALL_CLOCK_MINUTES = 24 * 60;

const ResearchProfileIdSchema = TrimmedNonEmptyString.check(
  Schema.isPattern(/^[A-Za-z0-9_-]+$/),
).check(Schema.isMaxLength(64));

export const ResearchAgentProfileId = ResearchProfileIdSchema.pipe(
  Schema.brand("ResearchAgentProfileId"),
);
export type ResearchAgentProfileId = typeof ResearchAgentProfileId.Type;

export const ResearchAgentProfile = Schema.Struct({
  id: ResearchAgentProfileId,
  name: TrimmedNonEmptyString.check(Schema.isMaxLength(80)),
  summary: TrimmedString.check(Schema.isMaxLength(240)),
  instructions: TrimmedNonEmptyString.check(Schema.isMaxLength(12_000)),
});
export type ResearchAgentProfile = typeof ResearchAgentProfile.Type;

export const ResearchAlgorithmId = ResearchProfileIdSchema.pipe(
  Schema.brand("ResearchAlgorithmId"),
);
export type ResearchAlgorithmId = typeof ResearchAlgorithmId.Type;

/** The task topology is independent from the optimization algorithm. */
export const ResearchTaskType = Schema.Literals([
  "bandit",
  "tabular",
  "discrete-control",
  "continuous-control",
  "robotics",
  "games",
  "multi-agent",
  "offline-dataset",
  "imitation",
  "model-based",
  "llm-alignment",
  "llm-reasoning",
  "custom",
]);
export type ResearchTaskType = typeof ResearchTaskType.Type;

/** RLVR/RLHF/RLAIF are reward regimes, not policy-optimization algorithms. */
export const ResearchRewardSource = Schema.Literals([
  "environment",
  "demonstrations",
  "human-feedback",
  "ai-feedback",
  "learned-reward-model",
  "verifiable",
  "rule-based",
  "hybrid",
  "custom",
]);
export type ResearchRewardSource = typeof ResearchRewardSource.Type;

export const ResearchStudyTarget = Schema.Struct({
  algorithmId: ResearchAlgorithmId,
  taskType: ResearchTaskType,
  rewardSource: ResearchRewardSource,
});
export type ResearchStudyTarget = typeof ResearchStudyTarget.Type;

const ResearchRunBudget = PositiveInt.check(Schema.isLessThanOrEqualTo(RESEARCH_MAX_RUN_BUDGET));
const ResearchWallClockBudget = PositiveInt.check(
  Schema.isLessThanOrEqualTo(RESEARCH_MAX_WALL_CLOCK_MINUTES),
);

export const ResearchStudyDraft = Schema.Struct({
  target: ResearchStudyTarget,
  objective: TrimmedString.check(Schema.isMaxLength(2_000)),
  successMetric: TrimmedString.check(Schema.isMaxLength(160)),
  direction: Schema.Literals(["increase", "decrease"]),
  baselineRunId: Schema.NullOr(RlRunId),
  maxRuns: ResearchRunBudget,
  maxWallClockMinutes: ResearchWallClockBudget,
  seeds: Schema.Array(Schema.Int).check(Schema.isMaxLength(RESEARCH_MAX_RUN_BUDGET)),
  specialistProfileIds: Schema.Array(ResearchAgentProfileId).check(
    Schema.isMaxLength(RESEARCH_MAX_SELECTED_PROFILES),
  ),
  /** The first slice deliberately has no unattended execution mode. */
  approvalPolicy: Schema.Literal("review-every-trial"),
});
export type ResearchStudyDraft = typeof ResearchStudyDraft.Type;

export const ResearchWorkspaceDocument = Schema.Struct({
  version: Schema.Literal(RESEARCH_WORKSPACE_VERSION),
  profiles: Schema.Array(ResearchAgentProfile).check(Schema.isMaxLength(RESEARCH_MAX_PROFILES)),
  studyDraft: ResearchStudyDraft,
}).check(
  Schema.makeFilter((document) => {
    const profileIds = new Set(document.profiles.map((profile) => profile.id));
    if (profileIds.size !== document.profiles.length) return "Research profile IDs must be unique.";
    if (document.studyDraft.seeds.length > document.studyDraft.maxRuns) {
      return "The seed list cannot exceed the run budget.";
    }
    if (new Set(document.studyDraft.seeds).size !== document.studyDraft.seeds.length) {
      return "Research seeds must be unique.";
    }
    return (
      document.studyDraft.specialistProfileIds.every((profileId) => profileIds.has(profileId)) ||
      "Every selected specialist must reference an existing profile."
    );
  }),
);
export type ResearchWorkspaceDocument = typeof ResearchWorkspaceDocument.Type;
