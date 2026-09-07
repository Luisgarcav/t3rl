import * as Schema from "effect/Schema";

import { TrimmedNonEmptyString } from "./baseSchemas.ts";
import { RlArtifactId, RlRunId, RlSha256, RlStudyEstimator, RlStudyId } from "./rl.ts";

export const RL_EVIDENCE_MAX_RUNS = 12;
export const RL_EVIDENCE_MAX_BYTES = 512 * 1024 * 1024;
export const RL_EVIDENCE_MAX_FILES = 4096;
const Identifier = TrimmedNonEmptyString.check(Schema.isMaxLength(64)).check(
  Schema.isPattern(/^[A-Za-z0-9_-]+$/),
);
const Text = TrimmedNonEmptyString.check(Schema.isMaxLength(4000));
const ProjectId = TrimmedNonEmptyString.check(Schema.isMaxLength(128));

export const RlEvidenceReference = Schema.Struct({
  runId: RlRunId,
  artifactId: Schema.NullOr(RlArtifactId),
});
export type RlEvidenceReference = typeof RlEvidenceReference.Type;

/** A research note records its author's assertions; it never authorizes execution. */
export const RlResearchRecordInput = Schema.Struct({
  projectId: ProjectId,
  requestId: Identifier,
  parentRecordId: Schema.NullOr(Identifier),
  hypothesis: Text,
  evidence: Schema.Array(RlEvidenceReference)
    .check(Schema.isMinLength(1))
    .check(Schema.isMaxLength(32)),
  proposedChange: Text,
  authorizationReference: Schema.NullOr(Text),
  outcome: Schema.Literals(["proposed", "supported", "rejected", "inconclusive"]),
  interpretation: Text,
  limitations: Text,
});
export type RlResearchRecordInput = typeof RlResearchRecordInput.Type;

export const RlResearchRecord = Schema.Struct({
  version: Schema.Literal(1),
  recordId: Identifier,
  recordedAt: Schema.String,
  input: RlResearchRecordInput,
  evidence: Schema.Array(
    Schema.Struct({
      ...RlEvidenceReference.fields,
      sha256: RlSha256,
      sourceRevision: Schema.NullOr(Schema.String),
      sourceDirty: Schema.NullOr(Schema.Boolean),
    }),
  ).check(Schema.isMaxLength(32)),
  sha256: RlSha256,
});
export type RlResearchRecord = typeof RlResearchRecord.Type;

export const RlGetResearchRecordInput = Schema.Struct({
  projectId: ProjectId,
  recordId: Identifier,
});

export const RlExportEvidenceInput = Schema.Struct({
  projectId: ProjectId,
  runIds: Schema.Array(RlRunId)
    .check(Schema.isMinLength(1))
    .check(Schema.isMaxLength(RL_EVIDENCE_MAX_RUNS)),
  /** Null selects a run bundle. A study bundle also includes its protocol and comparison. */
  study: Schema.NullOr(
    Schema.Struct({
      studyId: RlStudyId,
      baselineLabel: Identifier,
      candidateLabel: Identifier,
      metricKey: TrimmedNonEmptyString.check(Schema.isMaxLength(64)),
      estimator: RlStudyEstimator,
    }),
  ),
  recordIds: Schema.Array(Identifier).check(Schema.isMaxLength(16)),
  /** Large checkpoints and arbitrary logs are opt-in; every omitted artifact is listed. */
  additionalArtifacts: Schema.Array(RlEvidenceReference).check(Schema.isMaxLength(128)),
});
export type RlExportEvidenceInput = typeof RlExportEvidenceInput.Type;

export const RlEvidenceExport = Schema.Struct({
  version: Schema.Literal(1),
  exportId: Identifier,
  projectId: ProjectId,
  sha256: RlSha256,
  indexSha256: RlSha256,
  bytes: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  fileCount: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  omittedCount: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
});
export type RlEvidenceExport = typeof RlEvidenceExport.Type;

export class RlEvidenceError extends Schema.TaggedErrorClass<RlEvidenceError>()("RlEvidenceError", {
  detail: Schema.String.check(Schema.isMaxLength(2048)),
}) {
  override get message(): string {
    return this.detail;
  }
}
