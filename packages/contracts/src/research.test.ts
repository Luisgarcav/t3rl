import * as Schema from "effect/Schema";
import { describe, expect, it } from "vite-plus/test";

import { RESEARCH_MAX_RUN_BUDGET, ResearchWorkspaceDocument } from "./research.ts";

const decode = Schema.decodeUnknownSync(ResearchWorkspaceDocument);

const validDocument = {
  version: 2,
  profiles: [
    {
      id: "research-lead",
      name: "Research lead",
      summary: "Forms falsifiable hypotheses.",
      instructions: "Cite run evidence and distinguish observations from hypotheses.",
    },
  ],
  studyDraft: {
    target: {
      algorithmId: "grpo",
      taskType: "llm-reasoning",
      rewardSource: "verifiable",
    },
    objective: "Improve evaluation return without increasing instability.",
    successMetric: "eval/mean_return",
    direction: "increase",
    baselineRunId: "run-baseline",
    maxRuns: 3,
    maxWallClockMinutes: 45,
    seeds: [7, 11, 19],
    specialistProfileIds: ["research-lead"],
    approvalPolicy: "review-every-trial",
  },
} as const;

describe("ResearchWorkspaceDocument", () => {
  it("decodes the version-controlled research setup", () => {
    expect(decode(validDocument)).toMatchObject(validDocument);
  });

  it("rejects invalid profile identifiers", () => {
    expect(() =>
      decode({
        ...validDocument,
        profiles: [{ ...validDocument.profiles[0], id: "../../escape" }],
      }),
    ).toThrow();
  });

  it("bounds the run budget", () => {
    expect(() =>
      decode({
        ...validDocument,
        studyDraft: { ...validDocument.studyDraft, maxRuns: RESEARCH_MAX_RUN_BUDGET + 1 },
      }),
    ).toThrow();
  });

  it("only accepts the review-gated approval policy", () => {
    expect(() =>
      decode({
        ...validDocument,
        studyDraft: { ...validDocument.studyDraft, approvalPolicy: "unattended" },
      }),
    ).toThrow();
  });

  it("keeps reward regimes separate from algorithm identifiers", () => {
    expect(() =>
      decode({
        ...validDocument,
        studyDraft: {
          ...validDocument.studyDraft,
          target: { ...validDocument.studyDraft.target, rewardSource: "grpo" },
        },
      }),
    ).toThrow();
  });

  it("rejects dangling specialist references", () => {
    expect(() =>
      decode({
        ...validDocument,
        studyDraft: {
          ...validDocument.studyDraft,
          specialistProfileIds: ["missing-profile"],
        },
      }),
    ).toThrow();
  });
});
