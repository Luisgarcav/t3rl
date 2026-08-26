import { ResearchAlgorithmId } from "@t3tools/contracts";

import {
  buildAutoresearchIterationPrompt,
  buildResearchSpecialistPrompt,
  createDefaultResearchWorkspaceDocument,
  parseResearchWorkspaceDocument,
  serializeResearchWorkspaceDocument,
} from "./research.ts";
import { describe, expect, it } from "vite-plus/test";

describe("research workspace", () => {
  it("round-trips the version-controlled document", () => {
    const document = createDefaultResearchWorkspaceDocument();
    const parsed = parseResearchWorkspaceDocument(serializeResearchWorkspaceDocument(document));
    expect(parsed.error).toBeNull();
    expect(parsed.document).toEqual(document);
  });

  it("falls back safely when the project file is malformed", () => {
    const parsed = parseResearchWorkspaceDocument("{not json");
    expect(parsed.error).not.toBeNull();
    expect(parsed.document.profiles.length).toBeGreaterThan(0);
  });

  it("migrates version-one workspaces and retains user profiles", () => {
    const current = createDefaultResearchWorkspaceDocument();
    const { target: _target, ...legacyStudyDraft } = current.studyDraft;
    const parsed = parseResearchWorkspaceDocument(
      JSON.stringify({
        ...current,
        version: 1,
        profiles: current.profiles.slice(0, 5),
        studyDraft: legacyStudyDraft,
      }),
    );
    expect(parsed.error).toBeNull();
    expect(parsed.document.version).toBe(2);
    expect(parsed.document.studyDraft.target.algorithmId).toBe("ppo");
    expect(parsed.document.profiles.some((entry) => entry.id === "reward-verifier-auditor")).toBe(
      true,
    );
  });

  it("builds portable specialist instructions without claiming a system role", () => {
    const document = createDefaultResearchWorkspaceDocument();
    const prompt = buildResearchSpecialistPrompt(document.profiles[0]!);
    expect(prompt).toContain("<research-specialist-instructions>");
    expect(prompt).toContain("do not grant permission");
    expect(prompt).toContain("Task:");
  });

  it("builds one bounded iteration and stops before side effects", () => {
    const initial = createDefaultResearchWorkspaceDocument();
    const document = {
      ...initial,
      studyDraft: {
        ...initial.studyDraft,
        target: {
          ...initial.studyDraft.target,
          algorithmId: ResearchAlgorithmId.make("grpo"),
          taskType: "llm-reasoning" as const,
          rewardSource: "verifiable" as const,
        },
        objective: "Improve robust evaluation return.",
      },
    };
    const prompt = buildAutoresearchIterationPrompt({
      study: document.studyDraft,
      profiles: document.profiles,
      baselineEvidence: {
        runId: "run-123",
        summary: "state: completed\nmetric eval/mean_return: first=10 last=18",
      },
    });
    expect(prompt).toContain("at most 3 runs");
    expect(prompt).toContain("run-123");
    expect(prompt).toContain("Algorithm: GRPO");
    expect(prompt).toContain("Reward source: verifiable");
    expect(prompt).toContain("For RLVR, inspect verifier identity");
    expect(prompt).toContain("Stop before editing files or starting training");
    expect(prompt).toContain("Never expand the budget");
  });
});
