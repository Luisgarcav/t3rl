import { ResearchAlgorithmId } from "@t3tools/contracts";
import { describe, expect, it } from "vite-plus/test";

import {
  RESEARCH_ALGORITHM_CATALOG,
  RESEARCH_ALGORITHM_FAMILIES,
  researchAlgorithmById,
  researchExecutionStatus,
} from "./researchAlgorithms.ts";

describe("research algorithm catalog", () => {
  it("keeps identifiers unique and covers every declared family", () => {
    expect(new Set(RESEARCH_ALGORITHM_CATALOG.map((entry) => entry.id)).size).toBe(
      RESEARCH_ALGORITHM_CATALOG.length,
    );
    for (const family of RESEARCH_ALGORITHM_FAMILIES) {
      expect(RESEARCH_ALGORITHM_CATALOG.some((entry) => entry.family === family.id)).toBe(true);
    }
  });

  it("covers classical, offline, model-based, multi-agent, and LLM methods", () => {
    const ids = new Set<string>(RESEARCH_ALGORITHM_CATALOG.map((entry) => entry.id));
    for (const id of ["dqn", "ppo", "sac", "dreamer-v3", "cql", "mappo", "grpo", "rloo"]) {
      expect(ids.has(id)).toBe(true);
    }
  });

  it("advertises native execution only for a real built-in combination", () => {
    const ppo = researchAlgorithmById(ResearchAlgorithmId.make("ppo"));
    expect(researchExecutionStatus(ppo, "discrete-control").native).toBe(true);
    expect(researchExecutionStatus(ppo, "continuous-control").native).toBe(true);
    const dqn = researchAlgorithmById(ResearchAlgorithmId.make("dqn"));
    expect(researchExecutionStatus(dqn, "continuous-control").native).toBe(false);
    const grpo = researchAlgorithmById(ResearchAlgorithmId.make("grpo"));
    expect(researchExecutionStatus(grpo, "llm-reasoning").native).toBe(false);
  });
});
