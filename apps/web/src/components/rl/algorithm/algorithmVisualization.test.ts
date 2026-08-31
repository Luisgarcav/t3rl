import type { RlResolvedManifest } from "@t3tools/contracts";
import { describe, expect, it } from "vite-plus/test";

import {
  clampRlAlgorithmStageIndex,
  resolveRlAlgorithmVisualization,
} from "./algorithmVisualization";

const manifest = (algorithm: string): RlResolvedManifest => ({
  experimentId: "custom-experiment",
  runnerId: "stable-baselines3",
  runnerVersion: "2.7.0",
  protocolVersion: 1,
  seed: 7,
  effectiveConfig: { algorithm },
  sourceRevision: null,
  sourceDirty: null,
  pythonExecutable: "/usr/bin/python",
  pythonVersion: "3.13",
  environmentFingerprint: "test",
  instrumentationLevel: "standard",
  hardwareSummary: "CPU",
});

describe("RL algorithm visualization", () => {
  it("prefers the immutable resolved manifest over an experiment-name guess", () => {
    const spec = resolveRlAlgorithmVisualization(manifest("PPO"), "cartpole-dqn");

    expect(spec.algorithm).toBe("PPO");
    expect(spec.source).toBe("resolved-manifest");
    expect(spec.stages.map((stage) => stage.id)).toContain("advantage");
  });

  it("models replay and a target network for DQN", () => {
    const spec = resolveRlAlgorithmVisualization(null, "cartpole-dqn");

    expect(spec.algorithm).toBe("DQN");
    expect(spec.source).toBe("experiment-id");
    expect(spec.stages.map((stage) => stage.id)).toEqual([
      "act",
      "replay",
      "target",
      "update",
      "sync",
      "evaluate",
    ]);
    expect(spec.edges.at(-1)).toEqual({ from: "evaluate", to: "act", kind: "loop" });
  });

  it("shows the completion, verifier, advantage, and update loop for GRPO", () => {
    const spec = resolveRlAlgorithmVisualization(manifest("GRPO"), "arithmetic-grpo-rlvr");

    expect(spec.algorithm).toBe("GRPO");
    expect(spec.family).toBe("Online RL for language models");
    expect(spec.stages.map((stage) => stage.id)).toEqual([
      "prompts",
      "sample",
      "verify",
      "advantage",
      "update",
      "evaluate",
      "inspect",
    ]);
    expect(spec.stages.find((stage) => stage.id === "verify")?.evidenceMetricKeys).toContain(
      "train/verifier_pass_rate",
    );
    expect(spec.stages.find((stage) => stage.id === "evaluate")?.evidenceMetricKeys).toContain(
      "eval/verifier_pass_rate",
    );
  });

  it("uses an honest generic flow for an unknown algorithm", () => {
    const spec = resolveRlAlgorithmVisualization(manifest("MyOptimizer"), "custom");

    expect(spec.algorithm).toBe("MyOptimizer");
    expect(spec.family).toBe("Project-defined");
    expect(spec.stages[0]?.label).toBe("Gather experience");
  });

  it("clamps navigation to the declared stage range", () => {
    expect(clampRlAlgorithmStageIndex(-3, 5)).toBe(0);
    expect(clampRlAlgorithmStageIndex(2.9, 5)).toBe(2);
    expect(clampRlAlgorithmStageIndex(99, 5)).toBe(4);
    expect(clampRlAlgorithmStageIndex(1, 0)).toBe(0);
  });
});
