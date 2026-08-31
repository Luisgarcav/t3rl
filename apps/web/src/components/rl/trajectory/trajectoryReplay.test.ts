import { describe, expect, it } from "vite-plus/test";

import {
  clampTrajectoryIndex,
  parseTrajectoryReplay,
  parseTrajectoryReplayJson,
  readBoundedResponseText,
  TrajectoryReplayError,
  trajectoryTerminalKind,
} from "./trajectoryReplay";

const validReplay = {
  environment: "CartPole-v1",
  evaluationSeed: 100_007,
  trajectory: [
    {
      step: 0,
      observation: [0.01, -0.02, 0.03, -0.04],
      action: 1,
      reward: 1,
      terminated: false,
      truncated: false,
    },
    {
      step: 1,
      observation: [0.02, 0.17, 0.01, -0.33],
      action: 0,
      reward: 1,
      terminated: true,
      truncated: false,
    },
  ],
};

describe("trajectory replay validation", () => {
  it("accepts and preserves the bounded SB3 replay contract", () => {
    const replay = parseTrajectoryReplay(validReplay);

    expect(replay.kind).toBe("environment");
    expect(replay.environment).toBe("CartPole-v1");
    expect(replay.evaluationSeed).toBe(100_007);
    expect(replay.trajectory).toHaveLength(2);
    expect(replay.trajectory[1]?.observation).toEqual([0.02, 0.17, 0.01, -0.33]);
    expect(trajectoryTerminalKind(replay.trajectory[1]!)).toBe("terminated");
  });

  it("accepts prompt/completion evidence without changing legacy replays", () => {
    const replay = parseTrajectoryReplay({
      kind: "llm-post-training",
      environment: "llm:Qwen/Qwen2.5-0.5B-Instruct",
      evaluationSeed: 7,
      trajectory: [
        {
          step: 0,
          observation: { prompt: "What is 7 + 5?", expected: "12" },
          action: {
            completion: "12",
            parsedAnswer: "12",
            verifier: { id: "exact-integer-v1", passed: true },
          },
          reward: 1,
          terminated: true,
          truncated: false,
        },
      ],
    });

    expect(replay.kind).toBe("llm-post-training");
    expect(replay.trajectory[0]?.reward).toBe(1);
  });

  it("rejects malformed, non-finite, out-of-order, and oversized trajectories", () => {
    expect(() => parseTrajectoryReplay({ ...validReplay, trajectory: [] })).toThrow(
      "must contain at least one step",
    );
    expect(() =>
      parseTrajectoryReplay({
        ...validReplay,
        trajectory: [{ ...validReplay.trajectory[0], reward: Number.POSITIVE_INFINITY }],
      }),
    ).toThrow("reward must be a finite number");
    expect(() =>
      parseTrajectoryReplay({
        ...validReplay,
        trajectory: [validReplay.trajectory[1], validReplay.trajectory[0]],
      }),
    ).toThrow("must be greater than the previous step");
    expect(() => parseTrajectoryReplay(validReplay, { maxSteps: 1 })).toThrow(
      "exceeds the limit of 1 steps",
    );
  });

  it("reports JSON syntax separately from schema failures", () => {
    expect(() => parseTrajectoryReplayJson("{not-json}")).toThrowError(TrajectoryReplayError);
    try {
      parseTrajectoryReplayJson("{not-json}");
    } catch (error) {
      expect(error).toMatchObject({ code: "invalid-json" });
    }
  });
});

describe("bounded replay transport", () => {
  it("reads a response below the byte cap", async () => {
    const json = JSON.stringify(validReplay);
    await expect(readBoundedResponseText(new Response(json), json.length + 1)).resolves.toBe(json);
  });

  it("stops reading when the response exceeds the byte cap", async () => {
    await expect(readBoundedResponseText(new Response("123456789"), 4)).rejects.toMatchObject({
      code: "too-large",
    });
  });
});

describe("trajectory navigation", () => {
  it("clamps requested frames to valid trajectory bounds", () => {
    expect(clampTrajectoryIndex(-5, 4)).toBe(0);
    expect(clampTrajectoryIndex(2.8, 4)).toBe(2);
    expect(clampTrajectoryIndex(99, 4)).toBe(3);
    expect(clampTrajectoryIndex(2, 0)).toBe(0);
  });
});
