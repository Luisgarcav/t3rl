import type { RlRunProjection } from "@t3tools/client-runtime/state/rl";
import { describe, expect, it } from "vite-plus/test";

import { formatResearchRunEvidence } from "./researchEvidence";

const projection: RlRunProjection = {
  summary: {
    runId: "run-1",
    projectId: "project-1",
    experimentId: "cartpole-ppo",
    state: "completed",
    requestedAt: "2026-08-25T00:00:00.000Z",
    startedAt: "2026-08-25T00:00:01.000Z",
    endedAt: "2026-08-25T00:01:00.000Z",
    lastMessageAt: "2026-08-25T00:01:00.000Z",
    errorCode: null,
    errorMessage: null,
  },
  manifest: null,
  artifacts: [],
  metrics: [
    { step: 10, wallClockMs: 10, values: { "eval/mean_return": 20, entropy: "nan" } },
    { step: 0, wallClockMs: 0, values: { "eval/mean_return": 5 } },
  ],
};

describe("formatResearchRunEvidence", () => {
  it("summarizes metrics in step order and preserves non-finite markers", () => {
    const evidence = formatResearchRunEvidence(projection);
    expect(evidence).toContain("first=5 @0, last=20 @10");
    expect(evidence).toContain("entropy: numeric points=0, non-finite=1");
    expect(evidence).toContain("manifest:\n- unavailable");
  });
});
