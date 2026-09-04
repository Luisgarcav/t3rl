import { describe, expect, it } from "vite-plus/test";

import {
  formatRlDuration,
  formatRlMetricValue,
  isLlmPostTrainingRunner,
  latestRlMetricValue,
  LLM_POST_TRAINING_METRIC_DEFINITIONS,
  rlMetricDefinitionsForRunner,
  rlStatusVariant,
  selectRlMetricPoints,
} from "./rlPresentation";

describe("RL presentation", () => {
  it("maps lifecycle states to semantic status variants", () => {
    expect(rlStatusVariant("running")).toBe("success");
    expect(rlStatusVariant("failed")).toBe("error");
    expect(rlStatusVariant("cancelling")).toBe("warning");
  });

  it("selects LLM post-training presentation for TRL and Axolotl runs", () => {
    expect(isLlmPostTrainingRunner("trl")).toBe(true);
    expect(isLlmPostTrainingRunner("axolotl")).toBe(true);
    expect(isLlmPostTrainingRunner("stable-baselines3")).toBe(false);
    expect(rlMetricDefinitionsForRunner("trl")).toBe(LLM_POST_TRAINING_METRIC_DEFINITIONS);
    expect(rlMetricDefinitionsForRunner("axolotl")).toBe(LLM_POST_TRAINING_METRIC_DEFINITIONS);
    expect(LLM_POST_TRAINING_METRIC_DEFINITIONS.map((definition) => definition.key)).toContain(
      "eval/verifier_pass_rate",
    );
    expect(LLM_POST_TRAINING_METRIC_DEFINITIONS.map((definition) => definition.key)).toContain(
      "system/tokens_per_second",
    );
    expect(rlMetricDefinitionsForRunner("stable-baselines3")[0]?.key).toBe("train/return");
  });

  it("keeps only finite chart points while preserving explicit latest markers", () => {
    const metrics = [
      { step: 1, wallClockMs: 1, values: { "train/loss": 2 } },
      { step: 2, wallClockMs: 2, values: { "train/loss": null } },
      { step: 3, wallClockMs: 3, values: { "train/loss": "nan" as const } },
    ];

    expect(selectRlMetricPoints(metrics, "train/loss")).toEqual([{ step: 1, value: 2 }]);
    expect(latestRlMetricValue(metrics, "train/loss")).toBe("nan");
    expect(formatRlMetricValue("nan")).toBe("NAN");
    expect(formatRlMetricValue(0.000005)).toBe("5e-6");
    expect(formatRlMetricValue(0.0000125)).toBe("1.25e-5");
  });

  it("downsamples long series and always keeps the final point", () => {
    const metrics = Array.from({ length: 301 }, (_, step) => ({
      step,
      wallClockMs: step,
      values: { score: step },
    }));
    const points = selectRlMetricPoints(metrics, "score", 20);

    expect(points.length).toBeLessThanOrEqual(20);
    expect(points.at(-1)).toEqual({ step: 300, value: 300 });
  });

  it("formats active and completed durations", () => {
    expect(
      formatRlDuration("2026-08-25T12:00:00.000Z", null, Date.parse("2026-08-25T12:01:05Z")),
    ).toBe("1m 05s");
    expect(
      formatRlDuration(
        "2026-08-25T12:00:00.000Z",
        "2026-08-25T14:03:00.000Z",
        Date.parse("2026-08-25T15:00:00Z"),
      ),
    ).toBe("2h 03m");
  });
});
