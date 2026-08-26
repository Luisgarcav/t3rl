import { describe, expect, it } from "vite-plus/test";

import {
  formatRlDuration,
  formatRlMetricValue,
  latestRlMetricValue,
  rlStatusVariant,
  selectRlMetricPoints,
} from "./rlPresentation";

describe("RL presentation", () => {
  it("maps lifecycle states to semantic status variants", () => {
    expect(rlStatusVariant("running")).toBe("success");
    expect(rlStatusVariant("failed")).toBe("error");
    expect(rlStatusVariant("cancelling")).toBe("warning");
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
