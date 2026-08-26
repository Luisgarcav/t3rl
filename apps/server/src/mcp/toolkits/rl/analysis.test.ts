import { describe, expect, it } from "vite-plus/test";

import type { RlMetricBatch } from "@t3tools/contracts";

import { queryMetrics, summarizeMetrics } from "./analysis.ts";

const metrics: ReadonlyArray<RlMetricBatch> = [
  {
    step: 1,
    wallClockMs: 10,
    values: { "train/return": 2, "train/value_loss": null },
  },
  {
    step: 2,
    wallClockMs: 20,
    values: { "train/return": 4, "train/value_loss": "nan" },
  },
  {
    step: 3,
    wallClockMs: 30,
    values: { "train/return": 8, "train/value_loss": "+inf" },
  },
];

describe("RL agent metric analysis", () => {
  it("summarizes finite, missing, and explicit non-finite evidence", () => {
    expect(summarizeMetrics(metrics)).toEqual([
      {
        key: "train/return",
        observations: 3,
        finiteCount: 3,
        nullCount: 0,
        nanCount: 0,
        positiveInfinityCount: 0,
        negativeInfinityCount: 0,
        firstStep: 1,
        lastStep: 3,
        firstFiniteValue: 2,
        lastFiniteValue: 8,
        minimum: 2,
        maximum: 8,
        mean: 14 / 3,
      },
      {
        key: "train/value_loss",
        observations: 3,
        finiteCount: 0,
        nullCount: 1,
        nanCount: 1,
        positiveInfinityCount: 1,
        negativeInfinityCount: 0,
        firstStep: 1,
        lastStep: 3,
        firstFiniteValue: null,
        lastFiniteValue: null,
        minimum: null,
        maximum: null,
        mean: null,
      },
    ]);
  });

  it("keeps metric queries bounded and returns only selected values", () => {
    expect(
      queryMetrics({
        metrics,
        metricKeys: ["train/return"],
        stepFrom: 1,
        stepTo: 3,
        limit: 2,
      }),
    ).toEqual({
      availableMetricKeys: ["train/return", "train/value_loss"],
      totalMatchingBatches: 3,
      returnedBatches: 2,
      truncated: true,
      batches: [
        { step: 2, wallClockMs: 20, values: { "train/return": 4 } },
        { step: 3, wallClockMs: 30, values: { "train/return": 8 } },
      ],
    });
  });
});
