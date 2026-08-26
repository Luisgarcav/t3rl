import type { RlMetricBatch } from "@t3tools/contracts";
import { describe, expect, it } from "vite-plus/test";

import {
  collectRlMetricKeys,
  createRlDataVisualizationSpec,
  selectRlMetricData,
} from "./rlDataExplorer";

const metrics: ReadonlyArray<RlMetricBatch> = [
  { step: 0, wallClockMs: 10, values: { "train/return": 1, "train/loss": null } },
  { step: 1, wallClockMs: 20, values: { "eval/return": 2, "train/return": 3 } },
  { step: 2, wallClockMs: 30, values: { "train/return": "nan" } },
];

describe("RL data explorer", () => {
  it("discovers arbitrary metric keys in stable order", () => {
    expect(collectRlMetricKeys(metrics)).toEqual(["eval/return", "train/loss", "train/return"]);
  });

  it("preserves null and explicit non-finite evidence while bounding rows", () => {
    expect(selectRlMetricData(metrics, "train/return", 2)).toEqual({
      rows: [
        { step: 1, wallClockMs: 20, value: 3 },
        { step: 2, wallClockMs: 30, value: "nan" },
      ],
      totalRows: 3,
      truncated: true,
    });
    expect(selectRlMetricData(metrics, "train/loss").rows[0]?.value).toBeNull();
  });

  it("creates a declarative source spec from exactly the visible rows", () => {
    const selection = selectRlMetricData(metrics, "eval/return");
    expect(createRlDataVisualizationSpec("Evaluation return", selection)).toEqual({
      version: 1,
      kind: "line",
      title: "Evaluation return",
      encoding: {
        x: { field: "step", type: "quantitative" },
        y: { field: "value", type: "quantitative" },
      },
      data: [{ step: 1, wallClockMs: 20, value: 2 }],
    });
  });
});
