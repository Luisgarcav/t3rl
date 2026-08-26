import type { RlMetricBatch } from "@t3tools/contracts";

export const RL_DATA_EXPLORER_ROW_LIMIT = 200;

export interface RlMetricDataRow {
  readonly step: number;
  readonly wallClockMs: number;
  readonly value: RlMetricBatch["values"][string];
}

export interface RlMetricDataSelection {
  readonly rows: ReadonlyArray<RlMetricDataRow>;
  readonly totalRows: number;
  readonly truncated: boolean;
}

export interface RlDataVisualizationSpec {
  readonly version: 1;
  readonly kind: "line";
  readonly title: string;
  readonly encoding: {
    readonly x: { readonly field: "step"; readonly type: "quantitative" };
    readonly y: { readonly field: "value"; readonly type: "quantitative" };
  };
  readonly data: ReadonlyArray<RlMetricDataRow>;
}

export function collectRlMetricKeys(metrics: ReadonlyArray<RlMetricBatch>): ReadonlyArray<string> {
  const keys = new Set<string>();
  for (const batch of metrics) {
    for (const key of Object.keys(batch.values)) keys.add(key);
  }
  return Array.from(keys).toSorted((left, right) => left.localeCompare(right));
}

export function selectRlMetricData(
  metrics: ReadonlyArray<RlMetricBatch>,
  metricKey: string,
  rowLimit = RL_DATA_EXPLORER_ROW_LIMIT,
): RlMetricDataSelection {
  const safeLimit = Number.isSafeInteger(rowLimit) && rowLimit > 0 ? rowLimit : 1;
  const matching = metrics.flatMap((batch): ReadonlyArray<RlMetricDataRow> => {
    if (!Object.hasOwn(batch.values, metricKey)) return [];
    return [{ step: batch.step, wallClockMs: batch.wallClockMs, value: batch.values[metricKey]! }];
  });
  return {
    rows: matching.slice(-safeLimit),
    totalRows: matching.length,
    truncated: matching.length > safeLimit,
  };
}

export function createRlDataVisualizationSpec(
  title: string,
  selection: RlMetricDataSelection,
): RlDataVisualizationSpec {
  return {
    version: 1,
    kind: "line",
    title,
    encoding: {
      x: { field: "step", type: "quantitative" },
      y: { field: "value", type: "quantitative" },
    },
    data: selection.rows,
  };
}
