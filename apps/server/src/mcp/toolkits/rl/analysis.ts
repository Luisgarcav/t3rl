import type { RlMetricBatch } from "@t3tools/contracts";

export interface RlAgentMetricSummary {
  readonly key: string;
  readonly observations: number;
  readonly finiteCount: number;
  readonly nullCount: number;
  readonly nanCount: number;
  readonly positiveInfinityCount: number;
  readonly negativeInfinityCount: number;
  readonly firstStep: number | null;
  readonly lastStep: number | null;
  readonly firstFiniteValue: number | null;
  readonly lastFiniteValue: number | null;
  readonly minimum: number | null;
  readonly maximum: number | null;
  readonly mean: number | null;
}

export interface RlAgentMetricQuery {
  readonly availableMetricKeys: ReadonlyArray<string>;
  readonly totalMatchingBatches: number;
  readonly returnedBatches: number;
  readonly truncated: boolean;
  readonly batches: ReadonlyArray<RlMetricBatch>;
}

export function listMetricKeys(metrics: ReadonlyArray<RlMetricBatch>): ReadonlyArray<string> {
  const keys = new Set<string>();
  for (const batch of metrics) {
    for (const key of Object.keys(batch.values)) keys.add(key);
  }
  return [...keys].sort((left, right) => left.localeCompare(right));
}

export function summarizeMetrics(
  metrics: ReadonlyArray<RlMetricBatch>,
  selectedKeys?: ReadonlyArray<string>,
): ReadonlyArray<RlAgentMetricSummary> {
  const keys = selectedKeys ?? listMetricKeys(metrics);
  return [...new Set(keys)]
    .sort((left, right) => left.localeCompare(right))
    .map((key) => {
      let observations = 0;
      let finiteCount = 0;
      let nullCount = 0;
      let nanCount = 0;
      let positiveInfinityCount = 0;
      let negativeInfinityCount = 0;
      let firstStep: number | null = null;
      let lastStep: number | null = null;
      let firstFiniteValue: number | null = null;
      let lastFiniteValue: number | null = null;
      let minimum = Number.POSITIVE_INFINITY;
      let maximum = Number.NEGATIVE_INFINITY;
      let sum = 0;

      for (const batch of metrics) {
        if (!Object.hasOwn(batch.values, key)) continue;
        const value = batch.values[key];
        if (value === undefined) continue;
        observations += 1;
        firstStep ??= batch.step;
        lastStep = batch.step;
        if (value === null) {
          nullCount += 1;
        } else if (value === "nan") {
          nanCount += 1;
        } else if (value === "+inf") {
          positiveInfinityCount += 1;
        } else if (value === "-inf") {
          negativeInfinityCount += 1;
        } else {
          finiteCount += 1;
          firstFiniteValue ??= value;
          lastFiniteValue = value;
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
          sum += value;
        }
      }

      return {
        key,
        observations,
        finiteCount,
        nullCount,
        nanCount,
        positiveInfinityCount,
        negativeInfinityCount,
        firstStep,
        lastStep,
        firstFiniteValue,
        lastFiniteValue,
        minimum: finiteCount === 0 ? null : minimum,
        maximum: finiteCount === 0 ? null : maximum,
        mean: finiteCount === 0 ? null : sum / finiteCount,
      };
    });
}

export function queryMetrics(input: {
  readonly metrics: ReadonlyArray<RlMetricBatch>;
  readonly metricKeys: ReadonlyArray<string>;
  readonly stepFrom?: number | undefined;
  readonly stepTo?: number | undefined;
  readonly limit: number;
}): RlAgentMetricQuery {
  const selectedKeys = [...new Set(input.metricKeys)];
  const matching = input.metrics
    .map((batch, sourceIndex) => ({ batch, sourceIndex }))
    .filter(
      ({ batch }) =>
        (input.stepFrom === undefined || batch.step >= input.stepFrom) &&
        (input.stepTo === undefined || batch.step <= input.stepTo) &&
        selectedKeys.some((key) => Object.hasOwn(batch.values, key)),
    )
    .sort(
      (left, right) =>
        left.batch.step - right.batch.step ||
        left.batch.wallClockMs - right.batch.wallClockMs ||
        left.sourceIndex - right.sourceIndex,
    )
    .map(({ batch }) => ({
      step: batch.step,
      wallClockMs: batch.wallClockMs,
      values: Object.fromEntries(
        selectedKeys.flatMap((key) => {
          const value = batch.values[key];
          return Object.hasOwn(batch.values, key) && value !== undefined
            ? ([[key, value]] as const)
            : [];
        }),
      ),
    }));
  const batches = matching.slice(-input.limit);
  return {
    availableMetricKeys: listMetricKeys(input.metrics),
    totalMatchingBatches: matching.length,
    returnedBatches: batches.length,
    truncated: batches.length < matching.length,
    batches,
  };
}
