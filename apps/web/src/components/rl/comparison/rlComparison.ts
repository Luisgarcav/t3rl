import type {
  RlExperimentId,
  RlMetricBatch,
  RlResolvedManifest,
  RlRunId,
  RlRunSummary,
} from "@t3tools/contracts";

export interface RlComparisonRun {
  readonly summary: RlRunSummary;
  readonly manifest: RlResolvedManifest | null;
  readonly metrics: ReadonlyArray<RlMetricBatch>;
}

export interface RlSeedComparisonSummary {
  readonly seed: number;
  readonly representative: RlComparisonRun;
  readonly runCount: number;
  readonly metricBatchCount: number;
  readonly latestMetricStep: number | null;
  readonly latestMetricValue: number | null;
}

export interface RlAggregatePoint {
  readonly step: number;
  readonly count: number;
  readonly mean: number;
  readonly median: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly firstQuartile: number;
  readonly thirdQuartile: number;
}

export type RlComparisonWarningCode =
  | "descriptive-only"
  | "no-eligible-runs"
  | "single-seed"
  | "small-sample"
  | "missing-manifest"
  | "manifest-experiment-mismatch"
  | "duplicate-seed"
  | "provisional-run"
  | "config-drift"
  | "runner-drift"
  | "source-drift"
  | "environment-drift"
  | "partial-metric-coverage"
  | "no-metric-observations";

export interface RlComparisonWarning {
  readonly code: RlComparisonWarningCode;
  readonly severity: "info" | "warning";
  readonly message: string;
}

export interface RlComparisonAnalysis {
  readonly experimentId: RlExperimentId;
  readonly metricKey: string;
  readonly seeds: ReadonlyArray<RlSeedComparisonSummary>;
  readonly series: ReadonlyArray<RlAggregatePoint>;
  readonly excludedRunIds: ReadonlyArray<RlRunId>;
  readonly warnings: ReadonlyArray<RlComparisonWarning>;
}

interface MetricObservation {
  readonly step: number;
  readonly value: number;
}

function compareRunsNewestFirst(left: RlComparisonRun, right: RlComparisonRun): number {
  const leftTime = Date.parse(left.summary.requestedAt);
  const rightTime = Date.parse(right.summary.requestedAt);
  if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) {
    return rightTime - leftTime;
  }
  return right.summary.runId.localeCompare(left.summary.runId);
}

function stableValue(value: unknown, seen: Set<object>): string {
  if (value === null) return "null";
  if (typeof value === "number") {
    if (Number.isNaN(value)) return '"nan"';
    if (value === Number.POSITIVE_INFINITY) return '"+inf"';
    if (value === Number.NEGATIVE_INFINITY) return '"-inf"';
    return JSON.stringify(value);
  }
  if (typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "undefined") return '"undefined"';
  if (typeof value === "bigint") return `"bigint:${value.toString()}"`;
  if (typeof value === "symbol" || typeof value === "function")
    return JSON.stringify(String(value));
  if (Array.isArray(value)) return `[${value.map((entry) => stableValue(entry, seen)).join(",")}]`;

  const object = value as Record<string, unknown>;
  if (seen.has(object)) return '"circular"';
  seen.add(object);
  const result = `{${Object.keys(object)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stableValue(object[key], seen)}`)
    .join(",")}}`;
  seen.delete(object);
  return result;
}

function stableKey(value: unknown): string {
  return stableValue(value, new Set());
}

function distinctCount(values: ReadonlyArray<string>): number {
  return new Set(values).size;
}

function quantile(sortedValues: ReadonlyArray<number>, fraction: number): number {
  if (sortedValues.length === 1) return sortedValues[0] ?? 0;
  const index = (sortedValues.length - 1) * fraction;
  const lowerIndex = Math.floor(index);
  const upperIndex = Math.ceil(index);
  const lower = sortedValues[lowerIndex] ?? 0;
  const upper = sortedValues[upperIndex] ?? lower;
  return lower + (upper - lower) * (index - lowerIndex);
}

/**
 * Returns at most one finite observation per step for a run. A later batch that
 * explicitly records null/non-finite for a metric replaces an older value at
 * that same step; absent keys do not.
 */
export function selectRunMetricObservations(
  metrics: ReadonlyArray<RlMetricBatch>,
  metricKey: string,
): ReadonlyArray<MetricObservation> {
  const byStep = new Map<
    number,
    { readonly wallClockMs: number; readonly index: number; readonly value: unknown }
  >();
  metrics.forEach((batch, index) => {
    const value = batch.values[metricKey];
    if (value === undefined) return;
    const previous = byStep.get(batch.step);
    if (
      previous === undefined ||
      batch.wallClockMs > previous.wallClockMs ||
      (batch.wallClockMs === previous.wallClockMs && index > previous.index)
    ) {
      byStep.set(batch.step, { wallClockMs: batch.wallClockMs, index, value });
    }
  });

  return [...byStep.entries()]
    .flatMap(([step, entry]) =>
      typeof entry.value === "number" && Number.isFinite(entry.value)
        ? [{ step, value: entry.value }]
        : [],
    )
    .sort((left, right) => left.step - right.step);
}

export function listObservedMetricKeys(
  runs: ReadonlyArray<RlComparisonRun>,
): ReadonlyArray<string> {
  const keys = new Set<string>();
  for (const run of runs) {
    for (const batch of run.metrics) {
      for (const [key, value] of Object.entries(batch.values)) {
        if (typeof value === "number" && Number.isFinite(value)) keys.add(key);
      }
    }
  }
  return [...keys].sort();
}

export function compatibleRunsForExperiment(
  runs: ReadonlyArray<RlRunSummary>,
  experimentId: RlExperimentId,
): ReadonlyArray<RlRunSummary> {
  return runs
    .filter((run) => run.experimentId === experimentId)
    .toSorted((left, right) => {
      const leftTime = Date.parse(left.requestedAt);
      const rightTime = Date.parse(right.requestedAt);
      if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) {
        return rightTime - leftTime;
      }
      return right.runId.localeCompare(left.runId);
    });
}

function aggregateByStep(
  representatives: ReadonlyArray<RlComparisonRun>,
  metricKey: string,
): ReadonlyArray<RlAggregatePoint> {
  const valuesByStep = new Map<number, number[]>();
  for (const run of representatives) {
    for (const observation of selectRunMetricObservations(run.metrics, metricKey)) {
      const values = valuesByStep.get(observation.step) ?? [];
      values.push(observation.value);
      valuesByStep.set(observation.step, values);
    }
  }

  return [...valuesByStep.entries()]
    .sort(([left], [right]) => left - right)
    .map(([step, values]) => {
      const sorted = values.toSorted((left, right) => left - right);
      const sum = sorted.reduce((total, value) => total + value, 0);
      return {
        step,
        count: sorted.length,
        mean: sum / sorted.length,
        median: quantile(sorted, 0.5),
        minimum: sorted[0] ?? 0,
        maximum: sorted.at(-1) ?? 0,
        firstQuartile: quantile(sorted, 0.25),
        thirdQuartile: quantile(sorted, 0.75),
      };
    });
}

function latestObservation(
  metrics: ReadonlyArray<RlMetricBatch>,
  metricKey: string,
): MetricObservation | null {
  return selectRunMetricObservations(metrics, metricKey).at(-1) ?? null;
}

function warning(
  code: RlComparisonWarningCode,
  severity: RlComparisonWarning["severity"],
  message: string,
): RlComparisonWarning {
  return { code, severity, message };
}

export function analyzeRlComparison(
  selectedRuns: ReadonlyArray<RlComparisonRun>,
  experimentId: RlExperimentId,
  metricKey: string,
): RlComparisonAnalysis {
  const warnings: RlComparisonWarning[] = [
    warning(
      "descriptive-only",
      "info",
      "These are descriptive aggregates only. No confidence interval, hypothesis test, or significance claim is computed.",
    ),
  ];
  const eligibleBySeed = new Map<number, RlComparisonRun[]>();
  const excludedRunIds: RlRunId[] = [];
  let missingManifestCount = 0;
  let mismatchedManifestCount = 0;

  for (const run of selectedRuns) {
    if (run.summary.experimentId !== experimentId || run.manifest?.experimentId !== experimentId) {
      excludedRunIds.push(run.summary.runId);
      if (run.manifest === null) missingManifestCount += 1;
      else mismatchedManifestCount += 1;
      continue;
    }
    const sameSeed = eligibleBySeed.get(run.manifest.seed) ?? [];
    sameSeed.push(run);
    eligibleBySeed.set(run.manifest.seed, sameSeed);
  }

  const seedGroups = [...eligibleBySeed.entries()]
    .sort(([left], [right]) => left - right)
    .map(([seed, runs]) => ({ seed, runs: runs.toSorted(compareRunsNewestFirst) }));
  const representatives = seedGroups.flatMap((group) => group.runs.slice(0, 1));
  const series = aggregateByStep(representatives, metricKey);
  const seeds = seedGroups.map((group): RlSeedComparisonSummary => {
    const representative = group.runs[0];
    if (representative === undefined) throw new Error("Seed group cannot be empty");
    const latest = latestObservation(representative.metrics, metricKey);
    return {
      seed: group.seed,
      representative,
      runCount: group.runs.length,
      metricBatchCount: representative.metrics.length,
      latestMetricStep: latest?.step ?? null,
      latestMetricValue: latest?.value ?? null,
    };
  });

  if (missingManifestCount > 0) {
    warnings.push(
      warning(
        "missing-manifest",
        "warning",
        `${missingManifestCount} selected run${missingManifestCount === 1 ? " was" : "s were"} excluded because ${missingManifestCount === 1 ? "its resolved manifest is" : "their resolved manifests are"} unavailable, so ${missingManifestCount === 1 ? "its seed cannot" : "their seeds cannot"} be verified.`,
      ),
    );
  }
  if (mismatchedManifestCount > 0) {
    warnings.push(
      warning(
        "manifest-experiment-mismatch",
        "warning",
        `${mismatchedManifestCount} selected run${mismatchedManifestCount === 1 ? " was" : "s were"} excluded because ${mismatchedManifestCount === 1 ? "its manifest does" : "their manifests do"} not match experiment ${experimentId}.`,
      ),
    );
  }
  const duplicateCount = seedGroups.reduce((total, group) => total + group.runs.length - 1, 0);
  if (duplicateCount > 0) {
    warnings.push(
      warning(
        "duplicate-seed",
        "info",
        `${duplicateCount} duplicate run${duplicateCount === 1 ? "" : "s"} across selected seeds ${duplicateCount === 1 ? "was" : "were"} not counted; the most recently requested run represents each seed.`,
      ),
    );
  }

  if (representatives.length === 0) {
    warnings.push(
      warning(
        "no-eligible-runs",
        "warning",
        "No selected run has a compatible resolved manifest, so no aggregate can be computed.",
      ),
    );
  } else if (representatives.length === 1) {
    warnings.push(
      warning(
        "single-seed",
        "warning",
        "Only one verified seed is represented. The curve is a single run, not a multi-seed estimate.",
      ),
    );
  } else if (representatives.length < 5) {
    warnings.push(
      warning(
        "small-sample",
        "warning",
        `Only ${representatives.length} verified seeds are represented. Mean, median, range, and quartiles can be unstable at this sample size.`,
      ),
    );
  }

  if (representatives.some((run) => run.summary.state !== "completed")) {
    warnings.push(
      warning(
        "provisional-run",
        "warning",
        "At least one representative run is not completed. Its metrics are provisional or truncated.",
      ),
    );
  }

  if (distinctCount(representatives.map((run) => stableKey(run.manifest?.effectiveConfig))) > 1) {
    warnings.push(
      warning(
        "config-drift",
        "warning",
        "Resolved configurations differ across seeds; the aggregate is not a pure seed-only comparison.",
      ),
    );
  }
  if (
    distinctCount(
      representatives.map((run) => `${run.manifest?.runnerId}@${run.manifest?.runnerVersion}`),
    ) > 1
  ) {
    warnings.push(
      warning("runner-drift", "warning", "Runner identity or version differs across seeds."),
    );
  }
  if (
    distinctCount(
      representatives.map(
        (run) =>
          `${run.manifest?.sourceRevision ?? "unknown"}:${String(run.manifest?.sourceDirty)}`,
      ),
    ) > 1
  ) {
    warnings.push(
      warning(
        "source-drift",
        "warning",
        "Source revision or dirty-tree evidence differs across seeds; unknown evidence is not treated as clean.",
      ),
    );
  }
  if (
    distinctCount(representatives.map((run) => run.manifest?.environmentFingerprint ?? "unknown")) >
    1
  ) {
    warnings.push(
      warning(
        "environment-drift",
        "warning",
        "Runtime environment fingerprints differ across seeds.",
      ),
    );
  }

  if (series.length === 0 && representatives.length > 0) {
    warnings.push(
      warning(
        "no-metric-observations",
        "warning",
        `No finite ${metricKey} observations are present in the selected snapshots. Missing and non-finite values are not replaced.`,
      ),
    );
  } else if (series.some((point) => point.count < representatives.length)) {
    warnings.push(
      warning(
        "partial-metric-coverage",
        "info",
        "Some steps are missing values for one or more seeds. Aggregates use only finite observations at the exact step; no interpolation or imputation is applied.",
      ),
    );
  }

  return { experimentId, metricKey, seeds, series, excludedRunIds, warnings };
}

export function sampleAggregateSeries(
  series: ReadonlyArray<RlAggregatePoint>,
  maxPoints = 240,
): ReadonlyArray<RlAggregatePoint> {
  if (maxPoints < 2 || series.length <= maxPoints) return series;
  const stride = Math.ceil((series.length - 1) / (maxPoints - 1));
  const sampled = series.filter((_, index) => index % stride === 0);
  const last = series.at(-1);
  if (last !== undefined && sampled.at(-1) !== last) sampled.push(last);
  return sampled;
}
