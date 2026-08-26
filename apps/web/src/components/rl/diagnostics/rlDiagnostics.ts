import type { RlMetricBatch, RlRunState } from "@t3tools/contracts";

export const RL_DIAGNOSTIC_KINDS = [
  "non-finite-metrics",
  "return-collapse",
  "excessive-kl",
  "low-entropy",
  "divergent-value-loss",
  "stalled-stream",
  "train-eval-gap",
] as const;

export type RlDiagnosticKind = (typeof RL_DIAGNOSTIC_KINDS)[number];
export type RlDiagnosticSeverity = "critical" | "warning" | "info";
export type RlDiagnosticCheckStatus = "finding" | "clear" | "insufficient";

export interface RlDiagnosticEvidence {
  readonly label: string;
  readonly value: string;
  readonly metricKey?: string;
  readonly step?: number;
}

export interface RlDiagnosticFinding {
  /** Stable across repeated analysis of the same run, so React and callers can key by it. */
  readonly id: RlDiagnosticKind;
  readonly kind: RlDiagnosticKind;
  readonly severity: RlDiagnosticSeverity;
  readonly title: string;
  readonly explanation: string;
  readonly evidence: ReadonlyArray<RlDiagnosticEvidence>;
  readonly suggestedChecks: ReadonlyArray<string>;
  readonly limitations: ReadonlyArray<string>;
}

export interface RlDiagnosticCheck {
  readonly kind: RlDiagnosticKind;
  readonly label: string;
  readonly status: RlDiagnosticCheckStatus;
  readonly reason: string;
}

export interface RlDiagnosticReport {
  readonly findings: ReadonlyArray<RlDiagnosticFinding>;
  readonly checks: ReadonlyArray<RlDiagnosticCheck>;
  readonly highestSeverity: RlDiagnosticSeverity | null;
  readonly analyzedBatchCount: number;
}

/**
 * Screening references, not claims about universally correct RL behavior.
 * Consumers can replace them for a known algorithm or experiment. Every
 * finding repeats the relevant limitation in user-facing language.
 */
export interface RlDiagnosticPolicy {
  readonly trendMinimumPoints: number;
  readonly trendWindowSize: number;
  readonly returnCollapseRelativeDrop: number;
  readonly returnCollapseSpreadMultiplier: number;
  readonly approximateKlMinimumPoints: number;
  readonly approximateKlRecentWindow: number;
  readonly approximateKlUpperReference: number | null;
  readonly entropyMinimumPoints: number;
  readonly entropyRetainedFraction: number;
  readonly valueLossMinimumPoints: number;
  readonly valueLossGrowthFactor: number;
  readonly stallAfterMs: number;
  readonly trainEvalMinimumTrainingPoints: number;
  readonly trainEvalRelativeGap: number;
  readonly trainEvalSpreadMultiplier: number;
}

export const DEFAULT_RL_DIAGNOSTIC_POLICY: RlDiagnosticPolicy = {
  trendMinimumPoints: 12,
  trendWindowSize: 4,
  returnCollapseRelativeDrop: 0.5,
  returnCollapseSpreadMultiplier: 2,
  approximateKlMinimumPoints: 5,
  approximateKlRecentWindow: 3,
  approximateKlUpperReference: 0.05,
  entropyMinimumPoints: 12,
  entropyRetainedFraction: 0.25,
  valueLossMinimumPoints: 12,
  valueLossGrowthFactor: 8,
  stallAfterMs: 15_000,
  trainEvalMinimumTrainingPoints: 8,
  trainEvalRelativeGap: 0.35,
  trainEvalSpreadMultiplier: 2,
};

export interface AnalyzeRlDiagnosticsInput {
  readonly metrics: ReadonlyArray<RlMetricBatch>;
  readonly runState: RlRunState;
  /** Server timestamp from the authoritative run projection. */
  readonly lastMessageAt: string | null;
  /** Required rather than read internally, keeping the engine deterministic. */
  readonly nowMs: number;
  readonly policy?: Partial<RlDiagnosticPolicy>;
}

interface MetricPoint {
  readonly step: number;
  readonly wallClockMs: number;
  readonly value: number;
  readonly sourceIndex: number;
}

interface CheckOutcome {
  readonly check: RlDiagnosticCheck;
  readonly finding?: RlDiagnosticFinding;
}

const CHECK_LABELS: Readonly<Record<RlDiagnosticKind, string>> = {
  "non-finite-metrics": "Non-finite values",
  "return-collapse": "Return collapse",
  "excessive-kl": "Approximate KL",
  "low-entropy": "Policy entropy",
  "divergent-value-loss": "Value loss divergence",
  "stalled-stream": "Metric stream freshness",
  "train-eval-gap": "Training/evaluation gap",
};

const SEVERITY_ORDER: Readonly<Record<RlDiagnosticSeverity, number>> = {
  critical: 0,
  warning: 1,
  info: 2,
};

const KIND_ORDER = new Map(RL_DIAGNOSTIC_KINDS.map((kind, index) => [kind, index] as const));

function policyWithDefaults(
  overrides: Partial<RlDiagnosticPolicy> | undefined,
): RlDiagnosticPolicy {
  return { ...DEFAULT_RL_DIAGNOSTIC_POLICY, ...overrides };
}

function finiteSeries(
  metrics: ReadonlyArray<RlMetricBatch>,
  key: string,
): ReadonlyArray<MetricPoint> {
  return metrics
    .flatMap((batch, sourceIndex) => {
      const value = batch.values[key];
      return typeof value === "number" && Number.isFinite(value)
        ? [{ step: batch.step, wallClockMs: batch.wallClockMs, value, sourceIndex }]
        : [];
    })
    .sort(
      (left, right) =>
        left.step - right.step ||
        left.wallClockMs - right.wallClockMs ||
        left.sourceIndex - right.sourceIndex,
    );
}

function median(values: ReadonlyArray<number>): number {
  const ordered = values.toSorted((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  const right = ordered[middle] ?? 0;
  return ordered.length % 2 === 0 ? ((ordered[middle - 1] ?? right) + right) / 2 : right;
}

function quantile(values: ReadonlyArray<number>, fraction: number): number {
  const ordered = values.toSorted((left, right) => left - right);
  if (ordered.length === 0) return 0;
  const index = (ordered.length - 1) * fraction;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const lowerValue = ordered[lower] ?? 0;
  const upperValue = ordered[upper] ?? lowerValue;
  return lowerValue + (upperValue - lowerValue) * (index - lower);
}

function interquartileRange(values: ReadonlyArray<number>): number {
  return quantile(values, 0.75) - quantile(values, 0.25);
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 10_000 || magnitude < 0.0001) return value.toExponential(3);
  return String(Number(value.toPrecision(5)));
}

function formatDuration(milliseconds: number): string {
  const seconds = Math.round(milliseconds / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
}

function result(
  kind: RlDiagnosticKind,
  status: RlDiagnosticCheckStatus,
  reason: string,
  finding?: RlDiagnosticFinding,
): CheckOutcome {
  return {
    check: { kind, label: CHECK_LABELS[kind], status, reason },
    ...(finding === undefined ? {} : { finding }),
  };
}

function finding(
  kind: RlDiagnosticKind,
  value: Omit<RlDiagnosticFinding, "id" | "kind">,
): RlDiagnosticFinding {
  return { id: kind, kind, ...value };
}

function checkNonFinite(metrics: ReadonlyArray<RlMetricBatch>): CheckOutcome {
  const observations = metrics.flatMap((batch) =>
    Object.entries(batch.values)
      .sort(([left], [right]) => left.localeCompare(right))
      .flatMap(([metricKey, value]) => {
        const nonFinite =
          value === "nan" ||
          value === "+inf" ||
          value === "-inf" ||
          (typeof value === "number" && !Number.isFinite(value));
        return nonFinite ? [{ metricKey, step: batch.step, value: String(value) }] : [];
      }),
  );

  if (metrics.length === 0) {
    return result("non-finite-metrics", "insufficient", "No metric batch has been observed yet.");
  }
  if (observations.length === 0) {
    return result("non-finite-metrics", "clear", "All captured metric values are finite or null.");
  }

  const sample = observations.slice(0, 6);
  return result(
    "non-finite-metrics",
    "finding",
    `${observations.length} non-finite value${observations.length === 1 ? " was" : "s were"} observed.`,
    finding("non-finite-metrics", {
      severity: "critical",
      title: "Non-finite metrics were emitted",
      explanation:
        "The worker explicitly reported NaN or infinity. This is direct numerical evidence, not a trend inferred from a quiet stream.",
      evidence: [
        ...sample.map((observation) => ({
          label: observation.metricKey,
          value: observation.value.toUpperCase(),
          metricKey: observation.metricKey,
          step: observation.step,
        })),
        ...(observations.length > sample.length
          ? [
              {
                label: "Additional observations",
                value: String(observations.length - sample.length),
              },
            ]
          : []),
      ],
      suggestedChecks: [
        "Inspect the first affected step and the worker log for the originating operation.",
        "Check observations, rewards, normalization and optimizer inputs before trusting later metrics.",
      ],
      limitations: [
        "This identifies a numerical symptom; it does not identify which model or environment operation caused it.",
      ],
    }),
  );
}

function checkReturnCollapse(
  metrics: ReadonlyArray<RlMetricBatch>,
  policy: RlDiagnosticPolicy,
): CheckOutcome {
  const kind = "return-collapse" as const;
  const points = finiteSeries(metrics, "train/return");
  const windowSize = Math.max(2, Math.floor(policy.trendWindowSize));
  const minimum = Math.max(policy.trendMinimumPoints, windowSize * 3);
  if (points.length < minimum) {
    return result(
      kind,
      "insufficient",
      `Need at least ${minimum} finite training-return observations; captured ${points.length}.`,
    );
  }

  const values = points.map((point) => point.value);
  const recent = values.slice(-windowSize);
  const recentMedian = median(recent);
  let peakMedian = Number.NEGATIVE_INFINITY;
  let peakEndIndex = windowSize - 1;
  for (let end = windowSize; end <= values.length - windowSize; end += 1) {
    const candidate = median(values.slice(end - windowSize, end));
    if (candidate > peakMedian) {
      peakMedian = candidate;
      peakEndIndex = end - 1;
    }
  }
  const drop = peakMedian - recentMedian;
  const peakWindow = values.slice(peakEndIndex + 1 - windowSize, peakEndIndex + 1);
  // Compare the drop with variation inside the two sustained windows. Using
  // the IQR of the entire trace would let the collapse itself inflate the noise
  // estimate and hide the signal it is supposed to screen for.
  const localSpread = Math.max(interquartileRange(peakWindow), interquartileRange(recent));
  const scale = Math.max(Math.abs(peakMedian), localSpread, 1);
  const relativeDrop = drop / scale;
  const spreadBarrier = localSpread * policy.returnCollapseSpreadMultiplier;
  const collapsed =
    drop > 0 && relativeDrop >= policy.returnCollapseRelativeDrop && drop >= spreadBarrier;

  if (!collapsed) {
    return result(
      kind,
      "clear",
      "The latest return window has not fallen far enough below an earlier sustained window.",
    );
  }

  return result(
    kind,
    "finding",
    "The latest sustained return window is materially below an earlier window in this run.",
    finding(kind, {
      severity: "warning",
      title: "Training return appears to have collapsed",
      explanation:
        "A recent window is substantially below the strongest earlier window after accounting for this trace’s reward scale and spread.",
      evidence: [
        {
          label: `Earlier ${windowSize}-point median`,
          value: formatNumber(peakMedian),
          metricKey: "train/return",
          step: points[peakEndIndex]!.step,
        },
        {
          label: `Latest ${windowSize}-point median`,
          value: formatNumber(recentMedian),
          metricKey: "train/return",
          step: points.at(-1)!.step,
        },
        { label: "Scale-adjusted drop", value: `${formatNumber(relativeDrop * 100)}%` },
      ],
      suggestedChecks: [
        "Inspect the episode summaries around the peak and the start of the decline.",
        "Check for environment, reward, learning-rate or normalization changes during the run.",
      ],
      limitations: [
        "The comparison is relative to this run; reward scales and expected variance differ by environment.",
        "A short regression can be normal exploration noise and is not proof of irreversible policy collapse.",
      ],
    }),
  );
}

function checkApproximateKl(
  metrics: ReadonlyArray<RlMetricBatch>,
  policy: RlDiagnosticPolicy,
): CheckOutcome {
  const kind = "excessive-kl" as const;
  const points = finiteSeries(metrics, "train/approx_kl");
  if (policy.approximateKlUpperReference === null) {
    return result(kind, "insufficient", "No approximate-KL screening reference is configured.");
  }
  const windowSize = Math.max(2, Math.floor(policy.approximateKlRecentWindow));
  const minimum = Math.max(policy.approximateKlMinimumPoints, windowSize);
  if (points.length < minimum) {
    return result(
      kind,
      "insufficient",
      `Need at least ${minimum} finite approximate-KL observations; captured ${points.length}.`,
    );
  }

  const recent = points.slice(-windowSize);
  const reference = policy.approximateKlUpperReference;
  const overReference = recent.filter((point) => point.value > reference).length;
  const recentMedian = median(recent.map((point) => point.value));
  const sustained = overReference >= Math.ceil(windowSize * 0.67) && recentMedian > reference;
  if (!sustained) {
    return result(
      kind,
      "clear",
      "Recent approximate KL does not sustain a breach of the configured screening reference.",
    );
  }

  return result(
    kind,
    "finding",
    "Recent approximate KL is repeatedly above the configured screening reference.",
    finding(kind, {
      severity: "warning",
      title: "Policy updates may be too large",
      explanation:
        "Most observations in the recent window exceed the configured approximate-KL screening reference, which can indicate abrupt PPO updates.",
      evidence: [
        {
          label: `Latest ${windowSize}-point median`,
          value: formatNumber(recentMedian),
          metricKey: "train/approx_kl",
          step: recent.at(-1)!.step,
        },
        { label: "Configured screening reference", value: formatNumber(reference) },
        { label: "Recent observations above reference", value: `${overReference}/${windowSize}` },
      ],
      suggestedChecks: [
        "Compare learning rate, clipping range, batch size and epoch count with a stable run.",
        "Inspect whether the return or value loss changed at the same optimizer updates.",
      ],
      limitations: [
        "The reference is a configurable screening aid, not a universal PPO correctness threshold.",
        "Approximate KL is estimator- and implementation-dependent; interpret it with return and loss evidence.",
      ],
    }),
  );
}

function checkEntropy(
  metrics: ReadonlyArray<RlMetricBatch>,
  policy: RlDiagnosticPolicy,
): CheckOutcome {
  const kind = "low-entropy" as const;
  const points = finiteSeries(metrics, "train/entropy");
  const windowSize = Math.max(2, Math.floor(policy.trendWindowSize));
  const minimum = Math.max(policy.entropyMinimumPoints, windowSize * 2);
  if (points.length < minimum) {
    return result(
      kind,
      "insufficient",
      `Need at least ${minimum} finite entropy observations; captured ${points.length}.`,
    );
  }

  const earlyMedian = median(points.slice(0, windowSize).map((point) => point.value));
  const recentMedian = median(points.slice(-windowSize).map((point) => point.value));
  if (earlyMedian <= 0) {
    return result(
      kind,
      "insufficient",
      "The run has no positive early entropy baseline for a relative comparison.",
    );
  }
  const retainedFraction = recentMedian / earlyMedian;
  if (recentMedian >= 0 && retainedFraction > policy.entropyRetainedFraction) {
    return result(
      kind,
      "clear",
      "Recent entropy remains above the configured fraction of its baseline.",
    );
  }

  return result(
    kind,
    "finding",
    "Recent policy entropy retains only a small fraction of this run’s early baseline.",
    finding(kind, {
      severity: "warning",
      title: "Policy entropy is low relative to this run",
      explanation:
        "The recent entropy window is much lower than the run’s own early window, suggesting that action selection has become substantially less diverse.",
      evidence: [
        {
          label: `Early ${windowSize}-point median`,
          value: formatNumber(earlyMedian),
          metricKey: "train/entropy",
          step: points[windowSize - 1]!.step,
        },
        {
          label: `Latest ${windowSize}-point median`,
          value: formatNumber(recentMedian),
          metricKey: "train/entropy",
          step: points.at(-1)!.step,
        },
        { label: "Baseline retained", value: `${formatNumber(retainedFraction * 100)}%` },
      ],
      suggestedChecks: [
        "Compare the entropy decline with episodic return and action behavior.",
        "Inspect entropy coefficient, reward scale and whether the policy settled on one action prematurely.",
      ],
      limitations: [
        "Lower entropy can be expected as a policy becomes confident; it is not automatically a defect.",
        "Without the action distribution and action-space maximum, this check can detect relative collapse only.",
      ],
    }),
  );
}

function checkValueLoss(
  metrics: ReadonlyArray<RlMetricBatch>,
  policy: RlDiagnosticPolicy,
): CheckOutcome {
  const kind = "divergent-value-loss" as const;
  const points = finiteSeries(metrics, "train/value_loss");
  const windowSize = Math.max(2, Math.floor(policy.trendWindowSize));
  const minimum = Math.max(policy.valueLossMinimumPoints, windowSize * 2);
  if (points.length < minimum) {
    return result(
      kind,
      "insufficient",
      `Need at least ${minimum} finite value-loss observations; captured ${points.length}.`,
    );
  }

  const magnitudes = points.map((point) => Math.abs(point.value));
  const earlyMedian = median(magnitudes.slice(0, windowSize));
  const recent = magnitudes.slice(-windowSize);
  const recentMedian = median(recent);
  const numericalFloor = Math.max(median(magnitudes) * 0.01, 1e-9);
  const baseline = Math.max(earlyMedian, numericalFloor);
  const ratio = recentMedian / baseline;
  const sustained =
    ratio >= policy.valueLossGrowthFactor &&
    recent.filter((value) => value >= baseline * policy.valueLossGrowthFactor).length >=
      Math.ceil(windowSize * 0.75);
  if (!sustained) {
    return result(
      kind,
      "clear",
      "Recent value-loss magnitude is not persistently above its early-run baseline.",
    );
  }

  return result(
    kind,
    "finding",
    "Recent value-loss magnitude is persistently many times its early-run baseline.",
    finding(kind, {
      severity: "warning",
      title: "Value loss appears to be diverging",
      explanation:
        "The latest value-loss window is persistently larger than this run’s early baseline, which can indicate unstable targets or critic optimization.",
      evidence: [
        {
          label: `Early ${windowSize}-point magnitude median`,
          value: formatNumber(earlyMedian),
          metricKey: "train/value_loss",
          step: points[windowSize - 1]!.step,
        },
        {
          label: `Latest ${windowSize}-point magnitude median`,
          value: formatNumber(recentMedian),
          metricKey: "train/value_loss",
          step: points.at(-1)!.step,
        },
        { label: "Growth relative to baseline", value: `${formatNumber(ratio)}×` },
      ],
      suggestedChecks: [
        "Inspect reward scale, return normalization and value-function targets near the onset.",
        "Compare value coefficient, learning rate, clipping and batch composition with a stable run.",
      ],
      limitations: [
        "Value-loss scale depends on reward scale and implementation; the check uses only relative growth within this run.",
        "A larger value loss can accompany a legitimate change in target magnitude and is not proof of critic divergence.",
      ],
    }),
  );
}

function checkStreamFreshness(
  input: AnalyzeRlDiagnosticsInput,
  policy: RlDiagnosticPolicy,
): CheckOutcome {
  const kind = "stalled-stream" as const;
  if (input.runState !== "running") {
    return result(
      kind,
      "insufficient",
      "Stream freshness is evaluated only while the run is running.",
    );
  }
  if (input.lastMessageAt === null) {
    return result(kind, "insufficient", "The run has no authoritative last-message timestamp yet.");
  }
  const lastMessageMs = Date.parse(input.lastMessageAt);
  if (!Number.isFinite(lastMessageMs) || !Number.isFinite(input.nowMs)) {
    return result(kind, "insufficient", "The freshness timestamps could not be interpreted.");
  }
  const ageMs = input.nowMs - lastMessageMs;
  if (ageMs < 0) {
    return result(
      kind,
      "insufficient",
      "The last-message timestamp is ahead of the observation clock.",
    );
  }
  if (ageMs <= policy.stallAfterMs) {
    return result(kind, "clear", `The latest server message is ${formatDuration(ageMs)} old.`);
  }

  return result(
    kind,
    "finding",
    "The authoritative run is still running, but its last server message is older than the configured freshness window.",
    finding(kind, {
      severity: "warning",
      title: "The metric stream may be stalled",
      explanation:
        "No lifecycle, metric or artifact message has updated the run within the configured freshness window.",
      evidence: [
        { label: "Last server message", value: input.lastMessageAt },
        { label: "Observed silence", value: formatDuration(ageMs) },
        { label: "Configured freshness window", value: formatDuration(policy.stallAfterMs) },
      ],
      suggestedChecks: [
        "Check the worker process and bounded log before deciding whether to cancel the run.",
        "Confirm whether the environment or evaluation phase is expected to be quiet for this long.",
      ],
      limitations: [
        "Silence is diagnostic evidence only; it does not prove that the worker exited or that training failed.",
        "The freshness window is an operational setting and should match the worker’s expected emission cadence.",
        "Server and browser clocks are assumed to be synchronized; clock skew can distort freshness.",
      ],
    }),
  );
}

function checkTrainEvalGap(
  metrics: ReadonlyArray<RlMetricBatch>,
  policy: RlDiagnosticPolicy,
): CheckOutcome {
  const kind = "train-eval-gap" as const;
  const training = finiteSeries(metrics, "train/return");
  const evaluation = finiteSeries(metrics, "eval/return");
  if (training.length < policy.trainEvalMinimumTrainingPoints || evaluation.length === 0) {
    return result(
      kind,
      "insufficient",
      `Need ${policy.trainEvalMinimumTrainingPoints} finite training returns and at least one evaluation return; captured ${training.length} and ${evaluation.length}.`,
    );
  }

  const windowSize = Math.min(Math.max(2, Math.floor(policy.trendWindowSize)), training.length);
  const trainingValues = training.map((point) => point.value);
  const trainingMedian = median(trainingValues.slice(-windowSize));
  const evaluationPoint = evaluation.at(-1)!;
  const difference = evaluationPoint.value - trainingMedian;
  const absoluteDifference = Math.abs(difference);
  const spread = interquartileRange(trainingValues);
  const scale = Math.max(Math.abs(trainingMedian), Math.abs(evaluationPoint.value), spread, 1);
  const relativeGap = absoluteDifference / scale;
  const material =
    relativeGap >= policy.trainEvalRelativeGap &&
    absoluteDifference >= spread * policy.trainEvalSpreadMultiplier;
  if (!material) {
    return result(
      kind,
      "clear",
      "Evaluation return is not materially separated from the latest training-return window.",
    );
  }

  const evaluationIsLower = difference < 0;
  return result(
    kind,
    "finding",
    "Evaluation return is materially separated from the latest training-return window.",
    finding(kind, {
      severity: evaluationIsLower ? "warning" : "info",
      title: evaluationIsLower
        ? "Evaluation return trails training return"
        : "Evaluation return exceeds recent training return",
      explanation: evaluationIsLower
        ? "The final evaluation aggregate is substantially below the recent training-return median, which may indicate a generalization or measurement gap."
        : "The final evaluation aggregate is substantially above the recent training-return median, so the two collection policies are producing different evidence.",
      evidence: [
        {
          label: `Latest ${windowSize}-point training median`,
          value: formatNumber(trainingMedian),
          metricKey: "train/return",
          step: training.at(-1)!.step,
        },
        {
          label: "Latest evaluation return",
          value: formatNumber(evaluationPoint.value),
          metricKey: "eval/return",
          step: evaluationPoint.step,
        },
        { label: "Scale-adjusted gap", value: `${formatNumber(relativeGap * 100)}%` },
      ],
      suggestedChecks: [
        "Compare evaluation seeds, deterministic policy settings and episode counts with training collection.",
        "Inspect the retained evaluation summary rather than drawing a conclusion from one aggregate alone.",
      ],
      limitations: [
        "Training and evaluation returns can differ legitimately because their policies, seeds and aggregation windows differ.",
        "A single evaluation aggregate does not provide uncertainty or establish a generalization failure.",
      ],
    }),
  );
}

export function analyzeRlDiagnostics(input: AnalyzeRlDiagnosticsInput): RlDiagnosticReport {
  const policy = policyWithDefaults(input.policy);
  const outcomes = [
    checkNonFinite(input.metrics),
    checkReturnCollapse(input.metrics, policy),
    checkApproximateKl(input.metrics, policy),
    checkEntropy(input.metrics, policy),
    checkValueLoss(input.metrics, policy),
    checkStreamFreshness(input, policy),
    checkTrainEvalGap(input.metrics, policy),
  ];
  const findings = outcomes
    .flatMap((outcome) => (outcome.finding === undefined ? [] : [outcome.finding]))
    .sort(
      (left, right) =>
        SEVERITY_ORDER[left.severity] - SEVERITY_ORDER[right.severity] ||
        (KIND_ORDER.get(left.kind) ?? 0) - (KIND_ORDER.get(right.kind) ?? 0),
    );

  return {
    findings,
    checks: outcomes.map((outcome) => outcome.check),
    highestSeverity: findings[0]?.severity ?? null,
    analyzedBatchCount: input.metrics.length,
  };
}
