import type { RlStudyComparison, RlStudyExcludedRun } from "@t3tools/contracts";

import { Alert, AlertDescription, AlertTitle } from "../../ui/alert";
import { formatRlMetricValue } from "../rlPresentation";

const CONCLUSION_LABELS = {
  interval: "Confidence interval available",
  "not-enough-evidence": "Not enough evidence",
  "incompatible-protocol": "Incompatible evaluation evidence",
  "missing-pairs": "Required pairs are missing",
  "unsupported-estimator": "Estimator is unsupported",
  "computation-budget-exceeded": "Comparison exceeds the computation budget",
  "invalid-variants": "Choose distinct study variants",
} satisfies Record<RlStudyComparison["conclusion"], string>;

const CONCLUSION_DESCRIPTIONS = {
  interval: "The interval estimates uncertainty in the candidate-minus-baseline mean difference.",
  "not-enough-evidence":
    "The retained evidence cannot support an interval. Inspect the seed pairs and exclusions before drawing a conclusion.",
  "incompatible-protocol":
    "The evaluation protocol, sample pairing, or generation seeds do not match. Use comparable evidence before requesting an interval.",
  "missing-pairs":
    "The selected missing-pair policy requires all pairs. Complete the missing evidence or explicitly select an exclusion policy.",
  "unsupported-estimator":
    "This estimator is unavailable. Select a supported statistic before interpreting the comparison.",
  "computation-budget-exceeded":
    "Reduce the requested resample count or study size to calculate within the server's comparison budget.",
  "invalid-variants": "The baseline and candidate must name different variants in the study.",
} satisfies Record<RlStudyComparison["conclusion"], string>;

const EXCLUSION_LABELS = {
  "not-completed": "Run has not completed",
  failed: "Run failed",
  cancelled: "Run was cancelled",
  interrupted: "Run was interrupted",
  "missing-run": "Run is unavailable",
  "missing-evaluation": "Verified evaluation samples are unavailable",
  "invalid-evaluation": "Evaluation evidence could not be verified",
  "incompatible-protocol": "Evaluation protocol does not match",
  "missing-metric": "Evaluation does not contain the selected metric",
} satisfies Record<RlStudyExcludedRun["reason"], string>;

const STATISTIC_LABELS = {
  mean: "Mean",
  median: "Median",
  "paired-mean-delta": "Paired mean difference",
} satisfies Record<RlStudyComparison["estimator"]["statistic"], string>;

export function RlStudyComparisonResult({
  comparison,
}: {
  readonly comparison: RlStudyComparison;
}) {
  const { estimator } = comparison;
  const hierarchical = estimator.statisticalUnit === "paired-sample-within-run-seed";
  const hasInterval = comparison.conclusion === "interval" && comparison.interval !== null;
  const metricValue = (value: number | null) => (value === null ? "—" : formatRlMetricValue(value));

  return (
    <div aria-live="polite" className="space-y-3">
      <Alert variant={hasInterval ? "info" : "warning"}>
        <AlertTitle>{CONCLUSION_LABELS[comparison.conclusion]}</AlertTitle>
        <AlertDescription>{CONCLUSION_DESCRIPTIONS[comparison.conclusion]}</AlertDescription>
      </Alert>
      <p className="break-all text-xs text-muted-foreground">
        Study {comparison.studyId} · {comparison.candidateLabel} minus {comparison.baselineLabel}
        {" · "}
        {comparison.metricKey}
      </p>
      <dl className="grid gap-x-4 gap-y-2 text-xs sm:grid-cols-2 lg:grid-cols-3">
        <div>
          <dt className="text-muted-foreground">Independent seed pairs (N)</dt>
          <dd>{comparison.n}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Training seeds</dt>
          <dd>{comparison.seedSet.join(", ") || "None"}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Statistical unit</dt>
          <dd>{hierarchical ? "Paired samples within training seeds" : "Training seeds"}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Method</dt>
          <dd>
            {hierarchical ? "Hierarchical bootstrap" : "Run-seed bootstrap"} v{estimator.version} ·{" "}
            {STATISTIC_LABELS[estimator.statistic]}
          </dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Baseline mean</dt>
          <dd>{metricValue(comparison.baselineMean)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Candidate mean</dt>
          <dd>{metricValue(comparison.candidateMean)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Candidate − baseline</dt>
          <dd>{metricValue(comparison.pairedDelta)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Seed-difference standard deviation</dt>
          <dd>{metricValue(comparison.dispersion)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">
            {(estimator.confidenceLevel * 100).toLocaleString(undefined, {
              maximumFractionDigits: 2,
            })}
            % confidence interval
          </dt>
          <dd>
            {hasInterval && comparison.interval !== null
              ? `${metricValue(comparison.interval[0])} to ${metricValue(comparison.interval[1])}`
              : "Unavailable"}
          </dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Resampling</dt>
          <dd>
            {estimator.resampleCount.toLocaleString()} resamples · seed {estimator.resamplingSeed}
          </dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Missing-pair policy</dt>
          <dd>
            {estimator.missingPairPolicy === "exclude"
              ? "Exclude missing pairs"
              : "Require every pair"}
          </dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Evidence gaps</dt>
          <dd>
            {comparison.unmatchedRuns} unmatched runs · {comparison.failedRuns} failed runs ·{" "}
            {comparison.unmatchedSamples} unmatched samples
          </dd>
        </div>
      </dl>
      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground">
          Evaluation protocol SHA-256
        </summary>
        <p className="mt-1 break-all font-mono">{comparison.protocolSha256}</p>
      </details>
      {comparison.excludedRuns.length > 0 ? (
        <details className="text-xs">
          <summary className="cursor-pointer">
            Excluded runs ({comparison.excludedRuns.length})
          </summary>
          <ul className="mt-2 max-h-48 space-y-1 overflow-auto">
            {comparison.excludedRuns.map((run) => (
              <li
                key={`${run.variantLabel}:${run.trainingSeed}:${run.runId ?? "missing"}`}
                className="break-words"
              >
                {run.variantLabel} · seed {run.trainingSeed} · {run.runId ?? "No run ID"}:{" "}
                {EXCLUSION_LABELS[run.reason]}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}
