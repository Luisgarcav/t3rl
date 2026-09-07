import type { RlStudyComparison } from "@t3tools/contracts";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vite-plus/test";

import { RlStudyComparisonResult } from "./RlStudyComparisonResult";

const comparison = {
  studyId: "study_reference",
  protocolSha256: "a".repeat(64),
  baselineLabel: "baseline",
  candidateLabel: "candidate",
  metricKey: "eval_after/loss",
  estimator: {
    version: 2,
    statistic: "paired-mean-delta",
    statisticalUnit: "run-seed",
    confidenceLevel: 0.95,
    resamplingSeed: 17,
    resampleCount: 1000,
    missingPairPolicy: "exclude",
  },
  seedSet: [7, 19, 41],
  n: 3,
  unmatchedRuns: 1,
  failedRuns: 1,
  unmatchedSamples: 2,
  excludedRuns: [
    {
      runId: "run_missing",
      trainingSeed: 99,
      variantLabel: "candidate",
      reason: "missing-evaluation",
    },
  ],
  baselineMean: 2,
  candidateMean: 1,
  pairedDelta: -1,
  dispersion: 0.25,
  interval: [-1.5, -0.5],
  conclusion: "interval",
} satisfies RlStudyComparison;

describe("RlStudyComparisonResult", () => {
  it("labels seed-only evidence honestly and retains result identity and exclusions", () => {
    const html = renderToStaticMarkup(<RlStudyComparisonResult comparison={comparison} />);

    expect(html).toContain("Run-seed bootstrap");
    expect(html).not.toContain("Hierarchical bootstrap");
    expect(html).toContain("study_reference");
    expect(html).toContain("candidate");
    expect(html).toContain("eval_after/loss");
    expect(html).toContain("Independent seed pairs (N)");
    expect(html).toContain("7, 19, 41");
    expect(html).toContain("95% confidence interval");
    expect(html).toContain("-1.5 to -0.5");
    expect(html).toContain("unmatched samples");
    expect(html).toContain("run_missing");
    expect(html).toContain("Verified evaluation samples are unavailable");
    expect(html).toContain(comparison.protocolSha256);
  });

  it("labels hierarchical resampling only for paired sample evidence", () => {
    const html = renderToStaticMarkup(
      <RlStudyComparisonResult
        comparison={{
          ...comparison,
          estimator: {
            ...comparison.estimator,
            statisticalUnit: "paired-sample-within-run-seed",
            confidenceLevel: 0.575,
          },
        }}
      />,
    );
    expect(html).toContain("Hierarchical bootstrap");
    expect(html).toContain("Paired samples within training seeds");
    expect(html).not.toContain("Run-seed bootstrap");
    expect(html).toContain("57.5% confidence interval");
  });

  it.each([
    ["not-enough-evidence", "Not enough evidence"],
    ["incompatible-protocol", "Incompatible evaluation evidence"],
    ["missing-pairs", "Required pairs are missing"],
    ["unsupported-estimator", "Estimator is unsupported"],
    ["computation-budget-exceeded", "Comparison exceeds the computation budget"],
    ["invalid-variants", "Choose distinct study variants"],
  ] as const)("explains %s without presenting a confidence interval", (conclusion, label) => {
    const html = renderToStaticMarkup(
      <RlStudyComparisonResult
        comparison={{
          ...comparison,
          conclusion,
          baselineMean: null,
          candidateMean: null,
          pairedDelta: null,
          dispersion: null,
          interval: null,
        }}
      />,
    );
    expect(html).toContain(label);
    expect(html).toContain("Unavailable");
    expect(html).not.toContain("Confidence interval available");
    expect(html).not.toContain("-1.5 to -0.5");
    expect(html).toContain(">—</dd>");
  });
});
