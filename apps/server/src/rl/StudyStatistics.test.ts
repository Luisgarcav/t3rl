import { assert, it } from "@effect/vitest";
import { comparePairedStudy } from "./StudyStatistics.ts";

const estimator = {
  version: 1 as const,
  statistic: "paired-mean-delta" as const,
  statisticalUnit: "paired-sample-within-run-seed" as const,
  confidenceLevel: 0.95,
  resamplingSeed: 17,
  resampleCount: 1_000,
  missingPairPolicy: "exclude" as const,
};

it("is deterministic under arrival reordering and preserves known paired deltas", () => {
  const baseline = [
    { trainingSeed: 2, value: 20, samples: { a: 10, b: 30 } },
    { trainingSeed: 1, value: 10, samples: { a: 5, b: 15 } },
  ];
  const candidate = [
    { trainingSeed: 1, value: 13, samples: { a: 8, b: 18 } },
    { trainingSeed: 2, value: 23, samples: { a: 13, b: 33 } },
  ];
  const compare = (left = baseline, right = candidate) =>
    comparePairedStudy({
      studyId: "study-1",
      protocolSha256: "a".repeat(64),
      baselineLabel: "a",
      candidateLabel: "b",
      metricKey: "eval/reward",
      baseline: left,
      candidate: right,
      failedRuns: 0,
      estimator,
      compatibleProtocol: true,
    });
  assert.equal(compare().pairedDelta, 3);
  assert.deepEqual(compare(), compare(baseline.toReversed(), candidate.toReversed()));
});

it("does not manufacture an interval from one paired seed", () => {
  const result = comparePairedStudy({
    studyId: "study-1",
    protocolSha256: "a".repeat(64),
    baselineLabel: "a",
    candidateLabel: "b",
    metricKey: "eval/reward",
    baseline: [{ trainingSeed: 1, value: 2 }],
    candidate: [{ trainingSeed: 1, value: 3 }],
    failedRuns: 0,
    estimator,
    compatibleProtocol: true,
  });
  assert.equal(result.conclusion, "not-enough-evidence");
  assert.equal(result.interval, null);
});
