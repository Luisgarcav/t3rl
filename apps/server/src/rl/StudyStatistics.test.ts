import { assert, it } from "@effect/vitest";
import { comparePairedStudy } from "./StudyStatistics.ts";

const estimator = {
  version: 2 as const,
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

const observations = {
  studyId: "study-1",
  protocolSha256: "a".repeat(64),
  baselineLabel: "baseline",
  candidateLabel: "candidate",
  metricKey: "eval_after/loss",
  baseline: [
    { trainingSeed: 1, value: 999, samples: { a: 0, b: 0 } },
    { trainingSeed: 2, value: 999, samples: { a: 0, b: 0 } },
  ],
  candidate: [
    { trainingSeed: 1, value: -999, samples: { a: -5, b: 5 } },
    { trainingSeed: 2, value: -999, samples: { a: -5, b: 5 } },
  ],
  failedRuns: 0,
  estimator,
  compatibleProtocol: true,
};

it("resamples samples only when the requested statistical unit includes them", () => {
  const hierarchical = comparePairedStudy(observations);
  const runSeed = comparePairedStudy({
    ...observations,
    estimator: { ...estimator, statisticalUnit: "run-seed" },
  });
  assert.equal(hierarchical.pairedDelta, 0);
  assert.isBelow(hierarchical.interval![0], 0);
  assert.isAbove(hierarchical.interval![1], 0);
  assert.deepEqual(runSeed.interval, [0, 0]);
  assert.equal(runSeed.baselineMean, 0);
});

it("pins sample ordering as well as seed ordering", () => {
  const reversed = observations.candidate
    .toReversed()
    .map((entry) => ({ ...entry, samples: { b: entry.samples.b, a: entry.samples.a } }));
  assert.deepEqual(
    comparePairedStudy(observations),
    comparePairedStudy({ ...observations, candidate: reversed }),
  );
});

it("uses the same matched samples for point estimates and bootstrap draws", () => {
  const baseline = observations.baseline.map((entry) => ({ ...entry, samples: { a: 1, b: 999 } }));
  const candidate = observations.candidate.map((entry) => ({ ...entry, samples: { a: 3 } }));
  const result = comparePairedStudy({ ...observations, baseline, candidate });
  assert.equal(result.baselineMean, 1);
  assert.equal(result.candidateMean, 3);
  assert.equal(result.pairedDelta, 2);
  assert.deepEqual(result.interval, [2, 2]);
  assert.equal(result.unmatchedSamples, 2);
  const failed = comparePairedStudy({
    ...observations,
    baseline,
    candidate,
    estimator: { ...estimator, missingPairPolicy: "fail" },
  });
  assert.equal(failed.conclusion, "missing-pairs");
  assert.equal(failed.interval, null);
});

it("does not substitute run scalars when hierarchical sample evidence is absent", () => {
  const withoutSamples = (entries: typeof observations.baseline) =>
    entries.map(({ trainingSeed, value }) => ({ trainingSeed, value }));
  const result = comparePairedStudy({
    ...observations,
    baseline: withoutSamples(observations.baseline),
    candidate: withoutSamples(observations.candidate),
  });
  assert.equal(result.n, 0);
  assert.equal(result.pairedDelta, null);
  assert.equal(result.conclusion, "not-enough-evidence");
});

it("enforces fail policy even when both variants lack the same scheduled seed or sample", () => {
  for (const missing of [
    { expectedSeeds: [1, 2, 3] },
    { expectedSampleIds: ["a", "b", "missing"] },
  ]) {
    const result = comparePairedStudy({
      ...observations,
      ...missing,
      estimator: { ...estimator, missingPairPolicy: "fail" },
    });
    assert.equal(result.conclusion, "missing-pairs");
    assert.equal(result.interval, null);
  }
});

it("refuses different generation seeds and duplicate training seeds", () => {
  const mismatch = comparePairedStudy({
    ...observations,
    baseline: observations.baseline.map((entry) => ({ ...entry, generationSeeds: { a: 7, b: 8 } })),
    candidate: observations.candidate.map((entry) => ({
      ...entry,
      generationSeeds: { a: 7, b: 9 },
    })),
  });
  const duplicate = comparePairedStudy({
    ...observations,
    baseline: [...observations.baseline, observations.baseline[0]!],
  });
  for (const result of [mismatch, duplicate]) {
    assert.equal(result.conclusion, "incompatible-protocol");
    assert.equal(result.interval, null);
  }
});

it("refuses unsupported statistics and excessive work explicitly", () => {
  assert.equal(
    comparePairedStudy({ ...observations, estimator: { ...estimator, version: 1 } }).conclusion,
    "unsupported-estimator",
  );
  assert.equal(
    comparePairedStudy({ ...observations, estimator: { ...estimator, statistic: "median" } })
      .conclusion,
    "unsupported-estimator",
  );
  const samples = Object.fromEntries(
    Array.from({ length: 1_000 }, (_, index) => [String(index), index]),
  );
  const result = comparePairedStudy({
    ...observations,
    baseline: observations.baseline.map((entry) => ({ ...entry, samples })),
    candidate: observations.candidate.map((entry) => ({ ...entry, samples })),
    estimator: { ...estimator, resampleCount: 100_000 },
  });
  assert.equal(result.conclusion, "computation-budget-exceeded");
  assert.equal(result.interval, null);
});

it("requires fixed generation seeds to remain fixed across training seeds", () => {
  const withSeeds = (entries: typeof observations.baseline) =>
    entries.map((entry) => ({
      ...entry,
      generationSeeds: { a: entry.trainingSeed, b: entry.trainingSeed },
    }));
  const input = {
    ...observations,
    baseline: withSeeds(observations.baseline),
    candidate: withSeeds(observations.candidate),
  };
  assert.equal(
    comparePairedStudy({ ...input, generationSeedPolicy: "fixed-per-sample" }).conclusion,
    "incompatible-protocol",
  );
  assert.equal(
    comparePairedStudy({ ...input, generationSeedPolicy: "per-run" }).conclusion,
    "interval",
  );
});
