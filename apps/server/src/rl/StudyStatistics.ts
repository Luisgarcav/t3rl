import type { RlStudyComparison, RlStudyEstimator } from "@t3tools/contracts";

export interface StudyObservation {
  readonly trainingSeed: number;
  readonly value: number;
  readonly samples?: Readonly<Record<string, number>> | undefined;
}

const mean = (values: ReadonlyArray<number>) =>
  values.reduce((sum, value) => sum + value, 0) / values.length;

const standardDeviation = (values: ReadonlyArray<number>) => {
  if (values.length < 2) return 0;
  const center = mean(values);
  return Math.sqrt(
    values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1),
  );
};

const quantile = (values: ReadonlyArray<number>, fraction: number) => {
  const sorted = values.toSorted((left, right) => left - right);
  const position = (sorted.length - 1) * fraction;
  const lower = sorted[Math.floor(position)] ?? 0;
  const upper = sorted[Math.ceil(position)] ?? lower;
  return lower + (upper - lower) * (position - Math.floor(position));
};

/** Small reproducible PRNG: estimator versions pin this algorithm and its sampling order. */
const random = (seed: number) => {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
};

export function comparePairedStudy(input: {
  readonly studyId: string;
  readonly protocolSha256: string;
  readonly baselineLabel: string;
  readonly candidateLabel: string;
  readonly metricKey: string;
  readonly baseline: ReadonlyArray<StudyObservation>;
  readonly candidate: ReadonlyArray<StudyObservation>;
  readonly failedRuns: number;
  readonly estimator: RlStudyEstimator;
  readonly compatibleProtocol: boolean;
}): RlStudyComparison {
  const baselineBySeed = new Map(input.baseline.map((entry) => [entry.trainingSeed, entry]));
  const pairs = input.candidate
    .flatMap((candidate) => {
      const baseline = baselineBySeed.get(candidate.trainingSeed);
      return baseline === undefined ? [] : [{ baseline, candidate }];
    })
    .toSorted((left, right) => left.baseline.trainingSeed - right.baseline.trainingSeed);
  const unmatchedRuns = input.baseline.length + input.candidate.length - pairs.length * 2;
  const base = {
    studyId: input.studyId,
    protocolSha256: input.protocolSha256,
    baselineLabel: input.baselineLabel,
    candidateLabel: input.candidateLabel,
    metricKey: input.metricKey,
    estimator: input.estimator,
    seedSet: pairs.map((pair) => pair.baseline.trainingSeed).toSorted((a, b) => a - b),
    n: pairs.length,
    unmatchedRuns,
    failedRuns: input.failedRuns,
  } as const;
  if (!input.compatibleProtocol) {
    return {
      ...base,
      baselineMean: null,
      candidateMean: null,
      pairedDelta: null,
      dispersion: null,
      interval: null,
      conclusion: "incompatible-protocol",
    };
  }

  const deltas = pairs.map(({ baseline, candidate }) => candidate.value - baseline.value);
  const baselineMean = pairs.length === 0 ? null : mean(pairs.map((pair) => pair.baseline.value));
  const candidateMean = pairs.length === 0 ? null : mean(pairs.map((pair) => pair.candidate.value));
  if (pairs.length < 2) {
    return {
      ...base,
      baselineMean,
      candidateMean,
      pairedDelta: pairs.length === 0 ? null : mean(deltas),
      dispersion: pairs.length === 0 ? null : 0,
      interval: null,
      conclusion: "not-enough-evidence",
    };
  }

  const nextRandom = random(input.estimator.resamplingSeed);
  const bootstraps: number[] = [];
  for (let iteration = 0; iteration < input.estimator.resampleCount; iteration += 1) {
    const selected = Array.from({ length: pairs.length }, () => {
      const pair = pairs[Math.floor(nextRandom() * pairs.length)]!;
      const baselineSamples = pair.baseline.samples;
      const candidateSamples = pair.candidate.samples;
      if (baselineSamples === undefined || candidateSamples === undefined) {
        return pair.candidate.value - pair.baseline.value;
      }
      const sampleIds = Object.keys(baselineSamples).filter(
        (id) => candidateSamples[id] !== undefined,
      );
      if (sampleIds.length === 0) return pair.candidate.value - pair.baseline.value;
      const sampleDeltas = Array.from({ length: sampleIds.length }, () => {
        const id = sampleIds[Math.floor(nextRandom() * sampleIds.length)]!;
        return candidateSamples[id]! - baselineSamples[id]!;
      });
      return mean(sampleDeltas);
    });
    bootstraps.push(mean(selected));
  }
  const tail = (1 - input.estimator.confidenceLevel) / 2;
  return {
    ...base,
    baselineMean,
    candidateMean,
    pairedDelta: mean(deltas),
    dispersion: standardDeviation(deltas),
    interval: [quantile(bootstraps, tail), quantile(bootstraps, 1 - tail)],
    conclusion: "interval",
  };
}
