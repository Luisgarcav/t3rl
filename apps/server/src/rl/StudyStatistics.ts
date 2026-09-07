import type {
  RlEvaluationProtocol,
  RlStudyComparison,
  RlStudyEstimator,
  RlStudyExcludedRun,
} from "@t3tools/contracts";
import * as Effect from "effect/Effect";

export interface StudyObservation {
  readonly trainingSeed: number;
  readonly value: number;
  readonly samples?: Readonly<Record<string, number>> | undefined;
  readonly generationSeeds?: Readonly<Record<string, number | null>> | undefined;
}

export interface StudyComparisonInput {
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
  readonly expectedSeeds?: ReadonlyArray<number> | undefined;
  readonly expectedSampleIds?: ReadonlyArray<string> | undefined;
  readonly excludedRuns?: ReadonlyArray<RlStudyExcludedRun> | undefined;
  readonly generationSeedPolicy?: RlEvaluationProtocol["generationSeedPolicy"] | undefined;
}

/** Requests above this budget must reduce samples or resamples explicitly. */
export const MAX_STUDY_RESAMPLING_DRAWS = 20_000_000;
const DRAWS_PER_YIELD = 16_384;

const mean = (values: ReadonlyArray<number>) =>
  values.reduce((sum, value) => sum + value / values.length, 0);

const standardDeviation = (values: ReadonlyArray<number>) => {
  const center = mean(values);
  return Math.sqrt(
    values.reduce((sum, value) => sum + (value - center) ** 2 / (values.length - 1), 0),
  );
};

const quantile = (sorted: ReadonlyArray<number>, fraction: number) => {
  const position = (sorted.length - 1) * fraction;
  const lower = sorted[Math.floor(position)]!;
  const upper = sorted[Math.ceil(position)]!;
  return lower * (1 - (position % 1)) + upper * (position % 1);
};

/** Estimator v2 pins this PRNG and sorts seeds and sample IDs before drawing. */
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

/** The generator bounds each synchronous slice without changing the random draw order. */
export function* studyComparisonSteps(
  input: StudyComparisonInput,
): Generator<void, RlStudyComparison> {
  const baselineBySeed = new Map(input.baseline.map((entry) => [entry.trainingSeed, entry]));
  const candidateBySeed = new Map(input.candidate.map((entry) => [entry.trainingSeed, entry]));
  const duplicateSeed =
    baselineBySeed.size !== input.baseline.length ||
    candidateBySeed.size !== input.candidate.length;
  const expectedSeeds = [
    ...new Set(input.expectedSeeds ?? [...baselineBySeed.keys(), ...candidateBySeed.keys()]),
  ].sort((left, right) => left - right);
  const expectedSampleIds =
    input.expectedSampleIds === undefined
      ? undefined
      : [...new Set(input.expectedSampleIds)].sort();
  let unmatchedSamples = 0;
  let generationMismatch = false;
  const fixedGenerationSeeds = new Map<string, number | null>();
  let preparationUntilYield = DRAWS_PER_YIELD;
  const pairs: { seed: number; baseline: number; candidate: number; deltas: number[] }[] = [];
  for (const seed of expectedSeeds) {
    const baseline = baselineBySeed.get(seed);
    const candidate = candidateBySeed.get(seed);
    if (baseline === undefined || candidate === undefined) continue;
    if (baseline.samples === undefined || candidate.samples === undefined) {
      if (input.estimator.statisticalUnit !== "run-seed") continue;
      if (!Number.isFinite(baseline.value) || !Number.isFinite(candidate.value)) continue;
      pairs.push({ seed, baseline: baseline.value, candidate: candidate.value, deltas: [] });
      continue;
    }
    const sampleIds =
      expectedSampleIds ??
      [...new Set([...Object.keys(baseline.samples), ...Object.keys(candidate.samples)])].sort();
    const baselineValues: number[] = [];
    const candidateValues: number[] = [];
    const deltas: number[] = [];
    for (const id of sampleIds) {
      if (--preparationUntilYield <= 0) {
        preparationUntilYield = DRAWS_PER_YIELD;
        yield;
      }
      const left = baseline.samples[id];
      const right = candidate.samples[id];
      if (
        left === undefined ||
        right === undefined ||
        !Number.isFinite(left) ||
        !Number.isFinite(right)
      ) {
        unmatchedSamples += 1;
        continue;
      }
      if (baseline.generationSeeds?.[id] !== candidate.generationSeeds?.[id]) {
        generationMismatch = true;
        continue;
      }
      const generationSeed = baseline.generationSeeds?.[id];
      if (input.generationSeedPolicy === "fixed-per-sample" && generationSeed !== undefined) {
        if (fixedGenerationSeeds.has(id) && fixedGenerationSeeds.get(id) !== generationSeed) {
          generationMismatch = true;
          continue;
        }
        fixedGenerationSeeds.set(id, generationSeed);
      }
      baselineValues.push(left);
      candidateValues.push(right);
      deltas.push(right - left);
    }
    if (deltas.length > 0)
      pairs.push({
        seed,
        baseline: mean(baselineValues),
        candidate: mean(candidateValues),
        deltas,
      });
  }
  const base = {
    studyId: input.studyId,
    protocolSha256: input.protocolSha256,
    baselineLabel: input.baselineLabel,
    candidateLabel: input.candidateLabel,
    metricKey: input.metricKey,
    estimator: input.estimator,
    seedSet: pairs.map((pair) => pair.seed),
    n: pairs.length,
    unmatchedRuns: input.baseline.length + input.candidate.length - pairs.length * 2,
    failedRuns: input.failedRuns,
    unmatchedSamples,
    excludedRuns: input.excludedRuns ?? [],
  };
  const empty = (conclusion: RlStudyComparison["conclusion"]): RlStudyComparison => ({
    ...base,
    baselineMean: null,
    candidateMean: null,
    pairedDelta: null,
    dispersion: null,
    interval: null,
    conclusion,
  });
  if (input.baselineLabel === input.candidateLabel) return empty("invalid-variants");
  if (!input.compatibleProtocol || generationMismatch || duplicateSeed) {
    return empty("incompatible-protocol");
  }
  if (input.estimator.version !== 2 || input.estimator.statistic !== "paired-mean-delta") {
    return empty("unsupported-estimator");
  }
  if (
    input.estimator.missingPairPolicy === "fail" &&
    (pairs.length !== expectedSeeds.length || unmatchedSamples > 0 || base.excludedRuns.length > 0)
  ) {
    return empty("missing-pairs");
  }
  const deltas = pairs.map((pair) => pair.candidate - pair.baseline);
  const result = {
    ...base,
    baselineMean: pairs.length === 0 ? null : mean(pairs.map((pair) => pair.baseline)),
    candidateMean: pairs.length === 0 ? null : mean(pairs.map((pair) => pair.candidate)),
    pairedDelta: pairs.length === 0 ? null : mean(deltas),
    dispersion: pairs.length < 2 ? null : standardDeviation(deltas),
    interval: null,
    conclusion: "not-enough-evidence",
  } satisfies RlStudyComparison;
  if (
    [result.baselineMean, result.candidateMean, result.pairedDelta, result.dispersion].some(
      (value) => value !== null && !Number.isFinite(value),
    )
  ) {
    return empty("not-enough-evidence");
  }
  if (pairs.length < 2) return result;

  const hierarchical = input.estimator.statisticalUnit === "paired-sample-within-run-seed";
  const maxSamples = Math.max(...pairs.map((pair) => pair.deltas.length));
  const worstCaseDraws =
    input.estimator.resampleCount * pairs.length * (1 + (hierarchical ? maxSamples : 0));
  if (worstCaseDraws > MAX_STUDY_RESAMPLING_DRAWS) {
    return { ...result, conclusion: "computation-budget-exceeded" };
  }

  const nextRandom = random(input.estimator.resamplingSeed);
  const bootstraps: number[] = [];
  let drawsUntilYield = DRAWS_PER_YIELD;
  for (let iteration = 0; iteration < input.estimator.resampleCount; iteration += 1) {
    let total = 0;
    for (let selected = 0; selected < pairs.length; selected += 1) {
      const pair = pairs[Math.floor(nextRandom() * pairs.length)]!;
      let delta = pair.candidate - pair.baseline;
      drawsUntilYield -= 1;
      if (hierarchical) {
        delta = 0;
        for (let sample = 0; sample < pair.deltas.length; sample += 1) {
          delta += pair.deltas[Math.floor(nextRandom() * pair.deltas.length)]! / pair.deltas.length;
          if (--drawsUntilYield <= 0) {
            drawsUntilYield = DRAWS_PER_YIELD;
            yield;
          }
        }
      }
      total += delta / pairs.length;
      if (drawsUntilYield <= 0) {
        drawsUntilYield = DRAWS_PER_YIELD;
        yield;
      }
    }
    bootstraps.push(total);
  }
  bootstraps.sort((left, right) => left - right);
  const tail = (1 - input.estimator.confidenceLevel) / 2;
  return {
    ...result,
    interval: [quantile(bootstraps, tail), quantile(bootstraps, 1 - tail)],
    conclusion: "interval",
  };
}

/** Synchronous entry point for small fixtures and offline calculations. */
export function comparePairedStudy(input: StudyComparisonInput): RlStudyComparison {
  const steps = studyComparisonSteps(input);
  let next = steps.next();
  while (!next.done) next = steps.next();
  return next.value;
}

/** Server requests yield between bounded slices so resampling does not monopolize the event loop. */
export const comparePairedStudyEffect = Effect.fn("StudyStatistics.compare")(function* (
  input: StudyComparisonInput,
) {
  const steps = studyComparisonSteps(input);
  let next = steps.next();
  while (!next.done) {
    yield* Effect.yieldNow;
    next = steps.next();
  }
  return next.value;
});
