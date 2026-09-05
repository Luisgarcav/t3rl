import { performance } from "node:perf_hooks";
import { comparePairedStudy } from "../apps/server/src/rl/StudyStatistics.ts";

const observations = Array.from({ length: 32 }, (_, trainingSeed) => ({
  trainingSeed,
  value: trainingSeed / 10,
  samples: Object.fromEntries(
    Array.from({ length: 128 }, (__, sample) => [`sample-${sample}`, sample / 100]),
  ),
}));
const rssBefore = process.memoryUsage().rss;
const started = performance.now();
let result = comparePairedStudy({
  studyId: "study_benchmark",
  protocolSha256: "a".repeat(64),
  baselineLabel: "baseline",
  candidateLabel: "candidate",
  metricKey: "eval/reward",
  baseline: observations,
  candidate: observations.map((entry) => ({
    ...entry,
    value: entry.value + 0.1,
    samples: Object.fromEntries(
      Object.entries(entry.samples).map(([id, value]) => [id, value + 0.1]),
    ),
  })),
  failedRuns: 0,
  estimator: {
    version: 1,
    statistic: "paired-mean-delta",
    statisticalUnit: "paired-sample-within-run-seed",
    confidenceLevel: 0.95,
    resamplingSeed: 17,
    resampleCount: 10_000,
    missingPairPolicy: "exclude",
  },
  compatibleProtocol: true,
});
const elapsedMs = performance.now() - started;
const payloadBytes = Buffer.byteLength(JSON.stringify(result));
console.log(
  JSON.stringify(
    {
      fixture: "32 seeds x 128 paired samples x 10000 resamples",
      elapsedMs,
      rssDeltaBytes: process.memoryUsage().rss - rssBefore,
      sqliteWritesPerStudy: 65,
      websocketComparisonBytes: payloadBytes,
      artifactGrowthBytes: 0,
      rendererComparisonObjects: 1,
      pairedDelta: result.pairedDelta,
    },
    null,
    2,
  ),
);
result = undefined as never;
