/** Measures the server's cooperative estimator; excludes artifact I/O, SQLite, transport, and UI. */
import * as NodePerfHooks from "node:perf_hooks";
import * as Effect from "effect/Effect";
import {
  comparePairedStudyEffect,
  MAX_STUDY_RESAMPLING_DRAWS,
} from "../apps/server/src/rl/StudyStatistics.ts";

// Regression guard for this benchmark host; not a latency guarantee on arbitrary hardware.
const MAX_HEARTBEAT_GAP_MS = 50;

const makeInput = (seedCount: number, sampleCount: number, resampleCount: number) => {
  const sampleIds = Array.from({ length: sampleCount }, (_, index) => `sample_${index}`);
  const observations = (candidate: boolean) =>
    Array.from({ length: seedCount }, (_, seed) => ({
      trainingSeed: seed,
      value: 0,
      samples: Object.fromEntries(
        sampleIds.map((id, index) => [
          id,
          (index % 7) + (candidate ? (index % 3) - 1 + seed / 100 : 0),
        ]),
      ),
    }));
  return {
    studyId: "study_benchmark",
    protocolSha256: "a".repeat(64),
    baselineLabel: "baseline",
    candidateLabel: "candidate",
    metricKey: "eval_after/loss",
    baseline: observations(false),
    candidate: observations(true),
    failedRuns: 0,
    compatibleProtocol: true,
    expectedSampleIds: sampleIds,
    estimator: {
      version: 2 as const,
      statistic: "paired-mean-delta" as const,
      statisticalUnit: "paired-sample-within-run-seed" as const,
      confidenceLevel: 0.95,
      resamplingSeed: 17,
      resampleCount,
      missingPairPolicy: "exclude" as const,
    },
  };
};

for (const scenario of [
  { name: "within-budget", seeds: 32, samples: 512, resamples: 1_000 },
  { name: "over-budget", seeds: 64, samples: 10_000, resamples: 100_000 },
]) {
  const input = makeInput(scenario.seeds, scenario.samples, scenario.resamples);
  let heartbeats = 0;
  let maximumHeartbeatGapMs = 0;
  const started = NodePerfHooks.performance.now();
  let previous = started;
  let heartbeat: NodeJS.Immediate;
  const tick = () => {
    const now = NodePerfHooks.performance.now();
    maximumHeartbeatGapMs = Math.max(maximumHeartbeatGapMs, now - previous);
    previous = now;
    heartbeats += 1;
    heartbeat = setImmediate(tick);
  };
  heartbeat = setImmediate(tick);
  let elapsedMs = 0;
  const result = await Effect.runPromise(comparePairedStudyEffect(input)).finally(() => {
    const finished = NodePerfHooks.performance.now();
    elapsedMs = finished - started;
    maximumHeartbeatGapMs = Math.max(maximumHeartbeatGapMs, finished - previous);
    clearImmediate(heartbeat);
  });
  console.log(
    JSON.stringify({
      ...scenario,
      conclusion: result.conclusion,
      elapsedMs,
      heartbeats,
      maximumHeartbeatGapMs,
      heartbeatGapLimitMs: MAX_HEARTBEAT_GAP_MS,
      drawBudget: MAX_STUDY_RESAMPLING_DRAWS,
      estimatorVersion: result.estimator.version,
    }),
  );
  if (
    result.conclusion !==
    (scenario.name === "within-budget" ? "interval" : "computation-budget-exceeded")
  ) {
    throw new Error(`Unexpected conclusion: ${result.conclusion}`);
  }
  if (heartbeats === 0) throw new Error("Estimator did not yield to the event loop");
  if (maximumHeartbeatGapMs > MAX_HEARTBEAT_GAP_MS) {
    throw new Error(
      `Estimator heartbeat gap ${maximumHeartbeatGapMs.toFixed(2)} ms exceeded ${MAX_HEARTBEAT_GAP_MS} ms`,
    );
  }
}
