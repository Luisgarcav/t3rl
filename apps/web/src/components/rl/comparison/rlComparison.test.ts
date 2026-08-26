import type {
  RlMetricBatch,
  RlResolvedManifest,
  RlRunState,
  RlRunSummary,
} from "@t3tools/contracts";
import { describe, expect, it } from "vite-plus/test";

import {
  analyzeRlComparison,
  compatibleRunsForExperiment,
  sampleAggregateSeries,
  selectRunMetricObservations,
  type RlComparisonRun,
} from "./rlComparison";

function summary(
  runId: string,
  experimentId = "cartpole",
  requestedAt = "2026-01-01T00:00:00.000Z",
  state: RlRunState = "completed",
): RlRunSummary {
  return {
    runId,
    projectId: "project-1",
    experimentId,
    state,
    requestedAt,
    startedAt: requestedAt,
    endedAt: requestedAt,
    lastMessageAt: requestedAt,
    errorCode: null,
    errorMessage: null,
  };
}

function manifest(seed: number, overrides: Partial<RlResolvedManifest> = {}): RlResolvedManifest {
  return {
    experimentId: "cartpole",
    runnerId: "python",
    runnerVersion: "1.0.0",
    protocolVersion: 1,
    seed,
    effectiveConfig: { learningRate: 0.001 },
    sourceRevision: "abc123",
    sourceDirty: false,
    pythonExecutable: "/usr/bin/python3",
    pythonVersion: "3.12",
    environmentFingerprint: "env-1",
    instrumentationLevel: "standard",
    hardwareSummary: "cpu",
    ...overrides,
  };
}

function batch(
  step: number,
  value: RlMetricBatch["values"][string] | undefined,
  wallClockMs = step,
): RlMetricBatch {
  return {
    step,
    wallClockMs,
    values: value === undefined ? { other: 1 } : { "train/return": value },
  };
}

function run(
  runId: string,
  seed: number,
  metrics: ReadonlyArray<RlMetricBatch>,
  options: {
    readonly requestedAt?: string;
    readonly manifest?: RlResolvedManifest | null;
    readonly state?: RlRunState;
  } = {},
): RlComparisonRun {
  return {
    summary: summary(runId, "cartpole", options.requestedAt, options.state),
    manifest: options.manifest === undefined ? manifest(seed) : options.manifest,
    metrics,
  };
}

describe("RL multi-seed comparison", () => {
  it("only offers runs from the exact experiment and sorts newest first", () => {
    const result = compatibleRunsForExperiment(
      [
        summary("old", "cartpole", "2026-01-01T00:00:00.000Z"),
        summary("other", "lunar-lander", "2026-01-03T00:00:00.000Z"),
        summary("new", "cartpole", "2026-01-02T00:00:00.000Z"),
      ],
      "cartpole",
    );

    expect(result.map((entry) => entry.runId)).toEqual(["new", "old"]);
  });

  it("aggregates finite values at exact steps without interpolation", () => {
    const analysis = analyzeRlComparison(
      [
        run("seed-1", 1, [batch(0, 1), batch(10, 3), batch(20, null)]),
        run("seed-2", 2, [batch(0, 3), batch(20, 9)]),
        run("seed-3", 3, [batch(0, 8), batch(10, "nan"), batch(20, 12)]),
        run("seed-4", 4, [batch(0, 10), batch(10, 7), batch(20, 15)]),
      ],
      "cartpole",
      "train/return",
    );

    expect(analysis.series).toEqual([
      {
        step: 0,
        count: 4,
        mean: 5.5,
        median: 5.5,
        minimum: 1,
        maximum: 10,
        firstQuartile: 2.5,
        thirdQuartile: 8.5,
      },
      {
        step: 10,
        count: 2,
        mean: 5,
        median: 5,
        minimum: 3,
        maximum: 7,
        firstQuartile: 4,
        thirdQuartile: 6,
      },
      {
        step: 20,
        count: 3,
        mean: 12,
        median: 12,
        minimum: 9,
        maximum: 15,
        firstQuartile: 10.5,
        thirdQuartile: 13.5,
      },
    ]);
    expect(analysis.warnings.map((entry) => entry.code)).toContain("partial-metric-coverage");
    expect(analysis.warnings.map((entry) => entry.code)).toContain("small-sample");
  });

  it("uses the latest explicit value per run and step, including non-finite replacement", () => {
    expect(
      selectRunMetricObservations(
        [batch(5, 1, 10), batch(5, 2, 20), batch(5, "nan", 30), batch(5, undefined, 40)],
        "train/return",
      ),
    ).toEqual([]);
  });

  it("counts a seed once and chooses its most recently requested run", () => {
    const analysis = analyzeRlComparison(
      [
        run("old", 7, [batch(0, 100)], { requestedAt: "2026-01-01T00:00:00.000Z" }),
        run("new", 7, [batch(0, 2)], { requestedAt: "2026-01-02T00:00:00.000Z" }),
        run("seed-8", 8, [batch(0, 4)]),
      ],
      "cartpole",
      "train/return",
    );

    expect(analysis.seeds).toHaveLength(2);
    expect(analysis.seeds[0]?.representative.summary.runId).toBe("new");
    expect(analysis.seeds[0]?.runCount).toBe(2);
    expect(analysis.series[0]).toMatchObject({ count: 2, mean: 3, median: 3 });
    expect(analysis.warnings.map((entry) => entry.code)).toContain("duplicate-seed");
  });

  it("excludes unverifiable manifests and reports drift and provisional data", () => {
    const analysis = analyzeRlComparison(
      [
        run("missing", 1, [batch(0, 1)], { manifest: null }),
        run("base", 2, [batch(0, 2)]),
        run("drift", 3, [batch(0, 4)], {
          manifest: manifest(3, {
            effectiveConfig: { learningRate: 0.01 },
            runnerVersion: "2.0.0",
            sourceRevision: null,
            sourceDirty: null,
            environmentFingerprint: "env-2",
          }),
          state: "running",
        }),
      ],
      "cartpole",
      "train/return",
    );

    expect(analysis.excludedRunIds).toEqual(["missing"]);
    expect(analysis.warnings.map((entry) => entry.code)).toEqual(
      expect.arrayContaining([
        "descriptive-only",
        "missing-manifest",
        "small-sample",
        "provisional-run",
        "config-drift",
        "runner-drift",
        "source-drift",
        "environment-drift",
      ]),
    );
  });

  it("does not synthesize zero when a metric has no finite observations", () => {
    const analysis = analyzeRlComparison(
      [run("seed-1", 1, [batch(0, null), batch(1, "+inf")])],
      "cartpole",
      "train/return",
    );

    expect(analysis.series).toEqual([]);
    expect(analysis.seeds[0]).toMatchObject({
      latestMetricStep: null,
      latestMetricValue: null,
    });
    expect(analysis.warnings.map((entry) => entry.code)).toContain("no-metric-observations");
  });

  it("downsamples display points while retaining the final exact-step point", () => {
    const series = Array.from({ length: 20 }, (_, step) => ({
      step,
      count: 3,
      mean: step,
      median: step,
      minimum: step,
      maximum: step,
      firstQuartile: step,
      thirdQuartile: step,
    }));

    const sampled = sampleAggregateSeries(series, 5);
    expect(sampled.length).toBeLessThanOrEqual(5);
    expect(sampled[0]?.step).toBe(0);
    expect(sampled.at(-1)?.step).toBe(19);
  });
});
