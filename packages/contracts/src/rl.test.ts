import * as Schema from "effect/Schema";
import { describe, expect, it } from "vite-plus/test";

import { RlArtifactMetadata, RlMetricBatch, RlRunState, RlRunSummary } from "./rl.ts";

function decodes<S extends Schema.Top>(schema: S, input: unknown): boolean {
  try {
    Schema.decodeUnknownSync(schema as never)(input);
    return true;
  } catch {
    return false;
  }
}

describe("RlMetricBatch", () => {
  it("accepts finite values, nulls, and non-finite markers", () => {
    expect(
      decodes(RlMetricBatch, {
        step: 10,
        wallClockMs: 1234,
        values: { "train/return": 21.5, "train/kl": null, "train/loss": "nan" },
      }),
    ).toBe(true);
  });

  it("rejects a negative step", () => {
    expect(decodes(RlMetricBatch, { step: -1, wallClockMs: 0, values: {} })).toBe(false);
  });

  it("rejects an unbounded metric key set", () => {
    const values: Record<string, number> = {};
    for (let index = 0; index < 65; index += 1) values[`k${index}`] = index;
    expect(decodes(RlMetricBatch, { step: 0, wallClockMs: 0, values })).toBe(false);
  });

  it("rejects a metric key with path separators", () => {
    expect(decodes(RlMetricBatch, { step: 0, wallClockMs: 0, values: { "../escape": 1 } })).toBe(
      false,
    );
  });

  it("rejects an over-long metric key", () => {
    expect(
      decodes(RlMetricBatch, { step: 0, wallClockMs: 0, values: { ["k".repeat(65)]: 1 } }),
    ).toBe(false);
  });

  it("accepts a namespaced metric key", () => {
    expect(
      decodes(RlMetricBatch, { step: 0, wallClockMs: 0, values: { "train/policy/loss": 1 } }),
    ).toBe(true);
  });
});

describe("RlRunState", () => {
  it("accepts every documented state", () => {
    for (const state of [
      "requested",
      "preparing",
      "running",
      "cancelling",
      "completed",
      "failed",
      "cancelled",
      "interrupted",
    ]) {
      expect(decodes(RlRunState, state)).toBe(true);
    }
  });

  it("rejects an undocumented state", () => {
    expect(decodes(RlRunState, "paused")).toBe(false);
  });
});

describe("RlArtifactMetadata", () => {
  it("rejects an artifact id that looks like a path", () => {
    expect(
      decodes(RlArtifactMetadata, {
        artifactId: "../../etc/passwd",
        kind: "log",
        bytes: 10,
        contentType: "text/plain",
        producedAt: "2026-08-24T00:00:00.000Z",
      }),
    ).toBe(false);
  });
});

describe("RlRunSummary", () => {
  it("round trips a completed run", () => {
    const input = {
      runId: "run_01",
      projectId: "proj_01",
      experimentId: "fake",
      state: "completed",
      requestedAt: "2026-08-24T00:00:00.000Z",
      startedAt: "2026-08-24T00:00:01.000Z",
      endedAt: "2026-08-24T00:00:09.000Z",
      lastMessageAt: "2026-08-24T00:00:08.000Z",
      errorCode: null,
      errorMessage: null,
    };
    expect(decodes(RlRunSummary, input)).toBe(true);
  });
});
