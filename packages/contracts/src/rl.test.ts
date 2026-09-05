import * as Schema from "effect/Schema";
import { describe, expect, it } from "vite-plus/test";

import { AssetResource } from "./assets.ts";
import { RlArtifactMetadata, RlMetricBatch, RlRunState, RlRunSummary } from "./rl.ts";
import { WS_METHODS } from "./rpc.ts";

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
  it("keeps legacy artifacts readable and validates verified identity", () => {
    const legacy = {
      artifactId: "artifact_legacy",
      kind: "summary",
      bytes: 10,
      contentType: "application/json",
      producedAt: "2026-08-24T00:00:00.000Z",
    };
    expect(decodes(RlArtifactMetadata, legacy)).toBe(true);
    expect(
      decodes(RlArtifactMetadata, {
        ...legacy,
        sha256: "a".repeat(64),
        logicalName: "summary.json",
        format: "json",
        state: "ready",
        checkpointStep: null,
        fileCount: 1,
      }),
    ).toBe(true);
    expect(decodes(RlArtifactMetadata, { ...legacy, sha256: "not-a-hash" })).toBe(false);
  });

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

describe("RL RPC surface", () => {
  it("declares the paginated run kernel methods", () => {
    expect(WS_METHODS.rlCapabilities).toBe("rl.capabilities");
    expect(WS_METHODS.rlListRuns).toBe("rl.listRuns");
    expect(WS_METHODS.rlGetRun).toBe("rl.getRun");
    expect(WS_METHODS.rlListArtifacts).toBe("rl.listArtifacts");
    expect(WS_METHODS.rlStartRun).toBe("rl.startRun");
    expect(WS_METHODS.rlResumeRun).toBe("rl.resumeRun");
    expect(WS_METHODS.rlWarmStartRun).toBe("rl.warmStartRun");
    expect(WS_METHODS.rlCancelRun).toBe("rl.cancelRun");
    expect(WS_METHODS.rlSubscribeRun).toBe("rl.subscribeRun");
  });

  it("accepts an rl-artifact asset resource", () => {
    expect(
      decodes(AssetResource, { _tag: "rl-artifact", runId: "run_01", artifactId: "art_01" }),
    ).toBe(true);
  });
});
