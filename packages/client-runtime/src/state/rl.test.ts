import {
  type RlArtifactMetadata,
  type RlMetricBatch,
  type RlRunSummary,
  RL_MAX_RUN_ARTIFACTS,
  RL_MAX_SNAPSHOT_METRIC_BATCHES,
} from "@t3tools/contracts";
import { describe, expect, it } from "vite-plus/test";

import { applyRlSubscriptionEvent, type RlRunProjection } from "./rl.ts";

const summary: RlRunSummary = {
  runId: "run_1",
  projectId: "project_1",
  experimentId: "cartpole_ppo",
  state: "running",
  requestedAt: "2026-08-25T12:00:00.000Z",
  startedAt: "2026-08-25T12:00:01.000Z",
  endedAt: null,
  lastMessageAt: "2026-08-25T12:00:02.000Z",
  errorCode: null,
  errorMessage: null,
};

const metric = (step: number, wallClockMs = step): RlMetricBatch => ({
  step,
  wallClockMs,
  values: { "train/return": step },
});

const artifact = (index: number): RlArtifactMetadata => ({
  artifactId: `artifact_${index}`,
  kind: "log",
  bytes: index,
  contentType: "text/plain",
  producedAt: "2026-08-25T12:00:02.000Z",
});

const projection = (): RlRunProjection => ({
  summary,
  manifest: null,
  artifacts: [],
  metrics: [],
});

describe("applyRlSubscriptionEvent", () => {
  it("replaces stale state with the reconnect snapshot", () => {
    const result = applyRlSubscriptionEvent(
      { ...projection(), artifacts: [artifact(99)], metrics: [metric(99)] },
      {
        _tag: "Snapshot",
        summary,
        manifest: null,
        artifacts: [artifact(1)],
        metrics: [metric(1)],
      },
    );

    expect(result?.artifacts.map((entry) => entry.artifactId)).toEqual(["artifact_1"]);
    expect(result?.metrics.map((entry) => entry.step)).toEqual([1]);
  });

  it("accumulates live facts and replaces duplicate identities", () => {
    const withArtifact = applyRlSubscriptionEvent(projection(), {
      _tag: "Artifact",
      artifact: artifact(1),
    });
    const replacedArtifact = applyRlSubscriptionEvent(withArtifact, {
      _tag: "Artifact",
      artifact: { ...artifact(1), bytes: 42 },
    });
    const withMetric = applyRlSubscriptionEvent(replacedArtifact, {
      _tag: "Metrics",
      batch: metric(1),
    });
    const replacedMetric = applyRlSubscriptionEvent(withMetric, {
      _tag: "Metrics",
      batch: { ...metric(1), values: { "train/return": 42 } },
    });

    expect(replacedMetric?.artifacts).toHaveLength(1);
    expect(replacedMetric?.artifacts[0]?.bytes).toBe(42);
    expect(replacedMetric?.metrics).toHaveLength(1);
    expect(replacedMetric?.metrics[0]?.values["train/return"]).toBe(42);
  });

  it("keeps live collections within the wire snapshot bounds", () => {
    let current: RlRunProjection | null = projection();
    for (let index = 0; index <= RL_MAX_RUN_ARTIFACTS; index += 1) {
      current = applyRlSubscriptionEvent(current, {
        _tag: "Artifact",
        artifact: artifact(index),
      });
    }
    for (let index = 0; index <= RL_MAX_SNAPSHOT_METRIC_BATCHES; index += 1) {
      current = applyRlSubscriptionEvent(current, { _tag: "Metrics", batch: metric(index) });
    }

    expect(current?.artifacts).toHaveLength(RL_MAX_RUN_ARTIFACTS);
    expect(current?.artifacts[0]?.artifactId).toBe("artifact_1");
    expect(current?.metrics).toHaveLength(RL_MAX_SNAPSHOT_METRIC_BATCHES);
    expect(current?.metrics[0]?.step).toBe(1);
  });
});
