import {
  type RlArtifactMetadata,
  type RlMetricBatch,
  type RlResolvedManifest,
  type RlRunLineage,
  type RlRunSummary,
  type RlSubscriptionEvent,
  RL_MAX_SNAPSHOT_ARTIFACTS,
  RL_MAX_SNAPSHOT_METRIC_BATCHES,
  WS_METHODS,
} from "@t3tools/contracts";
import * as Stream from "effect/Stream";
import type { Atom } from "effect/unstable/reactivity";

import type { EnvironmentRegistry } from "../connection/registry.ts";
import {
  createEnvironmentRpcCommand,
  createEnvironmentRpcQueryAtomFamily,
  createEnvironmentRpcSubscriptionAtomFamily,
} from "./runtime.ts";

export interface RlRunProjection {
  readonly summary: RlRunSummary;
  readonly manifest: RlResolvedManifest | null;
  readonly lineage: RlRunLineage;
  readonly artifacts: ReadonlyArray<RlArtifactMetadata>;
  readonly metrics: ReadonlyArray<RlMetricBatch>;
}

function appendArtifact(
  artifacts: ReadonlyArray<RlArtifactMetadata>,
  artifact: RlArtifactMetadata,
): ReadonlyArray<RlArtifactMetadata> {
  const next = artifacts
    .filter((entry) => entry.artifactId !== artifact.artifactId)
    .concat(artifact);
  return next.slice(-RL_MAX_SNAPSHOT_ARTIFACTS);
}

function appendMetricBatch(
  metrics: ReadonlyArray<RlMetricBatch>,
  batch: RlMetricBatch,
): ReadonlyArray<RlMetricBatch> {
  const next = metrics
    .filter((entry) => entry.step !== batch.step || entry.wallClockMs !== batch.wallClockMs)
    .concat(batch);
  return next.slice(-RL_MAX_SNAPSHOT_METRIC_BATCHES);
}

/** Folds the reconnect snapshot and subsequent live facts into one renderable run. */
export function applyRlSubscriptionEvent(
  current: RlRunProjection | null,
  event: RlSubscriptionEvent,
): RlRunProjection | null {
  switch (event._tag) {
    case "Snapshot":
      return {
        summary: event.summary,
        manifest: event.manifest,
        lineage: event.lineage,
        artifacts: event.artifacts,
        metrics: event.metrics,
      };
    case "Lifecycle":
      return current === null
        ? {
            summary: event.summary,
            manifest: null,
            lineage: { edges: [], truncated: false },
            artifacts: [],
            metrics: [],
          }
        : { ...current, summary: event.summary };
    case "Manifest":
      return current === null ? null : { ...current, manifest: event.manifest };
    case "Artifact":
      return current === null
        ? null
        : { ...current, artifacts: appendArtifact(current.artifacts, event.artifact) };
    case "Metrics":
      return current === null
        ? null
        : { ...current, metrics: appendMetricBatch(current.metrics, event.batch) };
  }
}

export function createRlEnvironmentAtoms<R, E>(
  runtime: Atom.AtomRuntime<EnvironmentRegistry | R, E>,
) {
  return {
    capabilities: createEnvironmentRpcQueryAtomFamily(runtime, {
      label: "environment-data:rl:capabilities",
      tag: WS_METHODS.rlCapabilities,
      staleTimeMs: 30_000,
    }),
    runs: createEnvironmentRpcQueryAtomFamily(runtime, {
      label: "environment-data:rl:runs",
      tag: WS_METHODS.rlListRuns,
      staleTimeMs: 5_000,
    }),
    detail: createEnvironmentRpcQueryAtomFamily(runtime, {
      label: "environment-data:rl:detail",
      tag: WS_METHODS.rlGetRun,
      staleTimeMs: 5_000,
    }),
    artifacts: createEnvironmentRpcQueryAtomFamily(runtime, {
      label: "environment-data:rl:artifacts",
      tag: WS_METHODS.rlListArtifacts,
      staleTimeMs: 5_000,
    }),
    run: createEnvironmentRpcSubscriptionAtomFamily(runtime, {
      label: "environment-data:rl:run",
      tag: WS_METHODS.rlSubscribeRun,
      transform: (stream) =>
        stream.pipe(
          Stream.scan(null, applyRlSubscriptionEvent),
          Stream.filter((projection): projection is RlRunProjection => projection !== null),
        ),
    }),
    start: createEnvironmentRpcCommand(runtime, {
      label: "environment-data:rl:start",
      tag: WS_METHODS.rlStartRun,
    }),
    resume: createEnvironmentRpcCommand(runtime, {
      label: "environment-data:rl:resume",
      tag: WS_METHODS.rlResumeRun,
    }),
    warmStart: createEnvironmentRpcCommand(runtime, {
      label: "environment-data:rl:warm-start",
      tag: WS_METHODS.rlWarmStartRun,
    }),
    cancel: createEnvironmentRpcCommand(runtime, {
      label: "environment-data:rl:cancel",
      tag: WS_METHODS.rlCancelRun,
    }),
    createStudy: createEnvironmentRpcCommand(runtime, {
      label: "environment-data:rl:create-study",
      tag: WS_METHODS.rlCreateStudy,
    }),
    study: createEnvironmentRpcQueryAtomFamily(runtime, {
      label: "environment-data:rl:study",
      tag: WS_METHODS.rlGetStudy,
      staleTimeMs: 2_000,
    }),
    compareStudy: createEnvironmentRpcCommand(runtime, {
      label: "environment-data:rl:compare-study",
      tag: WS_METHODS.rlCompareStudy,
    }),
    validateExperiment: createEnvironmentRpcCommand(runtime, {
      label: "environment-data:rl:validate-experiment",
      tag: WS_METHODS.rlValidateExperiment,
    }),
  };
}
