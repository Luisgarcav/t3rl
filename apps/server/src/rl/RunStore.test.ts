import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import * as SqlClient from "effect/unstable/sql/SqlClient";

import { SqlitePersistenceMemory } from "../persistence/Layers/Sqlite.ts";
import { RunStore, RunStoreLive } from "./RunStore.ts";

const testLayer = RunStoreLive.pipe(
  Layer.provideMerge(SqlitePersistenceMemory),
  Layer.provideMerge(NodeServices.layer),
);

it.effect("persists immutable studies and member state independently of arrival order", () =>
  Effect.gen(function* () {
    const store = yield* RunStore;
    const seeds = { training: 7, data: 8, evaluationSample: 9, generation: 10 };
    const protocolSha256 = "a".repeat(64);
    yield* store.createStudy({
      studyId: "study_test",
      projectId: "project-1",
      state: "requested",
      definition: {
        variants: [
          { label: "baseline", experimentId: "exp-a" },
          { label: "candidate", experimentId: "exp-b" },
        ],
        seeds: [seeds, { ...seeds, training: 11 }],
        maxConcurrency: 2,
        maxRuns: 4,
        evaluationProtocol: {
          version: 1,
          protocolSha256,
          datasetFingerprint: "dataset-v1",
          split: "test",
          sampleIds: ["sample-1"],
          generationSeedPolicy: "fixed-per-sample",
          decoding: {},
          verifierSha256: "b".repeat(64),
        },
      },
      protocolSha256,
      createdAt: "2026-09-04T00:00:00.000Z",
      updatedAt: "2026-09-04T00:00:00.000Z",
      runs: [
        { variantLabel: "candidate", seeds, runId: null, state: "queued" },
        { variantLabel: "baseline", seeds, runId: null, state: "queued" },
        {
          variantLabel: "candidate",
          seeds: { ...seeds, training: 11 },
          runId: null,
          state: "queued",
        },
        {
          variantLabel: "baseline",
          seeds: { ...seeds, training: 11 },
          runId: null,
          state: "queued",
        },
      ],
    });
    yield* store.updateStudyRun({
      studyId: "study_test",
      variantLabel: "baseline",
      trainingSeed: 7,
      runId: "run_1",
      state: "running",
      studyState: "running",
      at: "2026-09-04T00:01:00.000Z",
    });
    const stored = yield* store.getStudy({ studyId: "study_test" });
    assert.equal(stored.state, "running");
    assert.equal(
      stored.runs.find((run) => run.variantLabel === "baseline" && run.seeds.training === 7)?.runId,
      "run_1",
    );
    assert.deepEqual(stored.definition.evaluationProtocol.sampleIds, ["sample-1"]);
  }).pipe(Effect.provide(testLayer)),
);

const runStoreLayer = it.layer(testLayer);

const requestedAt = "2026-08-24T00:00:00.000Z";

const compatibility = {
  model: {
    baseModelId: "test/tiny",
    baseModelRevision: "a".repeat(40),
    tokenizerRevision: "b".repeat(40),
    peftConfig: {
      peftType: "LORA" as const,
      taskType: "CAUSAL_LM",
      rank: 2,
      alpha: 4,
      dropout: 0,
      bias: "none" as const,
      targetModules: ["linear"],
      modulesToSave: [],
      useRslora: false,
    },
    peftConfigSha256: "c".repeat(64),
    quantization: "none" as const,
    precision: "fp32" as const,
    trainableModules: ["linear"],
  },
  framework: { id: "fake", version: "0.1.0" },
  environmentFingerprint: "fixture",
  environmentLockSha256: null,
};

runStoreLayer("RunStore", (it) => {
  it.effect("indexes deterministic evaluation samples by their verified source artifact", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      const source = {
        runId: "run_evaluation",
        artifactId: "artifact_eval",
        artifactSha256: "c".repeat(64),
      };
      const protocolSha256 = "b".repeat(64);
      yield* store.indexEvaluationSamples({
        ...source,
        evaluation: {
          version: 1,
          protocolSha256,
          samples: [
            { sampleId: "b", generationSeed: 7, values: { "eval_after/loss": 2 } },
            {
              sampleId: "a",
              generationSeed: null,
              values: { "eval_after/loss": 1, "eval_after/accuracy": null },
            },
          ],
        },
      });
      const samples = yield* store.listEvaluationSamples({ ...source, protocolSha256 });
      assert.deepEqual(
        samples.map((sample) => sample.sampleId),
        ["a", "b"],
      );
      assert.equal(samples[0]?.generationSeed, null);
      assert.equal(samples[0]?.values["eval_after/accuracy"], null);
      assert.deepEqual(
        yield* store.listEvaluationSamples({
          ...source,
          protocolSha256,
          artifactSha256: "d".repeat(64),
        }),
        [],
      );
      yield* store.indexEvaluationSamples({
        ...source,
        evaluation: { version: 1, protocolSha256, samples: [samples[0]!] },
      });
      assert.equal((yield* store.listEvaluationSamples({ ...source, protocolSha256 })).length, 1);
    }),
  );

  it.effect("round trips a requested run into a listing", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      yield* store.insertRequested({
        runId: "run_01",
        projectId: "proj_01",
        experimentId: "fake",
        requestedAt,
      });

      const runs = yield* store.listRuns({ projectId: "proj_01", limit: 10 });
      assert.strictEqual(runs.length, 1);
      assert.strictEqual(runs[0]?.state, "requested");
      assert.strictEqual(runs[0]?.endedAt, null);
      assert.strictEqual(runs[0]?.startedAt, null);
    }),
  );

  it.effect("returns the original run for a repeated client request", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      const input = {
        runId: "run_idempotent_01",
        projectId: "proj_idempotent",
        experimentId: "fake",
        requestedAt,
        requestId: "request_01",
      };
      const first = yield* store.insertRequested(input);
      const retry = yield* store.insertRequested({ ...input, runId: "run_idempotent_02" });

      assert.deepStrictEqual(first, { runId: "run_idempotent_01", inserted: true });
      assert.deepStrictEqual(retry, { runId: "run_idempotent_01", inserted: false });
    }),
  );

  it.effect("fails if an ignored insert has no matching client request", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      const sql = yield* SqlClient.SqlClient;
      yield* sql`
        INSERT INTO rl_runs (run_id, project_id, experiment_id, state, requested_at)
        VALUES ('run_collision', 'proj_existing', 'fake', 'requested', ${requestedAt})
      `;

      const exit = yield* Effect.exit(
        store.insertRequested({
          runId: "run_collision",
          projectId: "proj_other",
          experimentId: "fake",
          requestedAt,
          requestId: "request_collision",
        }),
      );
      assert.isTrue(exit._tag === "Failure");
    }),
  );

  it.effect("stamps startedAt on the first move to running and endedAt on a terminal state", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      yield* store.insertRequested({
        runId: "run_02",
        projectId: "proj_01",
        experimentId: "fake",
        requestedAt,
      });
      yield* store.updateState({
        runId: "run_02",
        state: "running",
        at: "2026-08-24T00:00:05.000Z",
      });
      yield* store.updateState({
        runId: "run_02",
        state: "completed",
        at: "2026-08-24T00:00:09.000Z",
      });

      const { summary } = yield* store.getRun({ runId: "run_02" });
      assert.strictEqual(summary.startedAt, "2026-08-24T00:00:05.000Z");
      assert.strictEqual(summary.endedAt, "2026-08-24T00:00:09.000Z");
      assert.strictEqual(summary.state, "completed");
    }),
  );

  it.effect("keeps metrics in their own table, out of any run lifecycle row", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      yield* store.insertRequested({
        runId: "run_03",
        projectId: "proj_01",
        experimentId: "fake",
        requestedAt,
      });
      yield* store.appendMetrics({
        runId: "run_03",
        seq: 1,
        batch: { step: 1, wallClockMs: 10, values: { "train/return": 9.5, "train/kl": null } },
        at: "2026-08-24T00:00:02.000Z",
      });

      const metrics = yield* store.listMetrics({ runId: "run_03", limit: 100 });
      assert.strictEqual(metrics.length, 1);
      assert.strictEqual(metrics[0]?.values["train/return"], 9.5);
      assert.strictEqual(metrics[0]?.values["train/kl"], null);

      const { summary } = yield* store.getRun({ runId: "run_03" });
      assert.strictEqual(summary.lastMessageAt, "2026-08-24T00:00:02.000Z");
      assert.strictEqual(summary.state, "requested");
    }),
  );

  it.effect("preserves a non-finite marker instead of collapsing it to null", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      yield* store.insertRequested({
        runId: "run_04",
        projectId: "proj_01",
        experimentId: "fake",
        requestedAt,
      });
      yield* store.appendMetrics({
        runId: "run_04",
        seq: 1,
        batch: { step: 1, wallClockMs: 10, values: { "train/loss": "nan" } },
        at: requestedAt,
      });

      const metrics = yield* store.listMetrics({ runId: "run_04", limit: 10 });
      assert.strictEqual(metrics[0]?.values["train/loss"], "nan");
    }),
  );

  it.effect("refuses to overwrite a manifest once a run has one", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      yield* store.insertRequested({
        runId: "run_05",
        projectId: "proj_01",
        experimentId: "fake",
        requestedAt,
      });
      const manifest = {
        experimentId: "fake",
        runnerId: "fake",
        runnerVersion: "0.1.0",
        protocolVersion: 1,
        seed: 7,
        effectiveConfig: {},
        sourceRevision: null,
        sourceDirty: false,
        pythonExecutable: "/usr/bin/python3",
        pythonVersion: "3.12.4",
        environmentFingerprint: "test",
        instrumentationLevel: "minimal" as const,
        hardwareSummary: "cpu",
      };
      yield* store.setManifest({ runId: "run_05", manifest });

      // A manifest that changes mid-run destroys reproducibility, so the second
      // write is refused rather than merged.
      const exit = yield* Effect.exit(store.setManifest({ runId: "run_05", manifest }));
      assert.isTrue(exit._tag === "Failure");

      const { manifest: stored } = yield* store.getRun({ runId: "run_05" });
      assert.strictEqual(stored?.seed, 7);
    }),
  );

  it.effect("fails with RlRunNotFoundError for an unknown run", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      const exit = yield* Effect.exit(store.getRun({ runId: "missing" }));
      assert.isTrue(exit._tag === "Failure");
    }),
  );

  it.effect("records artifacts with opaque ids scoped to their run", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      yield* store.insertRequested({
        runId: "run_06",
        projectId: "proj_01",
        experimentId: "fake",
        requestedAt,
      });
      const artifact = yield* store.recordArtifact({
        runId: "run_06",
        kind: "summary",
        relativePath: "summary.json",
        bytes: 42,
        contentType: "application/json",
        producedAt: requestedAt,
        sha256: "a".repeat(64),
        logicalName: "summary.json",
        format: "json",
        fileCount: 1,
        contentManifest: [{ path: "summary.json", bytes: 42, sha256: "a".repeat(64) }],
      });
      assert.isTrue(artifact.artifactId.length > 0);
      assert.isFalse(artifact.artifactId.includes("/"));

      const page = yield* store.listArtifacts({ runId: "run_06", limit: 10 });
      assert.strictEqual(page.artifacts.length, 1);
      assert.strictEqual(page.artifacts[0]?.kind, "summary");
      assert.strictEqual(page.artifacts[0]?.sha256, "a".repeat(64));
      assert.strictEqual(page.artifacts[0]?.state, "ready");
      assert.strictEqual(page.nextCursor, null);
    }),
  );

  it.effect("paginates artifacts deterministically when timestamps are equal", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      yield* store.insertRequested({
        runId: "run_artifact_pages",
        projectId: "proj_01",
        experimentId: "fake",
        requestedAt,
      });
      const recorded = [];
      for (const name of ["one.json", "two.json", "three.json"]) {
        recorded.push(
          yield* store.recordArtifact({
            runId: "run_artifact_pages",
            kind: "summary",
            relativePath: name,
            bytes: 2,
            contentType: "application/json",
            producedAt: requestedAt,
            sha256: "b".repeat(64),
            logicalName: name,
            format: "json",
            fileCount: 1,
            contentManifest: [{ path: name, bytes: 2, sha256: "b".repeat(64) }],
          }),
        );
      }

      const expectedIds = recorded
        .map((artifact) => artifact.artifactId)
        .sort()
        .toReversed();
      const first = yield* store.listArtifacts({ runId: "run_artifact_pages", limit: 2 });
      assert.deepStrictEqual(
        first.artifacts.map((artifact) => artifact.artifactId),
        expectedIds.slice(0, 2),
      );
      assert.strictEqual(first.nextCursor, expectedIds[1]);

      const second = yield* store.listArtifacts({
        runId: "run_artifact_pages",
        limit: 2,
        cursor: first.nextCursor!,
      });
      assert.deepStrictEqual(
        second.artifacts.map((artifact) => artifact.artifactId),
        expectedIds.slice(2),
      );
      assert.strictEqual(second.nextCursor, null);
    }),
  );

  it.effect("reads pre-identity artifact rows as legacy evidence", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      const sql = yield* SqlClient.SqlClient;
      yield* store.insertRequested({
        runId: "run_legacy_artifact",
        projectId: "proj_01",
        experimentId: "fake",
        requestedAt,
      });
      yield* sql`
        INSERT INTO rl_run_artifacts
          (artifact_id, run_id, kind, relative_path, bytes, content_type, produced_at)
        VALUES (
          'artifact_legacy',
          'run_legacy_artifact',
          'summary',
          'summary.json',
          42,
          'application/json',
          ${requestedAt}
        )
      `;

      const page = yield* store.listArtifacts({ runId: "run_legacy_artifact", limit: 10 });
      assert.deepInclude(page.artifacts[0], {
        artifactId: "artifact_legacy",
        sha256: null,
        logicalName: "summary.json",
        format: "unknown",
        state: "ready",
        fileCount: 0,
      });
    }),
  );

  it.effect("round trips checkpoint evidence and updates retention state", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      yield* store.insertRequested({
        runId: "run_checkpoint_evidence",
        projectId: "proj_01",
        experimentId: "fake",
        requestedAt,
      });
      const artifact = yield* store.recordArtifact({
        runId: "run_checkpoint_evidence",
        kind: "checkpoint",
        relativePath: "checkpoints/checkpoint-4-intermediate",
        bytes: 42,
        contentType: "application/vnd.t3rl.directory.v1",
        producedAt: requestedAt,
        sha256: "d".repeat(64),
        logicalName: "checkpoints/checkpoint-4-intermediate",
        format: "directory-v1",
        checkpointStep: 4,
        fileCount: 7,
        contentManifest: [],
        evidence: {
          _tag: "Checkpoint",
          checkpointClass: "intermediate",
          compatibility,
          globalStep: 4,
          tokensSeen: 64,
          datasetCursor: { epoch: 0.5, batchInEpoch: 4, sampleOffset: 4 },
          resumeState: {
            trainerState: true,
            optimizerState: true,
            schedulerState: true,
            rngState: true,
            datasetCursorState: true,
            gradientScalerState: "not-applicable",
            stateFiles: [
              "trainer_state.json",
              "optimizer.pt",
              "scheduler.pt",
              "rng_state.pth",
              "t3rl-dataset-cursor.json",
            ],
          },
        },
      });

      const found = yield* store.findArtifact({
        runId: "run_checkpoint_evidence",
        artifactId: artifact.artifactId,
      });
      assert.strictEqual(found?.metadata.evidence?._tag, "Checkpoint");
      assert.strictEqual(found?.metadata.checkpointStep, 4);
      assert.strictEqual(
        (yield* store.listReadyIntermediateCheckpoints({ runId: "run_checkpoint_evidence" }))
          .length,
        1,
      );
      const trashed = yield* store.setArtifactState({
        artifactId: artifact.artifactId,
        state: "trashed",
      });
      assert.strictEqual(trashed?.state, "trashed");
      assert.strictEqual(
        (yield* store.listReadyIntermediateCheckpoints({ runId: "run_checkpoint_evidence" }))
          .length,
        0,
      );
    }),
  );

  it.effect("stores and returns a bounded nearest-parent-first lineage chain", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      yield* store.insertRequested({
        runId: "run_lineage_root",
        projectId: "proj_lineage",
        experimentId: "fake",
        requestedAt,
      });
      yield* store.insertRequested({
        runId: "run_lineage_child",
        projectId: "proj_lineage",
        experimentId: "fake",
        requestedAt,
        lineage: {
          childRunId: "run_lineage_child",
          parentRunId: "run_lineage_root",
          sourceArtifactId: "artifact_root",
          sourceArtifactSha256: "e".repeat(64),
          relation: "resume",
          sourceStep: 4,
          createdAt: requestedAt,
        },
      });
      yield* store.insertRequested({
        runId: "run_lineage_grandchild",
        projectId: "proj_lineage",
        experimentId: "fake",
        requestedAt,
        lineage: {
          childRunId: "run_lineage_grandchild",
          parentRunId: "run_lineage_child",
          sourceArtifactId: "artifact_child",
          sourceArtifactSha256: "f".repeat(64),
          relation: "warm-start",
          sourceStep: 8,
          createdAt: requestedAt,
        },
      });

      const lineage = yield* store.getLineage({ runId: "run_lineage_grandchild" });
      assert.deepStrictEqual(
        lineage.edges.map((edge) => [edge.childRunId, edge.parentRunId, edge.relation]),
        [
          ["run_lineage_grandchild", "run_lineage_child", "warm-start"],
          ["run_lineage_child", "run_lineage_root", "resume"],
        ],
      );
      assert.isFalse(lineage.truncated);
    }),
  );
});

// The restart sweep is global by design, so it gets a database no other test
// has written to. Sharing one would make its count assertion depend on the
// order the surrounding tests happened to run in.
const sweepLayer = it.layer(
  RunStoreLive.pipe(
    Layer.provideMerge(SqlitePersistenceMemory),
    Layer.provideMerge(NodeServices.layer),
  ),
);

sweepLayer("RunStore restart sweep", (it) => {
  it.effect("marks active runs interrupted and leaves terminal runs alone", () =>
    Effect.gen(function* () {
      const store = yield* RunStore;
      for (const [runId, state] of [
        ["run_active", "running"],
        ["run_prep", "preparing"],
        ["run_cancelling", "cancelling"],
        ["run_done", "completed"],
        ["run_failed", "failed"],
      ] as const) {
        yield* store.insertRequested({
          runId,
          projectId: "proj_sweep",
          experimentId: "fake",
          requestedAt,
        });
        yield* store.updateState({ runId, state, at: "2026-08-24T00:00:01.000Z" });
      }

      const swept = yield* store.markActiveAsInterrupted({ at: "2026-08-24T01:00:00.000Z" });
      assert.strictEqual(swept, 3);

      const runs = yield* store.listRuns({ projectId: "proj_sweep", limit: 10 });
      const byId = new Map(runs.map((run) => [run.runId, run]));
      assert.strictEqual(byId.get("run_active")?.state, "interrupted");
      assert.strictEqual(byId.get("run_prep")?.state, "interrupted");
      assert.strictEqual(byId.get("run_cancelling")?.state, "interrupted");
      assert.strictEqual(byId.get("run_active")?.errorCode, "ServerInterrupted");
      // A finished run keeps its result: a restart must not rewrite history.
      assert.strictEqual(byId.get("run_done")?.state, "completed");
      assert.strictEqual(byId.get("run_failed")?.state, "failed");
    }),
  );
});
