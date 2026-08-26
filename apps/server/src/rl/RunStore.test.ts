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

const runStoreLayer = it.layer(testLayer);

const requestedAt = "2026-08-24T00:00:00.000Z";

runStoreLayer("RunStore", (it) => {
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
      });
      assert.isTrue(artifact.artifactId.length > 0);
      assert.isFalse(artifact.artifactId.includes("/"));

      const artifacts = yield* store.listArtifacts({ runId: "run_06" });
      assert.strictEqual(artifacts.length, 1);
      assert.strictEqual(artifacts[0]?.kind, "summary");
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
