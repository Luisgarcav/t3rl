// @effect-diagnostics nodeBuiltinImport:off globalDate:off - host benchmark measurements.
import * as NodeFSP from "node:fs/promises";
import * as NodeOS from "node:os";
import * as NodePath from "node:path";
import * as NodeProcess from "node:process";
import * as NodeSqlite from "node:sqlite";

import type { RlMetricBatch, RlSubscriptionEvent } from "@t3tools/contracts";

import { applyRlSubscriptionEvent, type RlRunProjection } from "@t3tools/client-runtime/state/rl";

const SEMANTIC_BATCHES = 5_000;
const INCOMING_BATCHES_PER_SECOND = 50;
const LIVE_BATCHES_PER_SECOND = 2;
const PUBLICATION_STRIDE = INCOMING_BATCHES_PER_SECOND / LIVE_BATCHES_PER_SECOND;

const fileSize = async (file: string): Promise<number> => {
  try {
    return (await NodeFSP.stat(file)).size;
  } catch {
    return 0;
  }
};

const summary = {
  runId: "run_benchmark",
  projectId: "project_benchmark",
  experimentId: "fake",
  state: "running" as const,
  requestedAt: "2026-09-04T00:00:00.000Z",
  startedAt: "2026-09-04T00:00:00.001Z",
  endedAt: null,
  lastMessageAt: "2026-09-04T00:00:00.001Z",
  errorCode: null,
  errorMessage: null,
};

const fixtureBatch = (step: number): RlMetricBatch => ({
  step,
  wallClockMs: step * 20,
  values: {
    "train/reward": (step % 101) / 100,
    "train/loss": 1 / (step + 1),
    "train/kl": step % 17 === 0 ? null : (step % 13) / 1_000,
    "system/tokens_per_second": 128 + (step % 32),
  },
});

const main = async (): Promise<void> => {
  if (!Number.isInteger(PUBLICATION_STRIDE)) {
    throw new Error("The benchmark fixture requires an integer publication stride.");
  }
  const directory = await NodeFSP.mkdtemp(NodePath.join(NodeOS.tmpdir(), "t3rl-fake-stream-"));
  try {
    const databasePath = NodePath.join(directory, "state.sqlite");
    const artifactDirectory = NodePath.join(directory, "artifacts");
    await NodeFSP.mkdir(artifactDirectory);
    const database = new NodeSqlite.DatabaseSync(databasePath);
    database.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
    database.exec(`
      CREATE TABLE rl_run_metrics (
        run_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        step INTEGER NOT NULL,
        wall_clock_ms INTEGER NOT NULL,
        values_json TEXT NOT NULL,
        PRIMARY KEY (run_id, seq)
      )
    `);
    const insert = database.prepare(
      "INSERT INTO rl_run_metrics (run_id, seq, step, wall_clock_ms, values_json) VALUES (?, ?, ?, ?, ?)",
    );

    let projection: RlRunProjection | null = applyRlSubscriptionEvent(null, {
      _tag: "Snapshot",
      summary,
      manifest: null,
      lineage: { edges: [], truncated: false },
      artifacts: [],
      metrics: [],
    });
    let websocketBytes = Buffer.byteLength(
      JSON.stringify({
        _tag: "Snapshot",
        summary,
        manifest: null,
        lineage: { edges: [], truncated: false },
        artifacts: [],
        metrics: [],
      }),
    );
    let publishedMetricBatches = 0;
    let reducerEvents = 1;
    let chartPointVisits = 0;
    let rssPeakBytes = NodeProcess.memoryUsage().rss;
    const rssStartBytes = rssPeakBytes;
    const cpuStart = NodeProcess.cpuUsage();
    const wallStart = performance.now();

    for (let step = 1; step <= SEMANTIC_BATCHES; step += 1) {
      const batch = fixtureBatch(step);
      insert.run(
        "run_benchmark",
        step,
        batch.step,
        batch.wallClockMs,
        JSON.stringify(batch.values),
      );
      if (step % PUBLICATION_STRIDE === 0 || step === SEMANTIC_BATCHES) {
        const event: RlSubscriptionEvent = { _tag: "Metrics", batch };
        websocketBytes += Buffer.byteLength(JSON.stringify(event));
        projection = applyRlSubscriptionEvent(projection, event);
        publishedMetricBatches += 1;
        reducerEvents += 1;
        chartPointVisits += Math.min(projection?.metrics.length ?? 0, 120);
      }
      if (step % 100 === 0) {
        rssPeakBytes = Math.max(rssPeakBytes, NodeProcess.memoryUsage().rss);
      }
    }

    const artifactSizes = [64 * 1_024, 256 * 1_024, 4 * 1_024];
    for (const [index, bytes] of artifactSizes.entries()) {
      await NodeFSP.writeFile(
        NodePath.join(artifactDirectory, `artifact-${index}.bin`),
        Buffer.alloc(bytes, index),
      );
    }
    database.exec("PRAGMA wal_checkpoint(PASSIVE)");
    const persisted = database
      .prepare("SELECT COUNT(*) AS count, COUNT(DISTINCT step) AS steps FROM rl_run_metrics")
      .get() as { count: number; steps: number };
    const sqliteBytes =
      (await fileSize(databasePath)) +
      (await fileSize(`${databasePath}-wal`)) +
      (await fileSize(`${databasePath}-shm`));
    database.close();

    const cpu = NodeProcess.cpuUsage(cpuStart);
    const rssEndBytes = NodeProcess.memoryUsage().rss;
    rssPeakBytes = Math.max(rssPeakBytes, rssEndBytes);
    const result = {
      schemaVersion: 1,
      recordedAt: new Date().toISOString(),
      fixture: {
        semanticMetricBatches: SEMANTIC_BATCHES,
        incomingBatchesPerSecond: INCOMING_BATCHES_PER_SECOND,
        liveBatchLimitPerSecond: LIVE_BATCHES_PER_SECOND,
        artifactCount: artifactSizes.length,
      },
      referenceHardware: {
        platform: `${NodeOS.type()} ${NodeOS.release()}`,
        architecture: NodeOS.machine(),
        cpuModel: NodeOS.cpus()[0]?.model ?? "unknown",
        logicalCpuCount: NodeOS.cpus().length,
        totalMemoryBytes: NodeOS.totalmem(),
        nodeVersion: NodeProcess.version,
      },
      measurements: {
        wallClockMs: Number((performance.now() - wallStart).toFixed(3)),
        cpuUserMs: Number((cpu.user / 1_000).toFixed(3)),
        cpuSystemMs: Number((cpu.system / 1_000).toFixed(3)),
        rssStartBytes,
        rssEndBytes,
        rssPeakBytes,
        sqliteWriteStatements: SEMANTIC_BATCHES,
        sqliteBytes,
        websocketBytes,
        artifactBytes: artifactSizes.reduce((total, bytes) => total + bytes, 0),
        renderer: { reducerEvents, chartPointVisits },
      },
      checks: {
        persistedMetricRows: persisted.count,
        persistedDistinctSteps: persisted.steps,
        publishedMetricBatches,
        semanticStepsPreserved:
          persisted.count === SEMANTIC_BATCHES && persisted.steps === SEMANTIC_BATCHES,
        transportBounded: publishedMetricBatches === SEMANTIC_BATCHES / PUBLICATION_STRIDE,
      },
    };
    NodeProcess.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  } finally {
    await NodeFSP.rm(directory, { force: true, recursive: true });
  }
};

await main();
