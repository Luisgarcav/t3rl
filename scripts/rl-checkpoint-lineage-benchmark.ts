// @effect-diagnostics nodeBuiltinImport:off globalDate:off - host benchmark measurements.
import * as NodeCrypto from "node:crypto";
import * as NodeFSP from "node:fs/promises";
import * as NodeOS from "node:os";
import * as NodePath from "node:path";
import * as NodeProcess from "node:process";
import * as NodeSqlite from "node:sqlite";

import type { RlArtifactMetadata, RlRunLineage, RlSubscriptionEvent } from "@t3tools/contracts";

import { applyRlSubscriptionEvent, type RlRunProjection } from "@t3tools/client-runtime/state/rl";

const CHECKPOINT_COUNT = 24;
const RETAINED_INTERMEDIATE_CHECKPOINTS = 2;
const LINEAGE_DEPTH = 16;
const STATE_FILES = [
  "adapter_config.json",
  "adapter_model.safetensors",
  "trainer_state.json",
  "optimizer.pt",
  "scheduler.pt",
  "rng_state.pth",
  "t3rl-dataset-cursor.json",
] as const;

const summary = {
  runId: "run_checkpoint_benchmark",
  projectId: "project_benchmark",
  experimentId: "fake-checkpoint-smoke",
  state: "running" as const,
  requestedAt: "2026-09-04T00:00:00.000Z",
  startedAt: "2026-09-04T00:00:00.001Z",
  endedAt: null,
  lastMessageAt: "2026-09-04T00:00:00.001Z",
  errorCode: null,
  errorMessage: null,
};

const compatibility = {
  model: {
    baseModelId: "t3rl/tiny-benchmark",
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
  environmentFingerprint: "benchmark-environment",
  environmentLockSha256: null,
};

const lineage: RlRunLineage = {
  edges: Array.from({ length: LINEAGE_DEPTH }, (_, depth) => ({
    childRunId: depth === 0 ? summary.runId : `run_parent_${depth}`,
    parentRunId: `run_parent_${depth + 1}`,
    sourceArtifactId: `artifact_parent_${depth + 1}`,
    sourceArtifactSha256: (depth % 16).toString(16).repeat(64),
    relation: depth % 2 === 0 ? ("resume" as const) : ("warm-start" as const),
    sourceStep: (LINEAGE_DEPTH - depth) * 100,
    createdAt: `2026-09-04T00:00:${String(depth).padStart(2, "0")}.000Z`,
  })),
  truncated: false,
};

const fileSize = async (file: string): Promise<number> => {
  try {
    return (await NodeFSP.stat(file)).size;
  } catch {
    return 0;
  }
};

const treeBytes = async (root: string): Promise<number> => {
  let bytes = 0;
  for (const name of await NodeFSP.readdir(root, { recursive: true })) {
    const stat = await NodeFSP.stat(NodePath.join(root, name));
    if (stat.isFile()) bytes += stat.size;
  }
  return bytes;
};

const computeIdentity = async (artifactPath: string) => {
  const names = (await NodeFSP.readdir(artifactPath, { recursive: true })).toSorted();
  const contentManifest: Array<{
    readonly path: string;
    readonly bytes: number;
    readonly sha256: string;
  }> = [];
  for (const name of names) {
    const file = NodePath.join(artifactPath, name);
    const stat = await NodeFSP.stat(file);
    if (!stat.isFile()) continue;
    const contents = await NodeFSP.readFile(file);
    contentManifest.push({
      path: name.replaceAll("\\", "/"),
      bytes: contents.byteLength,
      sha256: NodeCrypto.createHash("sha256").update(contents).digest("hex"),
    });
  }
  const bytes = contentManifest.reduce((total, entry) => total + entry.bytes, 0);
  const sha256 = NodeCrypto.createHash("sha256")
    .update(JSON.stringify({ version: 1, files: contentManifest }), "utf8")
    .digest("hex");
  return { bytes, fileCount: contentManifest.length, sha256 };
};

const publishCheckpoint = async (parent: string, step: number): Promise<string> => {
  const checkpoints = NodePath.join(parent, "checkpoints");
  await NodeFSP.mkdir(checkpoints, { recursive: true });
  const name = `checkpoint-${step}-intermediate`;
  const staging = await NodeFSP.mkdtemp(NodePath.join(checkpoints, `.${name}.`));
  const target = NodePath.join(checkpoints, name);
  try {
    await Promise.all([
      NodeFSP.writeFile(
        NodePath.join(staging, "adapter_config.json"),
        JSON.stringify({ base_model_name_or_path: compatibility.model.baseModelId }),
      ),
      NodeFSP.writeFile(
        NodePath.join(staging, "adapter_model.safetensors"),
        Buffer.alloc(32 * 1024, step),
      ),
      NodeFSP.writeFile(
        NodePath.join(staging, "trainer_state.json"),
        JSON.stringify({ global_step: step }),
      ),
      NodeFSP.writeFile(NodePath.join(staging, "optimizer.pt"), Buffer.alloc(8 * 1024, step)),
      NodeFSP.writeFile(NodePath.join(staging, "scheduler.pt"), String(step)),
      NodeFSP.writeFile(NodePath.join(staging, "rng_state.pth"), String(step)),
      NodeFSP.writeFile(
        NodePath.join(staging, "t3rl-dataset-cursor.json"),
        JSON.stringify({ epoch: step / CHECKPOINT_COUNT, batchInEpoch: step, sampleOffset: step }),
      ),
    ]);
    await NodeFSP.rename(staging, target);
    return target;
  } catch (error) {
    await NodeFSP.rm(staging, { force: true, recursive: true });
    throw error;
  }
};

const main = async (): Promise<void> => {
  const directory = await NodeFSP.mkdtemp(NodePath.join(NodeOS.tmpdir(), "t3rl-checkpoints-"));
  try {
    const databasePath = NodePath.join(directory, "state.sqlite");
    const parent = NodePath.join(directory, "parent");
    const childInput = NodePath.join(directory, "child", "inputs", "checkpoint");
    await NodeFSP.mkdir(parent, { recursive: true });
    const database = new NodeSqlite.DatabaseSync(databasePath);
    database.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
    database.exec(`
      CREATE TABLE rl_run_artifacts (
        artifact_id TEXT PRIMARY KEY,
        checkpoint_step INTEGER NOT NULL,
        state TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        bytes INTEGER NOT NULL,
        evidence_json TEXT NOT NULL
      );
      CREATE TABLE rl_run_lineage (
        child_run_id TEXT PRIMARY KEY,
        parent_run_id TEXT NOT NULL,
        source_artifact_id TEXT NOT NULL,
        source_artifact_sha256 TEXT NOT NULL,
        relation TEXT NOT NULL,
        source_step INTEGER NOT NULL
      );
    `);
    const insertArtifact = database.prepare(
      "INSERT INTO rl_run_artifacts VALUES (?, ?, 'ready', ?, ?, ?)",
    );
    const trashArtifact = database.prepare(
      "UPDATE rl_run_artifacts SET state = 'trashed' WHERE artifact_id = ?",
    );
    const insertLineage = database.prepare("INSERT INTO rl_run_lineage VALUES (?, ?, ?, ?, ?, ?)");

    let sqliteWriteStatements = 0;
    const rssStartBytes = NodeProcess.memoryUsage().rss;
    let rssPeakBytes = rssStartBytes;
    const cpuStart = NodeProcess.cpuUsage();
    const wallStart = performance.now();

    for (const edge of lineage.edges) {
      insertLineage.run(
        edge.childRunId,
        edge.parentRunId,
        edge.sourceArtifactId,
        edge.sourceArtifactSha256,
        edge.relation,
        edge.sourceStep,
      );
      sqliteWriteStatements += 1;
    }

    const snapshot: RlSubscriptionEvent = {
      _tag: "Snapshot",
      summary,
      manifest: null,
      lineage,
      artifacts: [],
      metrics: [],
    };
    let projection: RlRunProjection | null = applyRlSubscriptionEvent(null, snapshot);
    let websocketBytes = Buffer.byteLength(JSON.stringify(snapshot));
    let reducerEvents = 1;
    let artifactRowVisits = 0;
    const lineageEdgeVisits = lineage.edges.length;
    let artifactBytesProduced = 0;
    const ready: Array<{ artifact: RlArtifactMetadata; path: string }> = [];

    for (let index = 1; index <= CHECKPOINT_COUNT; index += 1) {
      const step = index * 100;
      const artifactPath = await publishCheckpoint(parent, step);
      const identity = await computeIdentity(artifactPath);
      artifactBytesProduced += identity.bytes;
      const evidence = {
        _tag: "Checkpoint" as const,
        checkpointClass: "intermediate" as const,
        compatibility,
        globalStep: step,
        tokensSeen: step * 16,
        datasetCursor: { epoch: index / CHECKPOINT_COUNT, batchInEpoch: step, sampleOffset: step },
        resumeState: {
          trainerState: true,
          optimizerState: true,
          schedulerState: true,
          rngState: true,
          datasetCursorState: true,
          gradientScalerState: "not-applicable" as const,
          stateFiles: STATE_FILES.slice(2),
        },
      };
      const artifact: RlArtifactMetadata = {
        artifactId: `artifact_checkpoint_${index}`,
        kind: "checkpoint",
        bytes: identity.bytes,
        contentType: "application/vnd.t3rl.directory.v1",
        producedAt: `2026-09-04T00:01:${String(index).padStart(2, "0")}.000Z`,
        sha256: identity.sha256,
        logicalName: NodePath.relative(parent, artifactPath),
        format: "directory-v1",
        state: "ready",
        checkpointStep: step,
        fileCount: identity.fileCount,
        evidence,
      };
      insertArtifact.run(
        artifact.artifactId,
        step,
        identity.sha256,
        identity.bytes,
        JSON.stringify(evidence),
      );
      sqliteWriteStatements += 1;
      ready.push({ artifact, path: artifactPath });
      let event: RlSubscriptionEvent = { _tag: "Artifact", artifact };
      websocketBytes += Buffer.byteLength(JSON.stringify(event));
      projection = applyRlSubscriptionEvent(projection, event);
      reducerEvents += 1;
      artifactRowVisits += projection?.artifacts.length ?? 0;

      if (ready.length > RETAINED_INTERMEDIATE_CHECKPOINTS) {
        const expired = ready.shift();
        if (expired === undefined) throw new Error("retention queue was unexpectedly empty");
        await NodeFSP.rm(expired.path, { recursive: true });
        trashArtifact.run(expired.artifact.artifactId);
        sqliteWriteStatements += 1;
        const trashed: RlArtifactMetadata = { ...expired.artifact, state: "trashed" };
        event = { _tag: "Artifact", artifact: trashed };
        websocketBytes += Buffer.byteLength(JSON.stringify(event));
        projection = applyRlSubscriptionEvent(projection, event);
        reducerEvents += 1;
        artifactRowVisits += projection?.artifacts.length ?? 0;
      }
      rssPeakBytes = Math.max(rssPeakBytes, NodeProcess.memoryUsage().rss);
    }

    const source = ready.at(-1);
    if (
      source === undefined ||
      source.artifact.sha256 === null ||
      source.artifact.sha256 === undefined
    ) {
      throw new Error("benchmark produced no verified resume source");
    }
    const parentBefore = await computeIdentity(parent);
    await NodeFSP.mkdir(NodePath.dirname(childInput), { recursive: true });
    await NodeFSP.cp(source.path, childInput, { recursive: true, errorOnExist: true });
    const copied = await computeIdentity(childInput);
    const parentAfter = await computeIdentity(parent);

    database.exec("PRAGMA wal_checkpoint(PASSIVE)");
    const counts = database
      .prepare(
        "SELECT COUNT(*) AS total, SUM(state = 'ready') AS ready, SUM(state = 'trashed') AS trashed FROM rl_run_artifacts",
      )
      .get() as { total: number; ready: number; trashed: number };
    const lineageRows = database.prepare("SELECT COUNT(*) AS count FROM rl_run_lineage").get() as {
      count: number;
    };
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
        checkpointCount: CHECKPOINT_COUNT,
        retainedIntermediateCheckpoints: RETAINED_INTERMEDIATE_CHECKPOINTS,
        filesPerCheckpoint: STATE_FILES.length,
        lineageDepth: LINEAGE_DEPTH,
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
        sqliteWriteStatements,
        sqliteBytes,
        websocketBytes,
        artifactBytesProduced,
        retainedParentCheckpointBytes: await treeBytes(parent),
        childInputBytes: await treeBytes(childInput),
        renderer: { reducerEvents, artifactRowVisits, lineageEdgeVisits },
      },
      checks: {
        artifactRows: counts.total,
        readyIntermediateCheckpoints: counts.ready,
        trashedIntermediateCheckpoints: counts.trashed,
        lineageRows: lineageRows.count,
        sourceCopyHashMatches: copied.sha256 === source.artifact.sha256,
        parentHashUnchanged: parentAfter.sha256 === parentBefore.sha256,
        retentionBounded: counts.ready === RETAINED_INTERMEDIATE_CHECKPOINTS,
      },
    };
    NodeProcess.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  } finally {
    await NodeFSP.rm(directory, { force: true, recursive: true });
  }
};

await main();
