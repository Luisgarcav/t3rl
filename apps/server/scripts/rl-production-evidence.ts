#!/usr/bin/env node
/** Opt-in CUDA reference through the real project, launcher, persistence, and evidence services. */
// @effect-diagnostics nodeBuiltinImport:off - standalone validation owns its temporary project and child processes.
// @effect-diagnostics preferSchemaOverJson:off - reports and canonical artifact identities are persisted evidence.
import * as NodeChildProcess from "node:child_process";
import * as NodeCrypto from "node:crypto";
import * as NodeFSP from "node:fs/promises";
import * as NodePath from "node:path";
import * as NodeUtil from "node:util";

import * as NodeServices from "@effect/platform-node/NodeServices";
import {
  isTerminalRlRunState,
  ProjectId,
  RlEvaluationProtocol,
  RlProjectExperimentDefinition,
  type RlArtifactKind,
  type RlRunState,
  type RlStudy,
} from "@t3tools/contracts";
import * as Console from "effect/Console";
import * as DateTime from "effect/DateTime";
import * as Deferred from "effect/Deferred";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Schema from "effect/Schema";
import * as SqlClient from "effect/unstable/sql/SqlClient";

import * as ServerConfig from "../src/config.ts";
import { ProjectionProjectRepositoryLive } from "../src/persistence/Layers/ProjectionProjects.ts";
import { makeSqlitePersistenceLive } from "../src/persistence/Layers/Sqlite.ts";
import { ProjectionProjectRepository } from "../src/persistence/Services/ProjectionProjects.ts";
import * as ProcessRunner from "../src/processRunner.ts";
import { computeArtifactIdentity, inferArtifactFormat } from "../src/rl/ArtifactIdentity.ts";
import { CapabilitiesLive } from "../src/rl/Capabilities.ts";
import {
  canonicalEvidenceJson,
  EVIDENCE_VERIFIER,
  exportEvidence,
  recordResearch,
} from "../src/rl/EvidenceBundle.ts";
import { evidenceExportPath } from "../src/rl/EvidencePaths.ts";
import * as Experiments from "../src/rl/Experiments.ts";
import { RlManager, RlManagerLive } from "../src/rl/Manager.ts";
import { RunStore, RunStoreLive } from "../src/rl/RunStore.ts";
import { SourceEvidenceLive } from "../src/rl/SourceEvidence.ts";
import { WorkerSpawnerLive } from "../src/rl/WorkerSpawner.ts";

if (process.env["T3RL_RUN_PRODUCTION_VALIDATION"] !== "1")
  throw new Error("Set T3RL_RUN_PRODUCTION_VALIDATION=1 to launch six small real training runs.");

const workspace = NodePath.resolve(import.meta.dirname, "../../..");
const output = NodePath.join(workspace, ".t3/serious-validation/production-evidence");
const python = NodePath.join(workspace, ".venv-trl-validation/bin/python");
const captured = NodePath.join(workspace, ".t3/serious-validation/trl-gpu");
const execFile = NodeUtil.promisify(NodeChildProcess.execFile);
const sha256 = (value: string | Uint8Array) =>
  NodeCrypto.createHash("sha256").update(value).digest("hex");
const decodeSummary = Schema.decodeUnknownSync(
  Schema.Struct({
    dataSeedApplied: Schema.Int,
    optimizerSteps: Schema.Int,
    device: Schema.String,
    after: Schema.Struct({ eval_loss: Schema.Finite }),
  }),
);
const decodeWeightComparison = Schema.decodeUnknownSync(
  Schema.Struct({
    maxAbsoluteDifference: Schema.Finite,
    tensorCount: Schema.Int,
    tolerance: Schema.Finite,
  }),
);
await NodeFSP.mkdir(output, { recursive: true });
const state = await NodeFSP.mkdtemp(NodePath.join(output, "launch-"));
const project = NodePath.join(state, "project");
const worker = NodePath.join(project, "worker");
const runsDir = NodePath.join(state, "runs");
const environment = NodePath.join(state, "environment");
await NodeFSP.mkdir(NodePath.join(project, ".t3rl/experiments"), { recursive: true });
await NodeFSP.mkdir(NodePath.join(worker, "experiments"), { recursive: true });
await NodeFSP.mkdir(environment);
await NodeFSP.cp(NodePath.join(captured, "model"), NodePath.join(project, "model"), {
  recursive: true,
});
await NodeFSP.copyFile(
  NodePath.join(captured, "sft-seed-7/dataset.json"),
  NodePath.join(project, "dataset.json"),
);
const workerHashes: Record<string, string> = {};
for (const name of (await NodeFSP.readdir(NodePath.join(workspace, "python/t3rl_worker"))).sort()) {
  if (!name.endsWith(".py")) continue;
  const bytes = await NodeFSP.readFile(NodePath.join(workspace, "python/t3rl_worker", name));
  await NodeFSP.writeFile(NodePath.join(worker, name), bytes);
  workerHashes[name] = sha256(bytes);
}
const modelHashes: Record<string, string> = {};
for (const name of (await NodeFSP.readdir(NodePath.join(project, "model"))).sort())
  modelHashes[name] = sha256(await NodeFSP.readFile(NodePath.join(project, "model", name)));
const modelRevision = sha256(canonicalEvidenceJson(modelHashes));
const environmentHashes: Record<string, string> = {};
for (const name of ["pyproject.toml", "uv.lock"]) {
  const bytes = await NodeFSP.readFile(NodePath.join(workspace, "python/environments/trl", name));
  await NodeFSP.writeFile(NodePath.join(environment, name), bytes);
  environmentHashes[name] = sha256(bytes);
}
// Reuse the installed environment without modifying it. Capabilities still probes Python,
// imports TRL/PyTorch, checks this real uv project, and fingerprints the selected interpreter.
await NodeFSP.symlink(
  NodePath.dirname(NodePath.dirname(python)),
  NodePath.join(environment, ".venv"),
);
process.env["T3RL_PYTHON_TRL"] = NodePath.join(environment, ".venv/bin/python");
const baseConfig = {
  method: "sft",
  datasetFormat: "sft-text",
  evaluationClaim: "held-out-loss",
  modelId: "model",
  modelRevision,
  tokenizerRevision: modelRevision,
  maxSteps: 6,
  evaluationRows: 2,
  maxSequenceLength: 48,
  learningRate: 0.005,
  checkpointCadenceSteps: 2,
  precision: "fp32",
  loraRank: 4,
  loraAlpha: 8,
};
const prepared = await execFile(
  python,
  [
    "-c",
    `import json, sys
sys.path.insert(0, sys.argv[1])
from offline import load_offline_dataset, split_offline_dataset, offline_evaluation_protocol
rows, digest = load_offline_dataset(sys.argv[2], "sft", "sft-text")
_, heldout = split_offline_dataset(rows, 2)
print(json.dumps(offline_evaluation_protocol(digest, heldout, json.loads(sys.argv[3]))))`,
    worker,
    NodePath.join(project, "dataset.json"),
    JSON.stringify(baseConfig),
  ],
  { maxBuffer: 1024 * 1024 },
);
const protocol = Schema.decodeUnknownSync(RlEvaluationProtocol)(JSON.parse(prepared.stdout));
const variants = [
  { label: "zero_lr", learningRate: 0 },
  { label: "trained", learningRate: 0.005 },
] as const;
for (const variant of variants) {
  const definition = Schema.decodeUnknownSync(RlProjectExperimentDefinition)({
    version: 1,
    experimentId: variant.label,
    displayName: `Production SFT reference ${variant.label}`,
    description: "Tiny local arithmetic fixture for launcher and evidence integration.",
    mode: "reproducible",
    adapter: "trl",
    method: "sft",
    evaluationClaim: "held-out-loss",
    model: { id: "model", revision: modelRevision, tokenizerRevision: modelRevision },
    dataset: { _tag: "ProjectFile", path: "dataset.json" },
    datasetFormat: "sft-text",
    chatTemplate: { source: "tokenizer", sha256: null },
    verifier: { _tag: "ProjectFile", path: "worker/offline.py" },
    splitPolicy: { train: "held-out-tail-excluded", evaluation: "test" },
    evaluationProtocol: protocol,
    budgets: { maxRuntimeSeconds: 180, maxSteps: 6, maxArtifactBytes: 32 * 1024 * 1024 },
    instrumentationLevel: "standard",
    defaultSeed: 7,
    config: { ...baseConfig, learningRate: variant.learningRate },
  });
  await NodeFSP.writeFile(
    NodePath.join(project, ".t3rl/experiments", `${variant.label}.json`),
    `${JSON.stringify(definition, null, 2)}\n`,
  );
}
await NodeFSP.writeFile(NodePath.join(project, ".gitignore"), "__pycache__/\n");
await execFile("git", ["init", "--quiet", project]);
await execFile("git", ["-C", project, "add", "."]);
await execFile("git", [
  "-C",
  project,
  "-c",
  "user.name=T3RL validation",
  "-c",
  "user.email=validation@example.invalid",
  "commit",
  "--quiet",
  "-m",
  "Freeze local SFT integration fixture",
]);
const sourceRevision = (await execFile("git", ["-C", project, "rev-parse", "HEAD"])).stdout.trim();
const sourceArchive = NodePath.join(state, "project-source.tar");
await execFile("git", [
  "-C",
  project,
  "archive",
  "--format=tar",
  `--output=${sourceArchive}`,
  "HEAD",
]);
const archiveDestination = NodePath.join(output, "sft-production-evidence.tar");
const layer = RlManagerLive.pipe(
  Layer.provide(WorkerSpawnerLive),
  Layer.provide(CapabilitiesLive),
  Layer.provide(Experiments.layerFromDirectory(NodePath.join(worker, "experiments"))),
  Layer.provide(SourceEvidenceLive),
  Layer.provideMerge(ProjectionProjectRepositoryLive),
  Layer.provideMerge(RunStoreLive),
  Layer.provideMerge(ProcessRunner.layer),
  Layer.provideMerge(
    Layer.succeed(
      ServerConfig.ServerConfig,
      ServerConfig.make({ baseDir: state, rlRunsDir: runsDir } as never),
    ),
  ),
  Layer.provideMerge(makeSqlitePersistenceLive(NodePath.join(state, "state.sqlite"))),
  Layer.provideMerge(NodeServices.layer),
);

const trainAndExport = Effect.gen(function* () {
  const store = yield* RunStore;
  const manager = yield* RlManager;
  const projects = yield* ProjectionProjectRepository;
  const fs = yield* FileSystem.FileSystem;
  const at = DateTime.formatIso(yield* DateTime.now);
  const projectId = "production-sft-reference";
  yield* projects.upsert({
    projectId: ProjectId.make(projectId),
    title: "Isolated SFT production reference",
    workspaceRoot: project,
    defaultModelSelection: null,
    defaultThreadEnvMode: null,
    scripts: [],
    createdAt: at,
    updatedAt: at,
    deletedAt: null,
  });
  const runs: RlStudy["runs"][number][] = [];
  const runEvidence: Array<{
    runId: string;
    variant: string;
    trainingSeed: number;
    dataSeedApplied: number;
    lifecycle: RlRunState[];
    completedAt: string | null;
    device: string;
    meanEvaluationLoss: number;
    checkpointSha256: string;
    evaluationSha256: string;
  }> = [];
  const evaluationReferences: { runId: string; artifactId: string }[] = [];
  const additionalArtifacts: { runId: string; artifactId: string }[] = [];
  const waitCompleted = Effect.fn("ProductionReference.waitCompleted")(function* (runId: string) {
    const done = yield* Deferred.make<RlRunState>();
    const lifecycle: RlRunState[] = [];
    const unsubscribe = yield* manager.subscribe({ runId }, (event) => {
      if (event._tag !== "Snapshot" && event._tag !== "Lifecycle") return;
      lifecycle.push(event.summary.state);
      if (isTerminalRlRunState(event.summary.state))
        Deferred.doneUnsafe(done, Effect.succeed(event.summary.state));
    });
    const terminal = yield* Deferred.await(done).pipe(Effect.ensuring(Effect.sync(unsubscribe)));
    const detail = yield* manager.get({ runId });
    if (terminal !== "completed")
      throw new Error(`Real worker failed: ${JSON.stringify(detail.summary)}`);
    return { detail, lifecycle };
  });
  const registerSupplement = Effect.fn("ProductionReference.registerSupplement")(function* (
    runId: string,
    relativePath: string,
    kind: RlArtifactKind,
  ) {
    const identity = yield* computeArtifactIdentity({
      artifactPath: NodePath.join(runsDir, runId, relativePath),
      maxBytes: 32 * 1024 * 1024,
    });
    return yield* store.recordArtifact({
      runId,
      kind,
      relativePath,
      logicalName: relativePath,
      contentType: identity.directory ? "application/vnd.t3rl.directory.v1" : "application/x-tar",
      format: inferArtifactFormat(relativePath, identity.directory),
      producedAt: at,
      ...identity,
    });
  });
  for (const seed of [7, 19, 41]) {
    for (const variant of variants) {
      const seeds = { training: seed, data: 19, evaluationSample: 0, generation: 0 };
      const started = yield* manager.start({
        projectId,
        experimentId: `project__${variant.label}`,
        seed,
        seeds,
        evaluationProtocol: protocol,
        requestId: `production_${variant.label}_${seed}`,
      });
      const runId = started.runId;
      const { detail, lifecycle } = yield* waitCompleted(runId);
      const manifest = detail.manifest;
      if (
        manifest === null ||
        manifest.sourceRevision !== sourceRevision ||
        manifest.sourceDirty !== false ||
        manifest.environmentLock?.lockfileSha256 !== environmentHashes["uv.lock"] ||
        manifest.environmentLock?.files?.length !== 2 ||
        manifest.seeds?.data !== 19 ||
        manifest.seeds.training !== seed ||
        canonicalEvidenceJson(manifest.effectiveConfig.evaluationProtocol) !==
          canonicalEvidenceJson(protocol)
      )
        throw new Error("Production manifest lost source, seed, environment, or protocol evidence");
      const summary = decodeSummary(
        JSON.parse(yield* fs.readFileString(NodePath.join(runsDir, runId, "summary.json"))),
      );
      if (
        summary.dataSeedApplied !== 19 ||
        summary.optimizerSteps !== 6 ||
        !summary.device.startsWith("cuda")
      )
        throw new Error("Real trainer did not apply declared steps/data seed on CUDA");
      const evaluation = detail.artifacts.find((artifact) => artifact.kind === "evaluation");
      const checkpoint = detail.artifacts.find(
        (artifact) =>
          artifact.evidence?._tag === "Checkpoint" && artifact.evidence.checkpointClass === "final",
      );
      if (
        evaluation?.state !== "ready" ||
        evaluation.sha256 == null ||
        checkpoint?.state !== "ready" ||
        checkpoint.sha256 == null ||
        checkpoint.evidence?._tag !== "Checkpoint"
      )
        throw new Error(
          "Production worker did not publish verified evaluation/final checkpoint artifacts",
        );
      const resume = checkpoint.evidence.resumeState;
      if (
        !resume.trainerState ||
        !resume.optimizerState ||
        !resume.schedulerState ||
        !resume.rngState ||
        !resume.datasetCursorState
      )
        throw new Error("Production checkpoint is incomplete");
      evaluationReferences.push({ runId, artifactId: evaluation.artifactId });
      additionalArtifacts.push({ runId, artifactId: checkpoint.artifactId });
      yield* fs.copy(NodePath.join(project, "model"), NodePath.join(runsDir, runId, "base-model"));
      yield* fs.copyFile(sourceArchive, NodePath.join(runsDir, runId, "project-source.tar"));
      const baseModel = yield* registerSupplement(runId, "base-model", "model");
      additionalArtifacts.push({ runId, artifactId: baseModel.artifactId });
      yield* registerSupplement(runId, "project-source.tar", "source");
      runs.push({ runId, variantLabel: variant.label, seeds, state: "completed" });
      runEvidence.push({
        runId,
        variant: variant.label,
        trainingSeed: seed,
        dataSeedApplied: summary.dataSeedApplied,
        lifecycle,
        completedAt: detail.summary.endedAt,
        device: summary.device,
        meanEvaluationLoss: summary.after.eval_loss,
        checkpointSha256: checkpoint.sha256,
        evaluationSha256: evaluation.sha256,
      });
      yield* Console.log(
        JSON.stringify({
          event: "run-completed",
          seed,
          variant: variant.label,
          loss: summary.after.eval_loss,
        }),
      );
    }
  }
  const parentRunId = runs.find(
    (run) => run.variantLabel === "trained" && run.seeds.training === 7,
  )!.runId!;
  const parent = yield* manager.get({ runId: parentRunId });
  const source = parent.artifacts.find(
    (artifact) =>
      artifact.state === "ready" &&
      artifact.evidence?._tag === "Checkpoint" &&
      artifact.evidence.globalStep === 2,
  );
  if (source === undefined) throw new Error("Missing real step-2 checkpoint for production resume");
  const child = yield* manager.resume({
    projectId,
    parentRunId,
    sourceArtifactId: source.artifactId,
    requestId: "production_resume_seed_7",
  });
  const childResult = yield* waitCompleted(child.runId);
  const childManifest = childResult.detail.manifest;
  const childSummary = decodeSummary(
    JSON.parse(yield* fs.readFileString(NodePath.join(runsDir, child.runId, "summary.json"))),
  );
  if (
    canonicalEvidenceJson(childManifest?.seeds) !== canonicalEvidenceJson(parent.manifest?.seeds) ||
    canonicalEvidenceJson(childManifest?.effectiveConfig.evaluationProtocol) !==
      canonicalEvidenceJson(protocol) ||
    childSummary.dataSeedApplied !== 19 ||
    childSummary.optimizerSteps !== 6 ||
    childResult.detail.lineage.edges[0]?.parentRunId !== parentRunId
  )
    throw new Error(
      "Production resume lost the parent's seed set, evaluation protocol, or lineage",
    );
  const adapterPath = Effect.fn("ProductionReference.adapterPath")(function* (runId: string) {
    const detail = yield* manager.get({ runId });
    const adapter = detail.artifacts.find(
      (artifact) => artifact.kind === "adapter" && artifact.state === "ready",
    );
    if (adapter === undefined) throw new Error("Missing published adapter");
    const stored = yield* store.findArtifact({ runId, artifactId: adapter.artifactId });
    if (stored === null) throw new Error("Missing persisted adapter");
    return NodePath.join(runsDir, runId, stored.relativePath, "adapter_model.safetensors");
  });
  const runner = yield* ProcessRunner.ProcessRunner;
  const weights = yield* runner.run({
    command: python,
    args: [
      "-c",
      `import json, sys, torch
from safetensors.torch import load_file
left, right = load_file(sys.argv[1]), load_file(sys.argv[2])
if left.keys() != right.keys(): raise ValueError("Adapter tensor names changed")
maximum = max(float((left[key] - right[key]).abs().max()) for key in left)
if maximum > 1e-6: raise ValueError(f"Resume weights differ: {maximum}")
print(json.dumps({"maxAbsoluteDifference": maximum, "tensorCount": len(left), "tolerance": 1e-6}))`,
      yield* adapterPath(parentRunId),
      yield* adapterPath(child.runId),
    ],
    timeout: "30 seconds",
  });
  if (weights.code !== 0)
    throw new Error(`Production resume differs from uninterrupted training: ${weights.stderr}`);
  const weightComparison = decodeWeightComparison(JSON.parse(weights.stdout));
  const childCheckpoint = childResult.detail.artifacts.find(
    (artifact) =>
      artifact.evidence?._tag === "Checkpoint" && artifact.evidence.checkpointClass === "final",
  );
  if (childCheckpoint === undefined)
    throw new Error("Resumed worker did not publish final checkpoint");
  additionalArtifacts.push({ runId: parentRunId, artifactId: source.artifactId });
  additionalArtifacts.push({ runId: child.runId, artifactId: childCheckpoint.artifactId });
  yield* fs.copy(
    NodePath.join(project, "model"),
    NodePath.join(runsDir, child.runId, "base-model"),
  );
  yield* fs.copyFile(sourceArchive, NodePath.join(runsDir, child.runId, "project-source.tar"));
  const childModel = yield* registerSupplement(child.runId, "base-model", "model");
  additionalArtifacts.push({ runId: child.runId, artifactId: childModel.artifactId });
  yield* registerSupplement(child.runId, "project-source.tar", "source");
  const resumeEvidence = {
    parentRunId,
    childRunId: child.runId,
    sourceStep: 2,
    finalStep: childSummary.optimizerSteps,
    dataSeedApplied: childSummary.dataSeedApplied,
    seedSetPreserved: true,
    evaluationProtocolPreserved: true,
    lifecycle: childResult.lifecycle,
    weightComparison,
  };
  yield* Console.log(JSON.stringify({ event: "resume-completed", ...resumeEvidence }));
  // This script exercises start + terminal subscriptions. Study membership is persisted from
  // the six verified completed runs; createStudy's background scheduler is outside its scope.
  const study: RlStudy = {
    studyId: "study_production_sft_reference",
    projectId,
    state: "completed",
    protocolSha256: protocol.protocolSha256,
    createdAt: at,
    updatedAt: DateTime.formatIso(yield* DateTime.now),
    runs,
    definition: {
      variants: variants.map((variant) => ({
        label: variant.label,
        experimentId: `project__${variant.label}`,
      })),
      seeds: [7, 19, 41].map((training) => ({
        training,
        data: 19,
        evaluationSample: 0,
        generation: 0,
      })),
      maxConcurrency: 1,
      maxRuns: 6,
      evaluationProtocol: protocol,
    },
  };
  yield* store.createStudy(study);
  const comparisonInput = {
    studyId: study.studyId,
    baselineLabel: "zero_lr",
    candidateLabel: "trained",
    metricKey: "eval_after/loss",
    estimator: {
      version: 2 as const,
      statistic: "paired-mean-delta" as const,
      statisticalUnit: "paired-sample-within-run-seed" as const,
      confidenceLevel: 0.95,
      resamplingSeed: 17,
      resampleCount: 10_000,
      missingPairPolicy: "fail" as const,
    },
  };
  const comparison = yield* manager.compareStudy(comparisonInput);
  if (
    comparison.conclusion !== "interval" ||
    comparison.n !== 3 ||
    comparison.excludedRuns.length !== 0
  )
    throw new Error("Production comparison lacks complete verified seed/sample pairs");
  const record = yield* recordResearch({
    projectId,
    requestId: "production-sft-result",
    parentRecordId: null,
    hypothesis:
      "For this tiny fixed arithmetic fixture, six SFT steps with learning rate 0.005 lower held-out loss compared with zero learning rate.",
    evidence: evaluationReferences,
    proposedChange:
      "Compare 0 versus 0.005 learning rate at training seeds 7, 19, 41, fixed independent data seed 19, and identical held-out rows.",
    authorizationReference:
      "User requested bounded real production-launch validation and resolution of review findings.",
    outcome: comparison.interval![1] < 0 ? "supported" : "inconclusive",
    interpretation: `Retrospective fixture result: mean candidate-minus-baseline loss delta ${comparison.pairedDelta}; 95% hierarchical bootstrap interval [${comparison.interval![0]}, ${comparison.interval![1]}].`,
    limitations:
      "Three seeds, two held-out rows, one tiny local model and GPU. This is not preregistered and does not establish generalization. Study membership assembled after real sequential Manager.start calls; background createStudy scheduling and UI are outside this script.",
  });
  const exported = yield* exportEvidence({
    projectId,
    runIds: [...runs.map((run) => run.runId!), child.runId],
    study: comparisonInput,
    recordIds: [record.recordId],
    additionalArtifacts,
  });
  const archive = evidenceExportPath({
    rlRunsDir: runsDir,
    projectId,
    exportId: exported.exportId,
  });
  if (archive === null) throw new Error("Missing export path");
  yield* fs.copyFile(archive, archiveDestination);
  yield* fs.writeFileString(NodePath.join(output, "verify_evidence.py"), EVIDENCE_VERIFIER);
  const sql = yield* SqlClient.SqlClient;
  const counts = yield* sql<{ count: number }>`SELECT COUNT(*) AS count FROM rl_evaluation_samples`;
  if (counts[0]?.count !== 12) throw new Error("Expected twelve ingested real evaluation samples");
  return {
    comparison,
    exported,
    indexedSamples: counts[0].count,
    runEvidence,
    resumeEvidence,
    recordId: record.recordId,
  };
});
const launched = await Effect.runPromise(trainAndExport.pipe(Effect.provide(layer), Effect.scoped));
await NodeFSP.rm(state, { recursive: true, force: true });
const verified = await execFile(python, [
  NodePath.join(output, "verify_evidence.py"),
  archiveDestination,
  "--index-sha256",
  launched.exported.indexSha256,
]);
const reloaded = await execFile(
  python,
  [
    "-c",
    `import json, pathlib, sys, tarfile, tempfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
with tempfile.TemporaryDirectory(prefix="t3rl-production-reload-") as directory:
    root = pathlib.Path(directory)
    with tarfile.open(sys.argv[1], "r:") as archive:
        archive.extractall(root, filter="data")
    loaded = []
    for run in sorted((root / "runs").iterdir()):
        inventory = json.loads((run / "artifacts.json").read_text())
        base = next(item for item in inventory if item["logicalName"] == "base-model")
        adapter = next(item for item in inventory if item["kind"] == "adapter")
        base_path = run / "artifacts" / base["artifactId"] / "base-model"
        adapter_path = next((run / "artifacts" / adapter["artifactId"]).iterdir())
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True), adapter_path, local_files_only=True).eval()
        with torch.no_grad():
            logits = model(**tokenizer("What is 8 + 1 ? Answer :", return_tensors="pt")).logits
        if not torch.isfinite(logits).all():
            raise ValueError("Adapter reload produced non-finite logits")
        loaded.append({"runId": run.name, "finiteLogits": True, "logitShape": list(logits.shape)})
    print(json.dumps({"reloads": loaded, "baseModelRemappedToBundle": True, "hubOffline": True, "localFilesOnly": True}))`,
    archiveDestination,
  ],
  {
    env: {
      ...process.env,
      HF_HUB_OFFLINE: "1",
      CUDA_VISIBLE_DEVICES: "",
      TOKENIZERS_PARALLELISM: "false",
      OMP_NUM_THREADS: "1",
    },
    maxBuffer: 1024 * 1024,
  },
);
const report = {
  version: 1,
  status: "passed",
  scope:
    "Six actual CUDA TRL SFT runs plus one checkpoint resume launched with real RlManager, WorkerSpawnerLive, CapabilitiesLive, project Experiments, SourceEvidenceLive and SQLite",
  archive: ".t3/serious-validation/production-evidence/sft-production-evidence.tar",
  originalProjectAndSqliteRemovedBeforeVerification: true,
  sourceRevision,
  workerHashes,
  modelRevision,
  modelHashes,
  environmentHashes,
  ...launched,
  independentVerification: JSON.parse(verified.stdout),
  independentReload: JSON.parse(reloaded.stdout),
  limitations: [
    "Tiny synthetic fixture: three training seeds, two held-out rows, one GPU; no generalization claim",
    "Retrospective research record; evaluation and generation seeds are inapplicable to deterministic loss",
    "Uses Manager.start with terminal subscriptions; persists study membership after verified runs, without exercising createStudy scheduling",
    "Model and complete project Git archive are supplemental verified artifacts registered by the harness; worker emits checkpoints, adapters, summaries and evaluations",
    "uv lock check and pre-launch setup snapshots use the existing installed TRL environment; no independent environment rebuild",
    "Seven bundled adapters independently reload on CPU with local_files_only and remapped base-model paths; no UI check here",
    "Resume from a completed parent's intermediate step-2 checkpoint matches continuous final adapter tensors on this host; cancellation and cross-hardware equality are outside this script",
  ],
};
const reportPath = NodePath.join(
  workspace,
  "docs/benchmarks/rl-post-training-validation/production-evidence-2026-09-05.json",
);
await NodeFSP.mkdir(NodePath.dirname(reportPath), { recursive: true });
await NodeFSP.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
await Effect.runPromise(
  Console.log(
    JSON.stringify({
      status: report.status,
      indexedSamples: report.indexedSamples,
      seedPairs: report.comparison.n,
      pairedDelta: report.comparison.pairedDelta,
      interval: report.comparison.interval,
      archiveBytes: report.exported.bytes,
      independentReloads: report.independentReload.reloads.length,
      report: NodePath.relative(workspace, reportPath),
    }),
  ),
);
