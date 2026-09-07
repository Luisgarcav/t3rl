#!/usr/bin/env node
/** Imports six captured real TRL workers into isolated services, then verifies a portable bundle. */
// @effect-diagnostics nodeBuiltinImport:off - standalone reference validation owns only its temporary files and child process.
// @effect-diagnostics preferSchemaOverJson:off - reports and canonical artifact identities are persisted evidence.
import * as NodeChildProcess from "node:child_process";
import * as NodeCrypto from "node:crypto";
import * as NodeFSP from "node:fs/promises";
import * as NodePath from "node:path";
import * as NodeUtil from "node:util";

import * as NodeServices from "@effect/platform-node/NodeServices";
import {
  RlEvaluationProtocol,
  type RlArtifactEvidence,
  type RlArtifactKind,
  type RlStudy,
} from "@t3tools/contracts";
import * as Console from "effect/Console";
import * as DateTime from "effect/DateTime";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Schema from "effect/Schema";
import * as SqlClient from "effect/unstable/sql/SqlClient";

import * as ServerConfig from "../src/config.ts";
import { makeSqlitePersistenceLive } from "../src/persistence/Layers/Sqlite.ts";
import { computeArtifactIdentity, inferArtifactFormat } from "../src/rl/ArtifactIdentity.ts";
import { Capabilities } from "../src/rl/Capabilities.ts";
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
import * as SourceEvidence from "../src/rl/SourceEvidence.ts";
import { decodeWorkerLine } from "../src/rl/WorkerProtocol.ts";
import { WorkerSpawner } from "../src/rl/WorkerSpawner.ts";

const workspace = NodePath.resolve(import.meta.dirname, "../../..");
const captured = NodePath.join(workspace, ".t3/serious-validation/trl-gpu");
const output = NodePath.join(workspace, ".t3/serious-validation/reference-evidence");
const python = NodePath.join(workspace, ".venv-trl-validation/bin/python");
const sourceSnapshot = NodePath.join(captured, "source-snapshot");
const execFile = NodeUtil.promisify(NodeChildProcess.execFile);
const sha256 = (value: string | Uint8Array) =>
  NodeCrypto.createHash("sha256").update(value).digest("hex");
const decodeProtocol = Schema.decodeUnknownSync(RlEvaluationProtocol);
const decodeDependencies = Schema.decodeUnknownSync(Schema.Record(Schema.String, Schema.Unknown));
const decodeHashes = Schema.decodeUnknownSync(Schema.Record(Schema.String, Schema.String));
const decodeValidation = Schema.decodeUnknownSync(
  Schema.Struct({
    status: Schema.Literal("passed"),
    backend: Schema.Literal("trl"),
    device: Schema.Literal("cuda"),
    environmentLockSha256: Schema.String,
  }),
);
const validation = decodeValidation(
  JSON.parse(await NodeFSP.readFile(NodePath.join(captured, "validation-report.json"), "utf8")),
);
const hashes = decodeHashes(
  JSON.parse(await NodeFSP.readFile(NodePath.join(sourceSnapshot, "source-hashes.json"), "utf8")),
);
for (const [name, digest] of Object.entries(hashes)) {
  if (sha256(await NodeFSP.readFile(NodePath.join(sourceSnapshot, name))) !== digest)
    throw new Error(`Captured source hash mismatch: ${name}`);
}
const environmentSource = NodePath.join(workspace, "python/environments/trl");
if (
  sha256(await NodeFSP.readFile(NodePath.join(environmentSource, "uv.lock"))) !==
  validation.environmentLockSha256
)
  throw new Error("Current environment lock differs from the captured workers");
await NodeFSP.mkdir(output, { recursive: true });
const state = await NodeFSP.mkdtemp(NodePath.join(output, "import-"));
const runsDir = NodePath.join(state, "runs");
const archiveDestination = NodePath.join(output, "sft-three-seed-evidence.tar");

const layer = RlManagerLive.pipe(
  Layer.provide(
    Layer.succeed(
      WorkerSpawner,
      WorkerSpawner.of({ spawn: () => Effect.die("Reference import must not launch training") }),
    ),
  ),
  Layer.provide(
    Layer.succeed(
      Capabilities,
      Capabilities.of({
        report: () => Effect.succeed({ runners: [], experiments: [] }),
        resolvePython: () => Effect.die("unused"),
        resolveRunner: () => Effect.die("unused"),
      }),
    ),
  ),
  Layer.provide(Experiments.layerFromRecord({})),
  Layer.provide(
    SourceEvidence.layerFromResolver(() =>
      Effect.die("Reference import does not read project state"),
    ),
  ),
  Layer.provideMerge(
    Layer.succeed(
      ServerConfig.ServerConfig,
      ServerConfig.make({ baseDir: state, rlRunsDir: runsDir } as never),
    ),
  ),
  Layer.provideMerge(RunStoreLive),
  Layer.provideMerge(makeSqlitePersistenceLive(NodePath.join(state, "state.sqlite"))),
  Layer.provideMerge(NodeServices.layer),
);

const importAndExport = Effect.gen(function* () {
  const store = yield* RunStore;
  const manager = yield* RlManager;
  const fs = yield* FileSystem.FileSystem;
  const at = DateTime.formatIso(yield* DateTime.now);
  const projectId = "reference-sft-ablation";
  const runs: RlStudy["runs"][number][] = [];
  const evaluationReferences: { runId: string; artifactId: string }[] = [];
  const additionalArtifacts: { runId: string; artifactId: string }[] = [];
  let protocol: ReturnType<typeof decodeProtocol> | undefined;
  const register = Effect.fn("Reference.register")(function* (
    runId: string,
    relativePath: string,
    kind: RlArtifactKind,
    evidence?: RlArtifactEvidence,
  ) {
    const artifactPath = NodePath.join(runsDir, runId, relativePath);
    const identity = yield* computeArtifactIdentity({ artifactPath, maxBytes: 32 * 1024 * 1024 });
    return yield* store.recordArtifact({
      runId,
      kind,
      relativePath,
      logicalName: relativePath,
      contentType: identity.directory
        ? "application/vnd.t3rl.directory.v1"
        : relativePath.endsWith(".json")
          ? "application/json"
          : "text/plain",
      format: inferArtifactFormat(relativePath, identity.directory),
      producedAt: at,
      ...identity,
      ...(evidence === undefined ? {} : { evidence }),
    });
  });
  for (const seed of [7, 19, 41]) {
    for (const variant of [
      { label: "zero_lr", directory: "zero-learning-rate" },
      { label: "trained", directory: "continuous" },
    ]) {
      const runId = `reference_${variant.label}_${seed}`;
      const original = NodePath.join(captured, `sft-seed-${seed}`, variant.directory);
      const target = NodePath.join(runsDir, runId);
      const parsed = (yield* fs.readFileString(NodePath.join(original, "protocol.ndjson")))
        .trim()
        .split("\n")
        .map(decodeWorkerLine);
      if (parsed.some((entry) => entry._tag !== "Message"))
        throw new Error(`Malformed captured worker: ${runId}`);
      const messages = parsed.flatMap((entry) => (entry._tag === "Message" ? [entry.message] : []));
      const hello = messages[0];
      const manifest = messages.find((entry) => entry._tag === "Manifest");
      const done = messages.at(-1);
      if (
        hello?._tag !== "Hello" ||
        manifest?._tag !== "Manifest" ||
        manifest.model === undefined ||
        done?._tag !== "Done" ||
        !done.success ||
        messages.some((entry) => entry._tag === "Error")
      )
        throw new Error(`Captured run is not a valid completed worker: ${runId}`);
      const currentProtocol = decodeProtocol(manifest.values.evaluationProtocol);
      if (
        protocol !== undefined &&
        canonicalEvidenceJson(protocol) !== canonicalEvidenceJson(currentProtocol)
      )
        throw new Error("Captured protocols differ across the six runs");
      protocol = currentProtocol;
      const dependencies = decodeDependencies(manifest.values.dependencies);
      const compatibility = messages.flatMap((message) =>
        message._tag === "Artifact" && message.evidence !== undefined
          ? [message.evidence.compatibility]
          : [],
      )[0];
      if (
        compatibility === undefined ||
        compatibility.environmentLockSha256 !== validation.environmentLockSha256
      ) {
        throw new Error("Captured checkpoint lacks matching environment evidence");
      }
      const evaluator = {
        sources: {
          "offline.py": hashes["offline.py"],
          "trl_offline_worker.py": hashes["trl_offline_worker.py"],
        },
        settings: Object.fromEntries(
          [
            "method",
            "datasetFormat",
            "maxSequenceLength",
            "modelId",
            "modelRevision",
            "tokenizerRevision",
            "precision",
          ].map((key) => [key, manifest.values[key]]),
        ),
        dependencies: Object.fromEntries(
          ["torch", "trl", "peft", "transformers"].map((key) => [key, dependencies[key]]),
        ),
      };
      if (sha256(canonicalEvidenceJson(evaluator)) !== protocol.verifierSha256)
        throw new Error("Source snapshot is not the evaluator used by the captured run");
      yield* fs.makeDirectory(target, { recursive: true });
      yield* fs.copy(original, target, { overwrite: true });
      yield* fs.copy(NodePath.join(captured, "model"), NodePath.join(target, "base-model"));
      yield* fs.copy(sourceSnapshot, NodePath.join(target, "worker-source"));
      yield* fs.copyFile(
        NodePath.join(captured, `sft-seed-${seed}`, "dataset.json"),
        NodePath.join(target, "dataset.json"),
      );
      if (
        sha256(yield* fs.readFile(NodePath.join(target, "dataset.json"))) !==
        protocol.datasetFingerprint
      )
        throw new Error("Captured dataset fingerprint mismatch");
      yield* fs.makeDirectory(NodePath.join(target, "environment"));
      for (const name of ["pyproject.toml", "uv.lock"])
        yield* fs.copyFile(
          NodePath.join(environmentSource, name),
          NodePath.join(target, "environment", name),
        );
      const seeds = { training: seed, data: seed, evaluationSample: 0, generation: 0 };
      yield* store.insertRequested({
        runId,
        projectId,
        experimentId: variant.label,
        requestedAt: at,
      });
      yield* store.setManifest({
        runId,
        manifest: {
          experimentId: variant.label,
          runnerId: hello.runner,
          runnerVersion: hello.runnerVersion,
          protocolVersion: hello.protocol,
          seed,
          seeds,
          model: manifest.model,
          effectiveConfig: {
            ...manifest.values,
            evaluationProtocol: protocol,
            referenceImport: {
              origin: "captured-real-worker-output",
              timestamps: "import-time; original wall-clock lifecycle unavailable",
              seedSemantics:
                "training and data seeds equal actual captured trainer seed; evaluation/generation seeds inapplicable to deterministic loss",
              sourceScope: "captured worker modules",
              environmentSetup:
                "uv.lock hash verified against capture; pyproject.toml captured at import",
              workerConfig: manifest.values,
            },
          },
          sourceRevision: null,
          sourceDirty: null,
          pythonExecutable: python,
          pythonVersion: String(dependencies.python),
          environmentFingerprint: compatibility.environmentFingerprint,
          environmentLock: {
            projectPath: NodePath.join(target, "environment"),
            lockfilePath: NodePath.join(target, "environment", "uv.lock"),
            lockfileSha256: validation.environmentLockSha256,
            pythonExecutable: python,
            pythonVersion: String(dependencies.python),
            platform: String(dependencies.platform),
            framework: compatibility.framework,
            pytorchVersion: String(dependencies.torch),
            cudaAvailable: true,
            cudaRuntime: String(dependencies.cudaRuntime),
            cudaDeviceCount: Number(dependencies.cudaDeviceCount),
            driverVersion: null,
          },
          instrumentationLevel: "standard",
          hardwareSummary: "NVIDIA GeForce RTX 4060 Laptop GPU; imported completed CUDA run",
          ...(manifest.checkpointPolicy === undefined
            ? {}
            : { checkpointPolicy: manifest.checkpointPolicy }),
        },
      });
      let seq = 0;
      for (const message of messages) {
        if (message._tag === "Metrics")
          yield* store.appendMetrics({ runId, seq: seq++, batch: message.batch, at });
        if (message._tag !== "Artifact") continue;
        if (message.evidence?._tag === "Checkpoint") {
          const state = message.evidence.resumeState;
          if (
            !state.trainerState ||
            !state.optimizerState ||
            !state.schedulerState ||
            !state.rngState ||
            !state.datasetCursorState
          ) {
            throw new Error("Captured checkpoint is incomplete");
          }
          for (const name of state.stateFiles) {
            if (!(yield* fs.exists(NodePath.join(target, message.path, name))))
              throw new Error("Captured checkpoint state file is missing");
          }
        }
        const artifact = yield* register(runId, message.path, message.kind, message.evidence);
        if (message.kind === "evaluation")
          evaluationReferences.push({ runId, artifactId: artifact.artifactId });
        if (message.evidence?._tag === "Checkpoint" && message.evidence.checkpointClass === "final")
          additionalArtifacts.push({ runId, artifactId: artifact.artifactId });
      }
      const model = yield* register(runId, "base-model", "model");
      additionalArtifacts.push({ runId, artifactId: model.artifactId });
      yield* register(runId, "worker-source", "source");
      yield* register(runId, "dataset.json", "dataset");
      yield* register(runId, "environment", "config");
      yield* register(runId, "protocol.ndjson", "config");
      yield* store.updateState({ runId, state: "completed", at });
      runs.push({ runId, variantLabel: variant.label, seeds, state: "completed" });
    }
  }
  if (protocol === undefined) throw new Error("No imported protocol");
  const study: RlStudy = {
    studyId: "study_real_sft_reference",
    projectId,
    state: "completed",
    protocolSha256: protocol.protocolSha256,
    createdAt: at,
    updatedAt: at,
    definition: {
      variants: [
        { label: "zero_lr", experimentId: "zero_lr" },
        { label: "trained", experimentId: "trained" },
      ],
      seeds: [7, 19, 41].map((seed) => ({
        training: seed,
        data: seed,
        evaluationSample: 0,
        generation: 0,
      })),
      maxConcurrency: 1,
      maxRuns: 6,
      evaluationProtocol: protocol,
    },
    runs,
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
    throw new Error("Reference comparison lacks complete verified seed/sample pairs");
  const record = yield* recordResearch({
    projectId,
    requestId: "retrospective-sft-reference",
    parentRecordId: null,
    hypothesis:
      "On this fixed tiny model and held-out dataset, six SFT optimizer steps with learning rate 0.005 lower loss compared with zero learning rate.",
    evidence: evaluationReferences,
    proposedChange:
      "Compare learning rate 0.005 with 0 using training/data seeds 7, 19, 41 and identical held-out sample IDs.",
    authorizationReference:
      "User requested bounded actual-framework validation and resolution of the review findings.",
    outcome: "inconclusive",
    interpretation:
      "Retrospective reference case: captured workers were imported after training. This record was not preregistered; the paired loss interval describes only this tiny synthetic fixture.",
    limitations:
      "Three seeds, two held-out rows, one local GPU and tiny local model. No production launch through Manager, UI verification, cross-hardware equality, or generalization claim.",
  });
  const result = yield* recordResearch({
    ...record.input,
    requestId: "retrospective-sft-result",
    parentRecordId: record.recordId,
    outcome: comparison.interval![1] < 0 ? "supported" : "inconclusive",
    interpretation: `Within this fixed fixture the candidate-minus-baseline mean loss delta is ${comparison.pairedDelta}, with a 95% hierarchical bootstrap interval [${comparison.interval![0]}, ${comparison.interval![1]}]. This does not establish generalization or a prospective research result.`,
  });
  const exported = yield* exportEvidence({
    projectId,
    runIds: runs.map((run) => run.runId!),
    study: comparisonInput,
    recordIds: [result.recordId],
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
  return {
    comparison,
    exported,
    indexedSamples: counts[0]!.count,
    recordIds: [record.recordId, result.recordId],
  };
});

let imported: Effect.Success<typeof importAndExport>;
try {
  imported = await Effect.runPromise(importAndExport.pipe(Effect.provide(layer), Effect.scoped));
} finally {
  await NodeFSP.rm(state, { recursive: true, force: true });
}
const verified = await execFile(
  python,
  [
    NodePath.join(output, "verify_evidence.py"),
    archiveDestination,
    "--index-sha256",
    imported.exported.indexSha256,
  ],
  {
    env: {
      ...process.env,
      HF_HUB_OFFLINE: "1",
      CUDA_VISIBLE_DEVICES: "",
      TOKENIZERS_PARALLELISM: "false",
    },
  },
);
const reloader = `import json, pathlib, sys, tarfile, tempfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
with tempfile.TemporaryDirectory(prefix="t3rl-independent-reload-") as directory:
    root = pathlib.Path(directory)
    with tarfile.open(sys.argv[1], "r:") as archive:
        archive.extractall(root, filter="data")
    loaded = []
    for run in sorted((root / "runs").iterdir()):
        inventory = json.loads((run / "artifacts.json").read_text())
        base = next(item for item in inventory if item["logicalName"] == "base-model")
        adapter = next(item for item in inventory if item["kind"] == "adapter")
        base_path = run / "artifacts" / base["artifactId"] / "base-model"
        adapter_root = run / "artifacts" / adapter["artifactId"]
        adapter_path = next(adapter_root.iterdir())
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True), adapter_path, local_files_only=True).eval()
        with torch.no_grad():
            logits = model(**tokenizer("What is 8 + 1 ? Answer :", return_tensors="pt")).logits
        if not torch.isfinite(logits).all():
            raise ValueError("Adapter reload produced non-finite logits")
        loaded.append({"runId": run.name, "finiteLogits": True, "logitShape": list(logits.shape)})
    print(json.dumps({"reloads": loaded, "baseModelRemappedToBundle": True, "hubOffline": True, "localFilesOnly": True}))
`;
const reloaded = await execFile(python, ["-c", reloader, archiveDestination], {
  env: {
    ...process.env,
    HF_HUB_OFFLINE: "1",
    CUDA_VISIBLE_DEVICES: "",
    TOKENIZERS_PARALLELISM: "false",
    OMP_NUM_THREADS: "1",
  },
  maxBuffer: 1024 * 1024,
});
const report = {
  version: 1,
  status: "passed",
  scope:
    "Imports six already captured actual CUDA TRL SFT workers into real RunStore and RlManager; no production worker-launch or browser claim",
  source: ".t3/serious-validation/trl-gpu",
  archive: ".t3/serious-validation/reference-evidence/sft-three-seed-evidence.tar",
  importedStateRemovedBeforeVerification: true,
  evaluatorSourceHashesVerifiedAgainstCapturedProtocols: true,
  capturedSourceHashes: hashes,
  capturedEnvironmentLockSha256: validation.environmentLockSha256,
  ...imported,
  independentVerification: JSON.parse(verified.stdout),
  independentReload: JSON.parse(reloaded.stdout),
  limitations: [
    "Retrospective tiny synthetic reference; three training seeds and two held-out samples",
    "Reload uses existing locked Python dependencies on CPU, not an independently rebuilt environment",
    "uv.lock hash matches worker capture; pyproject.toml was captured at import time",
    "Model/tokenizer paths explicitly remapped to extracted bundle; original imported SQLite/state removed",
    "Full trainer checkpoints included; trainer resume is validated by the source lifecycle harness, not rerun by this script",
  ],
};
const reportPath = NodePath.join(
  workspace,
  "docs/benchmarks/rl-post-training-validation/reference-evidence-2026-09-05.json",
);
await NodeFSP.mkdir(NodePath.dirname(reportPath), { recursive: true });
await NodeFSP.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
await Effect.runPromise(
  Console.log(
    JSON.stringify({
      status: report.status,
      indexedSamples: report.indexedSamples,
      seedPairs: imported.comparison.n,
      pairedDelta: imported.comparison.pairedDelta,
      interval: imported.comparison.interval,
      archiveBytes: imported.exported.bytes,
      independentReloads: report.independentReload.reloads.length,
      report: NodePath.relative(workspace, reportPath),
    }),
  ),
);
