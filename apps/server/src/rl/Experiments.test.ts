// @effect-diagnostics preferSchemaOverJson:off - checked-in catalog fixture.
import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Path from "effect/Path";
import { describe } from "vite-plus/test";

import * as Experiments from "./Experiments.ts";

const definition = (entrypoint: string) =>
  JSON.stringify({
    experimentId: "cartpole-ppo",
    displayName: "PPO on CartPole",
    description: "Test definition",
    runnerId: "stable-baselines3",
    instrumentationLevel: "standard",
    defaultSeed: 7,
    entrypoint,
    maxRuntimeSeconds: 60,
    config: { environment: "CartPole-v1" },
  });

const projectDefinition = (
  datasetPath: string,
  mode: "reproducible" | "exploratory" = "reproducible",
) =>
  JSON.stringify({
    version: 1,
    experimentId: "custom-grpo",
    displayName: "Custom GRPO",
    description: "Project fixture",
    mode,
    adapter: "trl",
    method: "grpo",
    evaluationClaim: "verifier-pass-rate",
    model: {
      id: "fixture/model",
      revision: mode === "reproducible" ? "model-rev" : null,
      tokenizerRevision: mode === "reproducible" ? "tokenizer-rev" : null,
    },
    dataset: { _tag: "ProjectFile", path: datasetPath },
    datasetFormat: "rlvr-prompt-answer",
    chatTemplate: { source: "tokenizer", sha256: null },
    verifier: { _tag: "ProjectFile", path: ".t3rl/verifiers/exact.py" },
    splitPolicy: { train: "train", evaluation: "test" },
    evaluationProtocol: {
      version: 1,
      protocolSha256: "a".repeat(64),
      datasetFingerprint: "fixture-v1",
      split: "test",
      sampleIds: ["one"],
      generationSeedPolicy: "fixed-per-sample",
      decoding: {},
      verifierSha256: "b".repeat(64),
    },
    budgets: { maxRuntimeSeconds: 60, maxSteps: 10, maxArtifactBytes: 1024 * 1024 },
    instrumentationLevel: "standard",
    defaultSeed: 7,
    config: {},
  });

describe("Experiments", () => {
  it.effect("lists definitions and resolves a bundle-confined absolute entrypoint", () =>
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-catalog-" });
      const catalog = path.join(root, "experiments");
      const worker = path.join(root, "sb3_worker.py");
      yield* fs.makeDirectory(catalog, { recursive: true });
      yield* fs.writeFileString(worker, "# worker\n");
      yield* fs.writeFileString(
        path.join(catalog, "cartpole-ppo.json"),
        definition("../sb3_worker.py"),
      );
      yield* fs.writeFileString(path.join(catalog, "broken.json"), "not json");

      yield* Effect.gen(function* () {
        const experiments = yield* Experiments.Experiments;
        const resolved = yield* experiments.resolve({ experimentId: "cartpole-ppo" });
        assert.strictEqual(resolved.entrypoint, worker);
        const listed = yield* experiments.list();
        assert.strictEqual(listed.length, 1);
        assert.strictEqual(listed[0]?.experimentId, "bundled__cartpole-ppo");
        assert.strictEqual(listed[0]?.defaultSeed, 7);
      }).pipe(Effect.provide(Experiments.layerFromDirectory(catalog)));
    }).pipe(Effect.provide(NodeServices.layer)),
  );

  it.effect("rejects an entrypoint outside the worker bundle", () =>
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-catalog-escape-" });
      const catalog = path.join(root, "worker", "experiments");
      yield* fs.makeDirectory(catalog, { recursive: true });
      yield* fs.writeFileString(
        path.join(catalog, "cartpole-ppo.json"),
        definition("../../../outside.py"),
      );

      const error = yield* Effect.gen(function* () {
        const experiments = yield* Experiments.Experiments;
        return yield* Effect.flip(experiments.resolve({ experimentId: "cartpole-ppo" }));
      }).pipe(Effect.provide(Experiments.layerFromDirectory(catalog)));
      assert.strictEqual(error.code, "InvalidExperiment");
    }).pipe(Effect.provide(NodeServices.layer)),
  );

  it.effect("refreshes and hashes project definitions without restarting", () =>
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-project-catalog-" });
      const catalog = path.join(root, "bundle", "experiments");
      const workspace = path.join(root, "workspace");
      yield* fs.makeDirectory(catalog, { recursive: true });
      yield* fs.makeDirectory(path.join(workspace, ".t3rl", "experiments"), { recursive: true });
      yield* fs.makeDirectory(path.join(workspace, ".t3rl", "verifiers"), { recursive: true });
      yield* fs.writeFileString(path.join(root, "bundle", "trl_worker.py"), "# worker\n");
      yield* fs.writeFileString(path.join(workspace, "dataset.jsonl"), '{"x":1}\n');
      yield* fs.writeFileString(
        path.join(workspace, ".t3rl", "verifiers", "exact.py"),
        "def verify(): return True\n",
      );
      const definitionPath = path.join(workspace, ".t3rl", "experiments", "custom-grpo.json");
      yield* fs.writeFileString(definitionPath, projectDefinition("dataset.jsonl"));

      yield* Effect.gen(function* () {
        const experiments = yield* Experiments.Experiments;
        const first = yield* experiments.resolve({
          experimentId: "project__custom-grpo",
          workspaceRoot: workspace,
        });
        assert.strictEqual(first.runnerId, "trl");
        assert.strictEqual(first.projectInputs?.length, 3);
        const firstHash = first.projectInputs?.find((entry) => entry.role === "dataset")?.sha256;
        yield* fs.writeFileString(path.join(workspace, "dataset.jsonl"), '{"x":2}\n');
        const refreshed = yield* experiments.resolve({
          experimentId: "project__custom-grpo",
          workspaceRoot: workspace,
        });
        assert.notStrictEqual(
          refreshed.projectInputs?.find((entry) => entry.role === "dataset")?.sha256,
          firstHash,
        );
      }).pipe(Effect.provide(Experiments.layerFromDirectory(catalog)));
    }).pipe(Effect.provide(NodeServices.layer)),
  );

  it.effect("rejects floating reproducible inputs and project symlink escapes", () =>
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-project-escape-" });
      const catalog = path.join(root, "bundle", "experiments");
      const workspace = path.join(root, "workspace");
      yield* fs.makeDirectory(catalog, { recursive: true });
      yield* fs.makeDirectory(path.join(workspace, ".t3rl", "experiments"), { recursive: true });
      yield* fs.makeDirectory(path.join(workspace, ".t3rl", "verifiers"), { recursive: true });
      yield* fs.writeFileString(path.join(root, "outside.jsonl"), "outside\n");
      yield* fs.symlink(path.join(root, "outside.jsonl"), path.join(workspace, "dataset.jsonl"));
      yield* fs.writeFileString(
        path.join(workspace, ".t3rl", "verifiers", "exact.py"),
        "# verifier\n",
      );
      const definitionPath = path.join(workspace, ".t3rl", "experiments", "custom-grpo.json");
      yield* fs.writeFileString(definitionPath, projectDefinition("dataset.jsonl"));
      yield* Effect.gen(function* () {
        const experiments = yield* Experiments.Experiments;
        const escaped = yield* Effect.flip(
          experiments.resolve({ experimentId: "project__custom-grpo", workspaceRoot: workspace }),
        );
        assert.match(escaped.detail, /symlink/);
        yield* fs.writeFileString(
          definitionPath,
          projectDefinition("dataset.jsonl", "reproducible").replace('"model-rev"', "null"),
        );
        const floating = yield* Effect.flip(
          experiments.resolve({ experimentId: "project__custom-grpo", workspaceRoot: workspace }),
        );
        assert.match(floating.detail, /pinned/);
      }).pipe(Effect.provide(Experiments.layerFromDirectory(catalog)));
    }).pipe(Effect.provide(NodeServices.layer)),
  );

  it.effect("maps the same project schema to TRL SFT and Axolotl DPO allowlisted adapters", () =>
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-offline-methods-" });
      const catalog = path.join(root, "bundle", "experiments");
      const workspace = path.join(root, "workspace");
      yield* fs.makeDirectory(catalog, { recursive: true });
      yield* fs.makeDirectory(path.join(workspace, ".t3rl", "experiments"), { recursive: true });
      yield* fs.makeDirectory(path.join(workspace, ".t3rl", "verifiers"), { recursive: true });
      yield* fs.writeFileString(path.join(root, "bundle", "trl_offline_worker.py"), "# trl\n");
      yield* fs.writeFileString(
        path.join(root, "bundle", "axolotl_offline_worker.py"),
        "# axolotl\n",
      );
      yield* fs.writeFileString(
        path.join(workspace, "dataset.jsonl"),
        '{"text":"a"}\n{"text":"b"}\n',
      );
      yield* fs.writeFileString(
        path.join(workspace, ".t3rl", "verifiers", "exact.py"),
        "# verifier\n",
      );
      const definitionPath = path.join(workspace, ".t3rl", "experiments", "custom-grpo.json");
      const base = JSON.parse(projectDefinition("dataset.jsonl"));
      yield* fs.writeFileString(
        definitionPath,
        JSON.stringify({
          ...base,
          method: "sft",
          evaluationClaim: "held-out-loss",
          datasetFormat: "sft-text",
        }),
      );
      yield* Effect.gen(function* () {
        const experiments = yield* Experiments.Experiments;
        const sft = yield* experiments.resolve({
          experimentId: "project__custom-grpo",
          workspaceRoot: workspace,
        });
        assert.strictEqual(sft.method, "sft");
        assert.match(sft.entrypoint, /trl_offline_worker\.py$/);
        yield* fs.writeFileString(
          definitionPath,
          JSON.stringify({
            ...base,
            adapter: "axolotl",
            method: "dpo",
            evaluationClaim: "preference-accuracy",
            datasetFormat: "dpo-preference",
          }),
        );
        const dpo = yield* experiments.resolve({
          experimentId: "project__custom-grpo",
          workspaceRoot: workspace,
        });
        assert.strictEqual(dpo.method, "dpo");
        assert.match(dpo.entrypoint, /axolotl_offline_worker\.py$/);
      }).pipe(Effect.provide(Experiments.layerFromDirectory(catalog)));
    }).pipe(Effect.provide(NodeServices.layer)),
  );
});
