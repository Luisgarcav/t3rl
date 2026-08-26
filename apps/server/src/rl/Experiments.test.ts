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
        assert.strictEqual(listed[0]?.experimentId, "cartpole-ppo");
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
});
