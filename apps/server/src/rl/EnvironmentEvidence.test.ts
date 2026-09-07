import * as NodeServices from "@effect/platform-node/NodeServices";
import { expect, it } from "@effect/vitest";
import { type RlEnvironmentLock } from "@t3tools/contracts";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Path from "effect/Path";

import { computeArtifactIdentity } from "./ArtifactIdentity.ts";
import { snapshotEnvironment } from "./EnvironmentEvidence.ts";

it.effect(
  "retains both setup files independently of later environment edits and refuses stale locks",
  () =>
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-environment-evidence-" });
      const lockfilePath = path.join(root, "uv.lock");
      yield* fs.writeFileString(lockfilePath, "version = 1\n");
      yield* fs.writeFileString(
        path.join(root, "pyproject.toml"),
        '[project]\nname = "fixture"\nversion = "1"\n',
      );
      const identity = yield* computeArtifactIdentity({
        artifactPath: lockfilePath,
        maxBytes: 1024,
      });
      const lock: RlEnvironmentLock = {
        projectPath: root,
        lockfilePath,
        lockfileSha256: identity.sha256,
        pythonExecutable: "python3",
        pythonVersion: "3.12.0",
        platform: "linux",
        framework: { id: "trl", version: "1.10.0" },
        pytorchVersion: null,
        cudaAvailable: false,
        cudaRuntime: null,
        cudaDeviceCount: 0,
        driverVersion: null,
      };
      const run = path.join(root, "run");
      const snapshot = yield* snapshotEnvironment(lock, run);
      expect(snapshot?.files?.map((file) => file.name)).toEqual(["pyproject.toml", "uv.lock"]);
      yield* fs.writeFileString(lockfilePath, "version = 2\n");
      yield* fs.writeFileString(path.join(root, "pyproject.toml"), "changed");
      expect(yield* fs.readFileString(path.join(run, "inputs", "environment", "uv.lock"))).toBe(
        "version = 1\n",
      );
      expect(
        yield* fs.readFileString(path.join(run, "inputs", "environment", "pyproject.toml")),
      ).toContain('name = "fixture"');
      expect(
        yield* snapshotEnvironment(lock, path.join(root, "new-run")).pipe(Effect.flip),
      ).toMatchObject({ code: "RunnerUnavailable" });
    }).pipe(Effect.provide(NodeServices.layer), Effect.scoped),
);
