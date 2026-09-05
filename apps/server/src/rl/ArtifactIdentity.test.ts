// @effect-diagnostics nodeBuiltinImport:off - test oracle for the production digest.
import * as NodeCrypto from "node:crypto";

import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Path from "effect/Path";
import { describe } from "vite-plus/test";

import { computeArtifactIdentity, verifyArtifactIdentity } from "./ArtifactIdentity.ts";

describe("artifact identity", () => {
  it.effect("streams a file hash and detects later corruption", () =>
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-artifact-" });
      const file = `${directory}/summary.json`;
      yield* fs.writeFileString(file, "evidence");

      const identity = yield* computeArtifactIdentity({ artifactPath: file, maxBytes: 1024 });
      assert.strictEqual(
        identity.sha256,
        NodeCrypto.createHash("sha256").update("evidence").digest("hex"),
      );
      assert.strictEqual(identity.bytes, 8);
      assert.strictEqual(identity.fileCount, 1);
      assert.isTrue(
        yield* verifyArtifactIdentity({
          artifactPath: file,
          expectedSha256: identity.sha256,
          maxBytes: 1024,
        }),
      );

      yield* fs.writeFileString(file, "Evidence");
      assert.isFalse(
        yield* verifyArtifactIdentity({
          artifactPath: file,
          expectedSha256: identity.sha256,
          maxBytes: 1024,
        }),
      );
    }).pipe(Effect.provide(NodeServices.layer), Effect.scoped),
  );

  it.effect("hashes a directory through a canonical sorted content manifest", () =>
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-artifact-dir-" });
      yield* fs.makeDirectory(path.join(directory, "nested"));
      yield* fs.writeFileString(path.join(directory, "z.txt"), "last");
      yield* fs.writeFileString(path.join(directory, "nested", "a.txt"), "first");

      const identity = yield* computeArtifactIdentity({ artifactPath: directory, maxBytes: 1024 });
      assert.deepStrictEqual(
        identity.contentManifest.map((entry) => entry.path),
        ["nested/a.txt", "z.txt"],
      );
      assert.strictEqual(identity.fileCount, 2);
      assert.strictEqual(identity.bytes, 9);
      assert.match(identity.sha256, /^[0-9a-f]{64}$/);
    }).pipe(Effect.provide(NodeServices.layer), Effect.scoped),
  );

  it.effect("rejects a symlink inside a directory artifact", () =>
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-artifact-link-" });
      yield* fs.writeFileString(path.join(directory, "target.txt"), "target");
      yield* fs.symlink(path.join(directory, "target.txt"), path.join(directory, "alias.txt"));

      const exit = yield* Effect.exit(
        computeArtifactIdentity({ artifactPath: directory, maxBytes: 1024 }),
      );
      assert.isTrue(exit._tag === "Failure");
    }).pipe(Effect.provide(NodeServices.layer), Effect.scoped),
  );
});
