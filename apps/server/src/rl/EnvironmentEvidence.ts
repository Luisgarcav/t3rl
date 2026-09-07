import { RlRunStartError, type RlEnvironmentLock } from "@t3tools/contracts";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Path from "effect/Path";
import * as Schema from "effect/Schema";

import { computeArtifactIdentity } from "./ArtifactIdentity.ts";
const isRunStartError = Schema.is(RlRunStartError);

/** Copies the setup bytes into the run before a worker can start. It never syncs an environment. */
export const snapshotEnvironment = Effect.fn("RlEnvironment.snapshot")(
  function* (lock: RlEnvironmentLock | null | undefined, runRoot: string) {
    if (lock == null) return null;
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const directory = path.join(runRoot, "inputs", "environment");
    yield* fs.makeDirectory(directory, { recursive: true });
    const files: Array<{ name: "uv.lock" | "pyproject.toml"; sha256: string; bytes: number }> = [];
    for (const name of ["pyproject.toml", "uv.lock"] as const) {
      const source = name === "uv.lock" ? lock.lockfilePath : path.join(lock.projectPath, name);
      const identity = yield* computeArtifactIdentity({
        artifactPath: source,
        maxBytes: 4 * 1024 * 1024,
      });
      if (identity.directory || (name === "uv.lock" && identity.sha256 !== lock.lockfileSha256)) {
        return yield* new RlRunStartError({
          code: "RunnerUnavailable",
          detail:
            "Environment setup changed after capability validation; validate the locked environment again.",
        });
      }
      const target = path.join(directory, name);
      yield* fs.copyFile(source, target);
      const copied = yield* computeArtifactIdentity({
        artifactPath: target,
        maxBytes: identity.bytes,
      });
      if (copied.sha256 !== identity.sha256)
        return yield* new RlRunStartError({
          code: "RunnerUnavailable",
          detail: "Environment setup changed while it was being snapshotted.",
        });
      files.push({ name, sha256: copied.sha256, bytes: copied.bytes });
    }
    return { ...lock, files };
  },
  Effect.mapError((error) =>
    isRunStartError(error)
      ? error
      : new RlRunStartError({
          code: "RunnerUnavailable",
          detail: "Unable to retain the declared locked environment setup files.",
        }),
  ),
);
