// @effect-diagnostics nodeBuiltinImport:off - incremental hashing must not buffer large artifacts.
import * as NodeCrypto from "node:crypto";

import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Path from "effect/Path";
import * as Schema from "effect/Schema";
import * as Stream from "effect/Stream";

export const MAX_ARTIFACT_FILES = 4096;

export interface ArtifactContentEntry {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

export interface ArtifactIdentity {
  readonly bytes: number;
  readonly fileCount: number;
  readonly sha256: string;
  readonly contentManifest: ReadonlyArray<ArtifactContentEntry>;
  readonly directory: boolean;
}

export class ArtifactIdentityError extends Schema.TaggedErrorClass<ArtifactIdentityError>()(
  "ArtifactIdentityError",
  { detail: Schema.String },
) {}

const fail = (detail: string) => new ArtifactIdentityError({ detail });

const hashFile = Effect.fn("ArtifactIdentity.hashFile")(function* (file: string) {
  const fs = yield* FileSystem.FileSystem;
  const hash = NodeCrypto.createHash("sha256");
  let bytes = 0;
  yield* fs.stream(file).pipe(
    Stream.runForEach((chunk) =>
      Effect.sync(() => {
        hash.update(chunk);
        bytes += chunk.byteLength;
      }),
    ),
  );
  return { bytes, sha256: hash.digest("hex") };
});

export function canonicalArtifactManifest(entries: ReadonlyArray<ArtifactContentEntry>): string {
  return JSON.stringify({ version: 1, files: entries });
}

/**
 * Hashes a file directly or a directory through a sorted, path-normalized
 * content manifest. Every byte is streamed and symlinked entries are refused,
 * so the recorded identity cannot silently point outside the published tree.
 */
export const computeArtifactIdentity = Effect.fn("ArtifactIdentity.compute")(function* (input: {
  readonly artifactPath: string;
  readonly maxBytes: number;
  readonly maxFiles?: number | undefined;
}) {
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const canonicalRoot = yield* fs.realPath(input.artifactPath);
  const rootInfo = yield* fs.stat(canonicalRoot);
  const maxFiles = input.maxFiles ?? MAX_ARTIFACT_FILES;

  if (rootInfo.type === "File") {
    const hashed = yield* hashFile(canonicalRoot);
    if (hashed.bytes > input.maxBytes) {
      return yield* fail(`artifact exceeded ${input.maxBytes} bytes`);
    }
    return {
      ...hashed,
      fileCount: 1,
      contentManifest: [
        { path: path.basename(canonicalRoot), bytes: hashed.bytes, sha256: hashed.sha256 },
      ],
      directory: false,
    } satisfies ArtifactIdentity;
  }

  if (rootInfo.type !== "Directory") {
    return yield* fail("artifact was neither a regular file nor a directory");
  }

  const names = (yield* fs.readDirectory(canonicalRoot, { recursive: true })).sort();
  const contentManifest: ArtifactContentEntry[] = [];
  let totalBytes = 0;
  for (const name of names) {
    const lexical = path.resolve(path.join(canonicalRoot, name));
    const canonical = yield* fs.realPath(lexical);
    if (canonical !== lexical) {
      return yield* fail(`artifact contains a symbolic link: ${name}`);
    }
    const relative = path.relative(canonicalRoot, canonical).replaceAll("\\", "/");
    if (relative === "" || relative.startsWith("../") || path.isAbsolute(relative)) {
      return yield* fail(`artifact entry escaped its root: ${name}`);
    }
    const info = yield* fs.stat(canonical);
    if (info.type === "Directory") continue;
    if (info.type !== "File") {
      return yield* fail(`artifact contains an unsupported entry: ${relative}`);
    }
    if (contentManifest.length >= maxFiles) {
      return yield* fail(`artifact exceeded ${maxFiles} files`);
    }
    const hashed = yield* hashFile(canonical);
    totalBytes += hashed.bytes;
    if (!Number.isSafeInteger(totalBytes) || totalBytes > input.maxBytes) {
      return yield* fail(`artifact exceeded ${input.maxBytes} bytes`);
    }
    contentManifest.push({ path: relative, bytes: hashed.bytes, sha256: hashed.sha256 });
  }

  const sha256 = NodeCrypto.createHash("sha256")
    .update(canonicalArtifactManifest(contentManifest), "utf8")
    .digest("hex");
  return {
    bytes: totalBytes,
    fileCount: contentManifest.length,
    sha256,
    contentManifest,
    directory: true,
  } satisfies ArtifactIdentity;
});

export const verifyArtifactIdentity = Effect.fn("ArtifactIdentity.verify")(function* (input: {
  readonly artifactPath: string;
  readonly expectedSha256: string;
  readonly maxBytes: number;
  readonly maxFiles?: number | undefined;
}) {
  const identity = yield* computeArtifactIdentity(input);
  return NodeCrypto.timingSafeEqual(
    Buffer.from(identity.sha256, "hex"),
    Buffer.from(input.expectedSha256, "hex"),
  );
});

export function inferArtifactFormat(artifactPath: string, directory: boolean): string {
  if (directory) return "directory-v1";
  const name = artifactPath.split(/[\\/]/).at(-1) ?? "";
  const dot = name.lastIndexOf(".");
  return dot > 0 && dot < name.length - 1
    ? name
        .slice(dot + 1)
        .toLowerCase()
        .slice(0, 64)
    : "binary";
}
