// @effect-diagnostics nodeBuiltinImport:off - stable filesystem identifiers shared with asset delivery.
import * as NodeCrypto from "node:crypto";
import * as NodePath from "node:path";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";

export function evidenceProjectDirectory(rlRunsDir: string, projectId: string): string {
  const projectKey = NodeCrypto.createHash("sha256").update(projectId).digest("hex");
  return NodePath.resolve(rlRunsDir, "..", "rl-evidence", projectKey);
}

/** Resolve configured parent aliases (including macOS /var) before checking project storage. */
export const resolveEvidenceProjectDirectory = Effect.fn("RlEvidence.resolveProjectDirectory")(
  function* (rlRunsDir: string, projectId: string) {
    const fs = yield* FileSystem.FileSystem;
    const parent = yield* fs.realPath(NodePath.resolve(rlRunsDir, ".."));
    return evidenceProjectDirectory(NodePath.join(parent, NodePath.basename(rlRunsDir)), projectId);
  },
);

export function evidenceExportPath(input: {
  readonly rlRunsDir: string;
  readonly projectId: string;
  readonly exportId: string;
}): string | null {
  if (!/^export_[a-f0-9]{48}$/.test(input.exportId)) return null;
  return NodePath.join(
    evidenceProjectDirectory(input.rlRunsDir, input.projectId),
    "exports",
    `${input.exportId}.tar`,
  );
}
