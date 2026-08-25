// @effect-diagnostics nodeBuiltinImport:off
import * as NodePath from "node:path";

const RUN_ID_PATTERN = /^[A-Za-z0-9_-]+$/;

/**
 * Root directory owned by one run. A run id that is not a plain identifier is
 * refused outright rather than normalized, because normalizing an attacker's
 * id is how traversal bugs get reintroduced.
 */
export function runDirectory(input: {
  readonly rlRunsDir: string;
  readonly runId: string;
}): string | null {
  if (!RUN_ID_PATTERN.test(input.runId)) {
    return null;
  }
  return NodePath.join(NodePath.resolve(input.rlRunsDir), input.runId);
}

/**
 * Resolves a run-relative artifact path, or null when the result would leave
 * the run directory. Callers treat null as ArtifactNotFound; the distinction
 * between "missing" and "refused" is deliberately not exposed to clients.
 */
export function resolveArtifactPath(input: {
  readonly rlRunsDir: string;
  readonly runId: string;
  readonly relativePath: string;
}): string | null {
  const runRoot = runDirectory(input);
  if (runRoot === null) {
    return null;
  }
  if (input.relativePath.length === 0 || input.relativePath.includes("\0")) {
    return null;
  }
  if (NodePath.isAbsolute(input.relativePath)) {
    return null;
  }

  const normalized = NodePath.normalize(input.relativePath).replace(/^[/\\]+/, "");
  if (normalized.length === 0 || normalized.startsWith("..")) {
    return null;
  }

  const filePath = NodePath.resolve(NodePath.join(runRoot, normalized));
  if (!filePath.startsWith(`${runRoot}${NodePath.sep}`)) {
    return null;
  }
  return filePath;
}
