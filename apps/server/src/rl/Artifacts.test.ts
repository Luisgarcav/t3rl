import * as NodePath from "node:path";
import { describe, expect, it } from "vite-plus/test";

import * as Artifacts from "./Artifacts.ts";

const rlRunsDir = NodePath.resolve("/state/rl");

describe("resolveArtifactPath", () => {
  it("resolves a plain relative path inside the run", () => {
    const resolved = Artifacts.resolveArtifactPath({
      rlRunsDir,
      runId: "run_01",
      relativePath: "artifacts/summary.json",
    });
    expect(resolved).toBe(NodePath.join(rlRunsDir, "run_01", "artifacts", "summary.json"));
  });

  it("rejects a traversal escape", () => {
    expect(
      Artifacts.resolveArtifactPath({
        rlRunsDir,
        runId: "run_01",
        relativePath: "../run_02/model.zip",
      }),
    ).toBeNull();
  });

  it("rejects an absolute path", () => {
    expect(
      Artifacts.resolveArtifactPath({ rlRunsDir, runId: "run_01", relativePath: "/etc/passwd" }),
    ).toBeNull();
  });

  it("rejects a null byte", () => {
    expect(
      Artifacts.resolveArtifactPath({ rlRunsDir, runId: "run_01", relativePath: "log\0.txt" }),
    ).toBeNull();
  });

  it("rejects an empty path", () => {
    expect(
      Artifacts.resolveArtifactPath({ rlRunsDir, runId: "run_01", relativePath: "" }),
    ).toBeNull();
  });

  it("rejects a run id that is itself a traversal", () => {
    expect(
      Artifacts.resolveArtifactPath({ rlRunsDir, runId: "../..", relativePath: "log.txt" }),
    ).toBeNull();
    expect(Artifacts.runDirectory({ rlRunsDir, runId: "../escape" })).toBeNull();
  });

  it("rejects a path that normalizes back to the run root itself", () => {
    expect(
      Artifacts.resolveArtifactPath({ rlRunsDir, runId: "run_01", relativePath: "." }),
    ).toBeNull();
  });
});
