/** Resolves the project root and captures bounded Git evidence for a run manifest. */
import { ProjectId, RlRunStartError } from "@t3tools/contracts";
import * as Context from "effect/Context";
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import * as Option from "effect/Option";

import { ProjectionProjectRepository } from "../persistence/Services/ProjectionProjects.ts";
import * as ProcessRunner from "../processRunner.ts";

export interface ProjectSourceEvidence {
  readonly workspaceRoot: string;
  readonly sourceRevision: string | null;
  readonly sourceDirty: boolean | null;
}

export interface SourceEvidenceShape {
  readonly resolve: (projectId: string) => Effect.Effect<ProjectSourceEvidence, RlRunStartError>;
}

export class SourceEvidence extends Context.Service<SourceEvidence, SourceEvidenceShape>()(
  "t3/rl/SourceEvidence",
) {}

const makeSourceEvidence = Effect.gen(function* () {
  const projects = yield* ProjectionProjectRepository;
  const runner = yield* ProcessRunner.ProcessRunner;

  const runGit = (workspaceRoot: string, args: ReadonlyArray<string>) =>
    runner
      .run({
        command: "git",
        args,
        cwd: workspaceRoot,
        timeout: "10 seconds",
        maxOutputBytes: 1024 * 1024,
        outputMode: "truncate",
      })
      .pipe(Effect.orElseSucceed(() => null));

  const resolve: SourceEvidenceShape["resolve"] = (projectId) =>
    Effect.gen(function* () {
      const project = yield* projects
        .getById({ projectId: ProjectId.make(projectId) })
        .pipe(Effect.orDie);
      if (Option.isNone(project) || project.value.deletedAt !== null) {
        return yield* new RlRunStartError({
          code: "RunnerUnavailable",
          detail: `Project is unavailable: ${projectId}`,
        });
      }

      const revisionResult = yield* runGit(project.value.workspaceRoot, ["rev-parse", "HEAD"]);
      const statusResult = yield* runGit(project.value.workspaceRoot, [
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
      ]);
      const revision =
        revisionResult !== null && revisionResult.code === 0
          ? revisionResult.stdout.trim().slice(0, 64) || null
          : null;
      const sourceDirty =
        statusResult !== null && statusResult.code === 0
          ? statusResult.stdout.trim().length > 0 || statusResult.stdoutTruncated
          : null;

      return {
        workspaceRoot: project.value.workspaceRoot,
        sourceRevision: revision,
        sourceDirty,
      };
    });

  return SourceEvidence.of({ resolve });
});

export const SourceEvidenceLive = Layer.effect(SourceEvidence, makeSourceEvidence);

/** Small seam for manager tests and embedders that already resolved their project. */
export const layerFromResolver = (resolve: SourceEvidenceShape["resolve"]) =>
  Layer.succeed(SourceEvidence, SourceEvidence.of({ resolve }));
