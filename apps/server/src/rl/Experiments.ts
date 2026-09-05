/**
 * Experiments - resolves a version-controlled experiment definition.
 *
 * Definitions are ordinary JSON checked into the repository, so a run's inputs
 * are reviewable and diffable. Resolution is its own service so the manager can
 * be tested without a filesystem.
 *
 * @module Experiments
 */
import {
  RlExperimentValidationReport,
  RlProjectExperimentDefinition,
  RlRunStartError,
  type RlExperimentSummary,
} from "@t3tools/contracts";
import * as Context from "effect/Context";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Path from "effect/Path";
import * as Schema from "effect/Schema";
import * as ArtifactIdentity from "./ArtifactIdentity.ts";

export const RlExperimentDefinition = Schema.Struct({
  experimentId: Schema.String.check(Schema.isPattern(/^[A-Za-z0-9_-]+$/)),
  displayName: Schema.String.check(Schema.isMaxLength(128)),
  description: Schema.String.check(Schema.isMaxLength(512)),
  runnerId: Schema.String.check(Schema.isPattern(/^[A-Za-z0-9_-]+$/)),
  /** Omitted definitions are legacy protocol-v1 workers. */
  protocolVersion: Schema.optionalKey(
    Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 2 })),
  ),
  instrumentationLevel: Schema.Literals(["minimal", "standard", "deep"]),
  defaultSeed: Schema.Int,
  /** Repository-relative path to the worker entrypoint. */
  entrypoint: Schema.String.check(Schema.isMaxLength(512)),
  scenario: Schema.optional(Schema.String.check(Schema.isMaxLength(64))),
  maxRuntimeSeconds: Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 86_400 })),
  config: Schema.Record(Schema.String, Schema.Unknown).check(Schema.isMaxProperties(128)),
  method: Schema.optionalKey(Schema.Literals(["sft", "dpo", "grpo", "rloo", "ppo"])),
});
export type RlExperimentDefinition = typeof RlExperimentDefinition.Type;

export interface RlResolvedProjectInput {
  readonly role: "definition" | "dataset" | "verifier";
  readonly sourcePath: string;
  readonly snapshotName: string;
  readonly sha256: string;
  readonly bytes: number;
}

export interface ResolvedExperimentDefinition extends RlExperimentDefinition {
  readonly namespace?: "bundled" | "project" | undefined;
  readonly projectInputs?: ReadonlyArray<RlResolvedProjectInput> | undefined;
  readonly floatingInputs?: boolean | undefined;
  readonly externalInputs?:
    | ReadonlyArray<{
        readonly role: "dataset" | "verifier";
        readonly id: string;
        readonly revision: string | null;
      }>
    | undefined;
}

export interface ExperimentsShape {
  readonly resolve: (input: {
    readonly experimentId: string;
    readonly workspaceRoot?: string | undefined;
  }) => Effect.Effect<ResolvedExperimentDefinition, RlRunStartError>;
  readonly validate: (input: {
    readonly experimentId: string;
    readonly workspaceRoot: string;
  }) => Effect.Effect<RlExperimentValidationReport>;
  readonly list: (input?: {
    readonly workspaceRoot?: string | undefined;
  }) => Effect.Effect<ReadonlyArray<RlExperimentSummary>, RlRunStartError>;
}

export class Experiments extends Context.Service<Experiments, ExperimentsShape>()(
  "t3/rl/Experiments",
) {}

const decodeDefinition = Schema.decodeUnknownSync(RlExperimentDefinition);
const decodeProjectDefinition = Schema.decodeUnknownSync(
  Schema.fromJsonString(RlProjectExperimentDefinition),
);
const isRunStartError = Schema.is(RlRunStartError);

const toSummary = (definition: RlExperimentDefinition): RlExperimentSummary => {
  const method =
    definition.method ??
    (typeof definition.config.algorithm === "string" &&
    definition.config.algorithm.toLowerCase() === "grpo"
      ? "grpo"
      : undefined);
  return {
    experimentId: definition.experimentId,
    displayName: definition.displayName,
    description: definition.description,
    runnerId: definition.runnerId,
    defaultSeed: definition.defaultSeed,
    instrumentationLevel: definition.instrumentationLevel,
    config: definition.config,
    ...(method === undefined ? {} : { method }),
  };
};

/**
 * Reads definitions from a directory of JSON files named by experiment id. The
 * id is pattern-checked before it reaches the path join, so it cannot select a
 * file outside the catalog.
 */
export const layerFromDirectories = (directories: ReadonlyArray<string>) =>
  Layer.effect(
    Experiments,
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      let directory = directories[0] ?? "";
      for (const candidate of directories) {
        if (yield* fs.exists(candidate)) {
          directory = candidate;
          break;
        }
      }

      const resolveDefinition = (experimentId: string, publicId = experimentId) =>
        Effect.gen(function* () {
          if (!/^[A-Za-z0-9_-]+$/.test(experimentId)) {
            return yield* new RlRunStartError({
              code: "InvalidExperiment",
              detail: `Invalid experiment id: ${experimentId}`,
            });
          }
          const file = path.join(directory, `${experimentId}.json`);
          const contents = yield* fs.readFileString(file).pipe(
            Effect.mapError(
              () =>
                new RlRunStartError({
                  code: "InvalidExperiment",
                  detail: `No experiment definition at ${file}`,
                }),
            ),
          );
          const definition = yield* Effect.try({
            // @effect-diagnostics-next-line preferSchemaOverJson:off - file on disk.
            try: () => decodeDefinition(JSON.parse(contents)),
            catch: () =>
              new RlRunStartError({
                code: "InvalidExperiment",
                detail: `Malformed experiment definition at ${file}`,
              }),
          });
          if (definition.experimentId !== experimentId) {
            return yield* new RlRunStartError({
              code: "InvalidExperiment",
              detail: `Experiment ${file} declares id ${definition.experimentId}`,
            });
          }

          const workerRoot = path.resolve(directory, "..");
          const entrypoint = path.resolve(directory, definition.entrypoint);
          const relative = path.relative(workerRoot, entrypoint);
          if (relative === "" || relative.startsWith("..") || path.isAbsolute(relative)) {
            return yield* new RlRunStartError({
              code: "InvalidExperiment",
              detail: `Experiment ${file} declares an entrypoint outside the worker bundle`,
            });
          }
          return {
            ...definition,
            experimentId: publicId,
            entrypoint,
            namespace: "bundled" as const,
          };
        });

      const projectFailure = (detail: string) =>
        new RlRunStartError({ code: "InvalidExperiment", detail });

      const confinedInput = (workspaceRoot: string, relativePath: string) =>
        Effect.gen(function* () {
          if (path.isAbsolute(relativePath) || relativePath.includes("\0")) {
            return yield* projectFailure(`Project input must be relative: ${relativePath}`);
          }
          const root = yield* fs
            .realPath(workspaceRoot)
            .pipe(Effect.mapError(() => projectFailure("Project workspace is unavailable")));
          const lexical = path.resolve(root, relativePath);
          const relative = path.relative(root, lexical);
          if (relative === "" || relative.startsWith("..") || path.isAbsolute(relative)) {
            return yield* projectFailure(`Project input escapes the workspace: ${relativePath}`);
          }
          const canonical = yield* fs
            .realPath(lexical)
            .pipe(
              Effect.mapError(() =>
                projectFailure(`Project input does not exist: ${relativePath}`),
              ),
            );
          if (canonical !== lexical) {
            return yield* projectFailure(
              `Project input contains a symlink escape: ${relativePath}`,
            );
          }
          return canonical;
        });

      const resolveProject = (experimentId: string, workspaceRoot: string) =>
        Effect.gen(function* () {
          const rawId = experimentId.startsWith("project__")
            ? experimentId.slice("project__".length)
            : experimentId;
          if (!/^[A-Za-z0-9_-]+$/.test(rawId)) {
            return yield* projectFailure(`Invalid project experiment id: ${experimentId}`);
          }
          const definitionPath = yield* confinedInput(
            workspaceRoot,
            `.t3rl/experiments/${rawId}.json`,
          );
          const definition = yield* fs.readFileString(definitionPath).pipe(
            Effect.flatMap((contents) =>
              Effect.try({
                try: () => decodeProjectDefinition(contents),
                catch: () => projectFailure(`Malformed project experiment at ${definitionPath}`),
              }),
            ),
            Effect.mapError((error) =>
              isRunStartError(error)
                ? error
                : projectFailure(`Unable to read project experiment at ${definitionPath}`),
            ),
          );
          if (definition.experimentId !== rawId) {
            return yield* projectFailure(
              `Project experiment file declares id ${definition.experimentId}, expected ${rawId}`,
            );
          }
          if (
            definition.mode === "reproducible" &&
            (definition.model.revision === null ||
              definition.model.tokenizerRevision === null ||
              (definition.dataset._tag === "External" && definition.dataset.revision === null) ||
              (definition.verifier._tag === "External" && definition.verifier.revision === null))
          ) {
            return yield* projectFailure(
              "Reproducible project experiments require pinned model, tokenizer, dataset, and verifier revisions",
            );
          }
          const expectedFormat =
            definition.method === "sft"
              ? ["sft-text", "sft-conversation"]
              : definition.method === "dpo"
                ? ["dpo-preference"]
                : ["rlvr-prompt-answer"];
          if (!expectedFormat.includes(definition.datasetFormat)) {
            return yield* projectFailure(
              `${definition.method} requires datasetFormat ${expectedFormat.join(" or ")}`,
            );
          }
          const expectedClaim =
            definition.method === "sft"
              ? "held-out-loss"
              : definition.method === "dpo"
                ? "preference-accuracy"
                : "verifier-pass-rate";
          if (definition.evaluationClaim !== expectedClaim) {
            return yield* projectFailure(
              `${definition.method} requires evaluationClaim ${expectedClaim}`,
            );
          }
          const localReferences = [
            { role: "definition" as const, path: `.t3rl/experiments/${rawId}.json` },
            ...(definition.dataset._tag === "ProjectFile"
              ? [{ role: "dataset" as const, path: definition.dataset.path }]
              : []),
            ...(definition.verifier._tag === "ProjectFile"
              ? [{ role: "verifier" as const, path: definition.verifier.path }]
              : []),
          ];
          const projectInputs = yield* Effect.forEach(
            localReferences,
            ({ role, path: inputPath }) =>
              Effect.gen(function* () {
                const sourcePath = yield* confinedInput(workspaceRoot, inputPath);
                const identity = yield* ArtifactIdentity.computeArtifactIdentity({
                  artifactPath: sourcePath,
                  maxBytes: definition.budgets.maxArtifactBytes,
                }).pipe(
                  Effect.provideService(FileSystem.FileSystem, fs),
                  Effect.provideService(Path.Path, path),
                  Effect.mapError((error) =>
                    projectFailure(
                      Schema.is(ArtifactIdentity.ArtifactIdentityError)(error)
                        ? error.detail
                        : `Unable to hash project input: ${inputPath}`,
                    ),
                  ),
                );
                return {
                  role,
                  sourcePath,
                  snapshotName: `${role}-${path.basename(sourcePath)}`,
                  sha256: identity.sha256,
                  bytes: identity.bytes,
                };
              }),
          );
          const entrypoint = path.resolve(
            directory,
            definition.method === "grpo"
              ? definition.adapter === "trl"
                ? "../trl_worker.py"
                : "../axolotl_worker.py"
              : definition.adapter === "trl"
                ? "../trl_offline_worker.py"
                : "../axolotl_offline_worker.py",
          );
          return {
            experimentId: `project__${rawId}`,
            displayName: definition.displayName,
            description: definition.description,
            runnerId: definition.adapter,
            protocolVersion: 2,
            instrumentationLevel: definition.instrumentationLevel,
            defaultSeed: definition.defaultSeed,
            method: definition.method,
            entrypoint,
            maxRuntimeSeconds: definition.budgets.maxRuntimeSeconds,
            config: {
              ...definition.config,
              modelId: definition.model.id,
              modelRevision: definition.model.revision,
              tokenizerRevision: definition.model.tokenizerRevision,
              maxSteps: definition.budgets.maxSteps,
              method: definition.method,
              datasetFormat: definition.datasetFormat,
              evaluationClaim: definition.evaluationClaim,
              chatTemplate: definition.chatTemplate,
              projectDatasetPath: null,
              projectVerifierPath: null,
            },
            namespace: "project" as const,
            projectInputs,
            floatingInputs:
              definition.model.revision === null ||
              definition.model.tokenizerRevision === null ||
              (definition.dataset._tag === "External" && definition.dataset.revision === null) ||
              (definition.verifier._tag === "External" && definition.verifier.revision === null),
            externalInputs: [
              ...(definition.dataset._tag === "External"
                ? [
                    {
                      role: "dataset" as const,
                      id: definition.dataset.id,
                      revision: definition.dataset.revision,
                    },
                  ]
                : []),
              ...(definition.verifier._tag === "External"
                ? [
                    {
                      role: "verifier" as const,
                      id: definition.verifier.id,
                      revision: definition.verifier.revision,
                    },
                  ]
                : []),
            ],
          } satisfies ResolvedExperimentDefinition;
        });

      const validateProject: ExperimentsShape["validate"] = (input) =>
        resolveProject(input.experimentId, input.workspaceRoot).pipe(
          Effect.map((definition) => {
            const floating = definition.floatingInputs === true;
            return {
              experimentId: definition.experimentId,
              namespace: "project" as const,
              valid: true,
              issues: floating
                ? [
                    {
                      severity: "warning" as const,
                      code: "floating-revision",
                      message:
                        "Exploratory experiment uses a floating model, tokenizer, dataset, or verifier revision.",
                      path: null,
                    },
                  ]
                : [],
              resolvedInputs: [
                ...(definition.projectInputs ?? []).map((entry) => ({
                  role: entry.role,
                  logicalName: entry.snapshotName,
                  sha256: entry.sha256,
                  bytes: entry.bytes,
                  externalId: null,
                  revision: null,
                })),
                ...(definition.externalInputs ?? []).map((entry) => ({
                  role: entry.role,
                  logicalName: entry.id,
                  sha256: null,
                  bytes: null,
                  externalId: entry.id,
                  revision: entry.revision,
                })),
              ],
              supportedOperations: ["start", "resume", "warm-start", "study"] as const,
            };
          }),
          Effect.catch((error) =>
            Effect.succeed({
              experimentId: input.experimentId,
              namespace: "project" as const,
              valid: false,
              issues: [
                {
                  severity: "error" as const,
                  code: "invalid-definition",
                  message: error.detail,
                  path: null,
                },
              ],
              resolvedInputs: [],
              supportedOperations: [],
            }),
          ),
        );

      return Experiments.of({
        resolve: (input) => {
          if (input.experimentId.startsWith("project__")) {
            return input.workspaceRoot === undefined
              ? Effect.fail(projectFailure("Project experiment resolution requires a workspace"))
              : resolveProject(input.experimentId, input.workspaceRoot);
          }
          const rawId = input.experimentId.startsWith("bundled__")
            ? input.experimentId.slice("bundled__".length)
            : input.experimentId;
          if (input.workspaceRoot === undefined || input.experimentId.startsWith("bundled__")) {
            return resolveDefinition(rawId, input.experimentId);
          }
          return fs
            .exists(path.join(input.workspaceRoot, ".t3rl", "experiments", `${rawId}.json`))
            .pipe(
              Effect.flatMap((projectExists) =>
                projectExists
                  ? Effect.fail(
                      projectFailure(
                        `Ambiguous experiment id ${rawId}; use bundled__${rawId} or project__${rawId}`,
                      ),
                    )
                  : resolveDefinition(rawId, rawId),
              ),
              Effect.mapError((error) =>
                isRunStartError(error)
                  ? error
                  : projectFailure(`Unable to resolve experiment ${rawId}`),
              ),
            );
        },
        validate: validateProject,
        list: () =>
          Effect.gen(function* () {
            const files = (yield* fs.readDirectory(directory))
              .filter((entry) => entry.endsWith(".json"))
              .sort();
            const entries = yield* Effect.forEach(files, (file) =>
              resolveDefinition(
                file.slice(0, -".json".length),
                `bundled__${file.slice(0, -".json".length)}`,
              ).pipe(
                Effect.map(toSummary),
                Effect.tapError((error) =>
                  Effect.logWarning("skipping invalid RL experiment definition", {
                    file,
                    detail: error.detail,
                  }),
                ),
                Effect.option,
              ),
            );
            return entries.flatMap((entry) => (entry._tag === "Some" ? [entry.value] : []));
          }).pipe(
            Effect.mapError((error) =>
              isRunStartError(error)
                ? error
                : new RlRunStartError({
                    code: "RunnerUnavailable",
                    detail: `Unable to read the experiment catalog at ${directory}`,
                  }),
            ),
          ),
      });
    }),
  );

export const layerFromDirectory = (directory: string) => layerFromDirectories([directory]);

/** Fixed catalog for tests and for callers that already hold their definitions. */
export const layerFromRecord = (definitions: Record<string, RlExperimentDefinition>) =>
  Layer.succeed(
    Experiments,
    Experiments.of({
      resolve: (input) => {
        const definition = definitions[input.experimentId];
        return definition === undefined
          ? Effect.fail(
              new RlRunStartError({
                code: "InvalidExperiment",
                detail: `Unknown experiment: ${input.experimentId}`,
              }),
            )
          : Effect.succeed(definition);
      },
      validate: (input) =>
        Effect.succeed({
          experimentId: input.experimentId,
          namespace: "project",
          valid: false,
          issues: [
            {
              severity: "error",
              code: "unsupported",
              message: "Project validation is unavailable in this fixed catalog",
              path: null,
            },
          ],
          resolvedInputs: [],
          supportedOperations: [],
        }),
      list: () =>
        Effect.succeed(
          Object.values(definitions)
            .sort((left, right) => left.experimentId.localeCompare(right.experimentId))
            .map(toSummary),
        ),
    }),
  );
