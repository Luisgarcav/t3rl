/**
 * Experiments - resolves a version-controlled experiment definition.
 *
 * Definitions are ordinary JSON checked into the repository, so a run's inputs
 * are reviewable and diffable. Resolution is its own service so the manager can
 * be tested without a filesystem.
 *
 * @module Experiments
 */
import { RlRunStartError, type RlExperimentSummary } from "@t3tools/contracts";
import * as Context from "effect/Context";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Path from "effect/Path";
import * as Schema from "effect/Schema";

export const RlExperimentDefinition = Schema.Struct({
  experimentId: Schema.String.check(Schema.isPattern(/^[A-Za-z0-9_-]+$/)),
  displayName: Schema.String.check(Schema.isMaxLength(128)),
  description: Schema.String.check(Schema.isMaxLength(512)),
  runnerId: Schema.String.check(Schema.isPattern(/^[A-Za-z0-9_-]+$/)),
  instrumentationLevel: Schema.Literals(["minimal", "standard", "deep"]),
  defaultSeed: Schema.Int,
  /** Repository-relative path to the worker entrypoint. */
  entrypoint: Schema.String.check(Schema.isMaxLength(512)),
  scenario: Schema.optional(Schema.String.check(Schema.isMaxLength(64))),
  maxRuntimeSeconds: Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 86_400 })),
  config: Schema.Record(Schema.String, Schema.Unknown).check(Schema.isMaxProperties(128)),
});
export type RlExperimentDefinition = typeof RlExperimentDefinition.Type;

export interface ExperimentsShape {
  readonly resolve: (input: {
    readonly experimentId: string;
  }) => Effect.Effect<RlExperimentDefinition, RlRunStartError>;
  readonly list: () => Effect.Effect<ReadonlyArray<RlExperimentSummary>, RlRunStartError>;
}

export class Experiments extends Context.Service<Experiments, ExperimentsShape>()(
  "t3/rl/Experiments",
) {}

const decodeDefinition = Schema.decodeUnknownSync(RlExperimentDefinition);
const isRunStartError = Schema.is(RlRunStartError);

const toSummary = (definition: RlExperimentDefinition): RlExperimentSummary => ({
  experimentId: definition.experimentId,
  displayName: definition.displayName,
  description: definition.description,
  runnerId: definition.runnerId,
  defaultSeed: definition.defaultSeed,
  instrumentationLevel: definition.instrumentationLevel,
  config: definition.config,
});

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

      const resolveDefinition = (experimentId: string) =>
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
          return { ...definition, entrypoint };
        });

      return Experiments.of({
        resolve: (input) => resolveDefinition(input.experimentId),
        list: () =>
          Effect.gen(function* () {
            const files = (yield* fs.readDirectory(directory))
              .filter((entry) => entry.endsWith(".json"))
              .sort();
            const entries = yield* Effect.forEach(files, (file) =>
              resolveDefinition(file.slice(0, -".json".length)).pipe(
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
      list: () =>
        Effect.succeed(
          Object.values(definitions)
            .sort((left, right) => left.experimentId.localeCompare(right.experimentId))
            .map(toSummary),
        ),
    }),
  );
