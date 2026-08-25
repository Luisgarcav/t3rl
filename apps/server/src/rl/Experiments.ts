/**
 * Experiments - resolves a version-controlled experiment definition.
 *
 * Definitions are ordinary JSON checked into the repository, so a run's inputs
 * are reviewable and diffable. Resolution is its own service so the manager can
 * be tested without a filesystem.
 *
 * @module Experiments
 */
import { RlRunStartError } from "@t3tools/contracts";
import * as Context from "effect/Context";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Path from "effect/Path";
import * as Schema from "effect/Schema";

export const RlExperimentDefinition = Schema.Struct({
  experimentId: Schema.String.check(Schema.isPattern(/^[A-Za-z0-9_-]+$/)),
  runnerId: Schema.String.check(Schema.isPattern(/^[A-Za-z0-9_-]+$/)),
  instrumentationLevel: Schema.Literals(["minimal", "standard", "deep"]),
  defaultSeed: Schema.Int,
  /** Repository-relative path to the worker entrypoint. */
  entrypoint: Schema.String.check(Schema.isMaxLength(512)),
  scenario: Schema.String.check(Schema.isMaxLength(64)),
  config: Schema.Record(Schema.String, Schema.Unknown),
});
export type RlExperimentDefinition = typeof RlExperimentDefinition.Type;

export interface ExperimentsShape {
  readonly resolve: (input: {
    readonly experimentId: string;
  }) => Effect.Effect<RlExperimentDefinition, RlRunStartError>;
}

export class Experiments extends Context.Service<Experiments, ExperimentsShape>()(
  "t3/rl/Experiments",
) {}

const decodeDefinition = Schema.decodeUnknownSync(RlExperimentDefinition);

/**
 * Reads definitions from a directory of JSON files named by experiment id. The
 * id is pattern-checked before it reaches the path join, so it cannot select a
 * file outside the catalog.
 */
export const layerFromDirectory = (directory: string) =>
  Layer.effect(
    Experiments,
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;

      return Experiments.of({
        resolve: (input) =>
          Effect.gen(function* () {
            if (!/^[A-Za-z0-9_-]+$/.test(input.experimentId)) {
              return yield* new RlRunStartError({
                code: "RunnerUnavailable",
                detail: `Invalid experiment id: ${input.experimentId}`,
              });
            }
            const file = path.join(directory, `${input.experimentId}.json`);
            const contents = yield* fs.readFileString(file).pipe(
              Effect.mapError(
                () =>
                  new RlRunStartError({
                    code: "RunnerUnavailable",
                    detail: `No experiment definition at ${file}`,
                  }),
              ),
            );
            const definition = yield* Effect.try({
              // @effect-diagnostics-next-line preferSchemaOverJson:off - file on disk.
              try: () => decodeDefinition(JSON.parse(contents)),
              catch: () =>
                new RlRunStartError({
                  code: "RunnerUnavailable",
                  detail: `Malformed experiment definition at ${file}`,
                }),
            });
            if (definition.experimentId !== input.experimentId) {
              return yield* new RlRunStartError({
                code: "RunnerUnavailable",
                detail: `Experiment ${file} declares id ${definition.experimentId}`,
              });
            }
            return definition;
          }),
      });
    }),
  );

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
                code: "RunnerUnavailable",
                detail: `Unknown experiment: ${input.experimentId}`,
              }),
            )
          : Effect.succeed(definition);
      },
    }),
  );
