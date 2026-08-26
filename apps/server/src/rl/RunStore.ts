/**
 * RunStore - durable T3RL run state.
 *
 * Lifecycle facts live in `rl_runs` and are updated in place. Metrics live in
 * `rl_run_metrics` and artifacts in `rl_run_artifacts`, so training telemetry
 * has no path into the orchestration event log.
 *
 * @module RunStore
 */
import {
  RlArtifactMetadata,
  RlMetricBatch,
  RlResolvedManifest,
  RlRunNotFoundError,
  RlRunSummary,
  isTerminalRlRunState,
  type RlArtifactKind,
  type RlErrorCode,
  type RlRunState,
} from "@t3tools/contracts";
import * as Context from "effect/Context";
import * as Crypto from "effect/Crypto";
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import * as Schema from "effect/Schema";
import * as SqlClient from "effect/unstable/sql/SqlClient";

import { PersistenceSqlError, toPersistenceSqlError } from "../persistence/Errors.ts";

export class RlManifestAlreadySetError extends Schema.TaggedErrorClass<RlManifestAlreadySetError>()(
  "RlManifestAlreadySetError",
  { runId: Schema.String },
) {
  override get message() {
    return `RL run already has a resolved manifest: ${this.runId}`;
  }
}

export type RunStoreError = PersistenceSqlError;

export interface InsertRequestedInput {
  readonly runId: string;
  readonly projectId: string;
  readonly experimentId: string;
  readonly requestedAt: string;
  readonly requestId?: string | undefined;
}

export interface UpdateStateInput {
  readonly runId: string;
  readonly state: RlRunState;
  readonly at: string;
  readonly errorCode?: RlErrorCode | undefined;
  readonly errorMessage?: string | undefined;
  readonly workerPid?: number | undefined;
}

export interface AppendMetricsInput {
  readonly runId: string;
  readonly seq: number;
  readonly batch: RlMetricBatch;
  readonly at: string;
}

export interface RecordArtifactInput {
  readonly runId: string;
  readonly kind: RlArtifactKind;
  readonly relativePath: string;
  readonly bytes: number;
  readonly contentType: string;
  readonly producedAt: string;
}

export interface RunStoreShape {
  readonly insertRequested: (
    input: InsertRequestedInput,
  ) => Effect.Effect<{ readonly runId: string; readonly inserted: boolean }, RunStoreError>;
  readonly updateState: (input: UpdateStateInput) => Effect.Effect<void, RunStoreError>;
  /** Writes the manifest exactly once; a second write is refused. */
  readonly setManifest: (input: {
    readonly runId: string;
    readonly manifest: RlResolvedManifest;
  }) => Effect.Effect<void, RunStoreError | RlManifestAlreadySetError>;
  readonly appendMetrics: (input: AppendMetricsInput) => Effect.Effect<void, RunStoreError>;
  readonly recordArtifact: (
    input: RecordArtifactInput,
  ) => Effect.Effect<RlArtifactMetadata, RunStoreError>;
  readonly listRuns: (input: {
    readonly projectId: string;
    readonly limit: number;
  }) => Effect.Effect<ReadonlyArray<RlRunSummary>, RunStoreError>;
  readonly listMetrics: (input: {
    readonly runId: string;
    readonly limit: number;
  }) => Effect.Effect<ReadonlyArray<RlMetricBatch>, RunStoreError>;
  readonly listArtifacts: (input: {
    readonly runId: string;
  }) => Effect.Effect<ReadonlyArray<RlArtifactMetadata>, RunStoreError>;
  readonly getRun: (input: {
    readonly runId: string;
  }) => Effect.Effect<
    { readonly summary: RlRunSummary; readonly manifest: RlResolvedManifest | null },
    RunStoreError | RlRunNotFoundError
  >;
  /**
   * Marks every run that was still active as `interrupted`. Called once at
   * startup: a run whose server died has an unknown outcome, not a failed one.
   */
  readonly markActiveAsInterrupted: (input: {
    readonly at: string;
  }) => Effect.Effect<number, RunStoreError>;
  readonly findArtifact: (input: {
    readonly runId: string;
    readonly artifactId: string;
  }) => Effect.Effect<
    { readonly metadata: RlArtifactMetadata; readonly relativePath: string } | null,
    RunStoreError
  >;
}

export class RunStore extends Context.Service<RunStore, RunStoreShape>()("t3/rl/RunStore") {}

const ACTIVE_STATES = ["requested", "preparing", "running", "cancelling"] as const;

interface RunRow {
  readonly runId: string;
  readonly projectId: string;
  readonly experimentId: string;
  readonly state: string;
  readonly manifestJson: string | null;
  readonly errorCode: string | null;
  readonly errorMessage: string | null;
  readonly requestedAt: string;
  readonly startedAt: string | null;
  readonly endedAt: string | null;
  readonly lastMessageAt: string | null;
}

const decodeRunSummary = Schema.decodeUnknownSync(RlRunSummary);
const decodeManifest = Schema.decodeUnknownSync(RlResolvedManifest);
const decodeMetricBatch = Schema.decodeUnknownSync(RlMetricBatch);
const decodeArtifact = Schema.decodeUnknownSync(RlArtifactMetadata);
const isPersistenceSqlError = Schema.is(PersistenceSqlError);

const toSummary = (row: RunRow): RlRunSummary =>
  decodeRunSummary({
    runId: row.runId,
    projectId: row.projectId,
    experimentId: row.experimentId,
    state: row.state,
    requestedAt: row.requestedAt,
    startedAt: row.startedAt,
    endedAt: row.endedAt,
    lastMessageAt: row.lastMessageAt,
    errorCode: row.errorCode,
    errorMessage: row.errorMessage,
  });

const SELECT_RUN_COLUMNS = `
  run_id AS "runId",
  project_id AS "projectId",
  experiment_id AS "experimentId",
  state,
  manifest_json AS "manifestJson",
  error_code AS "errorCode",
  error_message AS "errorMessage",
  requested_at AS "requestedAt",
  started_at AS "startedAt",
  ended_at AS "endedAt",
  last_message_at AS "lastMessageAt"
`;

const makeRunStore = Effect.gen(function* () {
  const sql = yield* SqlClient.SqlClient;
  const crypto = yield* Crypto.Crypto;

  const insertRequested: RunStoreShape["insertRequested"] = (input) =>
    Effect.gen(function* () {
      if (input.requestId === undefined) {
        yield* sql`
          INSERT INTO rl_runs (run_id, project_id, experiment_id, state, requested_at)
          VALUES (${input.runId}, ${input.projectId}, ${input.experimentId}, 'requested', ${input.requestedAt})
        `;
        return { runId: input.runId, inserted: true };
      }

      yield* sql`
        INSERT OR IGNORE INTO rl_runs
          (run_id, project_id, experiment_id, state, requested_at, client_request_id)
        VALUES (
          ${input.runId},
          ${input.projectId},
          ${input.experimentId},
          'requested',
          ${input.requestedAt},
          ${input.requestId}
        )
      `;
      const changes = yield* sql<{ readonly inserted: number }>`SELECT changes() AS inserted`;
      if ((changes[0]?.inserted ?? 0) > 0) {
        return { runId: input.runId, inserted: true };
      }
      const existing = yield* sql<{ readonly runId: string }>`
        SELECT run_id AS "runId"
        FROM rl_runs
        WHERE project_id = ${input.projectId} AND client_request_id = ${input.requestId}
      `;
      const existingRun = existing[0];
      if (existingRun === undefined) {
        return yield* new PersistenceSqlError({
          operation: "rl.insertRequested",
          detail: "idempotent insert was ignored without an existing request match",
        });
      }
      return { runId: existingRun.runId, inserted: false };
    }).pipe(
      Effect.mapError((cause) =>
        isPersistenceSqlError(cause) ? cause : toPersistenceSqlError("rl.insertRequested")(cause),
      ),
    );

  const updateState: RunStoreShape["updateState"] = (input) =>
    Effect.gen(function* () {
      const terminal = isTerminalRlRunState(input.state);
      yield* sql`
        UPDATE rl_runs
        SET
          state = ${input.state},
          error_code = ${input.errorCode ?? null},
          error_message = ${input.errorMessage ?? null},
          worker_pid = COALESCE(${input.workerPid ?? null}, worker_pid),
          started_at = CASE
            WHEN started_at IS NULL AND ${input.state} = 'running' THEN ${input.at}
            ELSE started_at
          END,
          ended_at = CASE WHEN ${terminal ? 1 : 0} = 1 THEN ${input.at} ELSE ended_at END,
          last_message_at = ${input.at}
        WHERE run_id = ${input.runId}
      `;
    }).pipe(Effect.mapError(toPersistenceSqlError("rl.updateState")));

  const setManifest: RunStoreShape["setManifest"] = (input) =>
    Effect.gen(function* () {
      const rows = yield* sql<{ readonly manifestJson: string | null }>`
        SELECT manifest_json AS "manifestJson" FROM rl_runs WHERE run_id = ${input.runId}
      `.pipe(Effect.mapError(toPersistenceSqlError("rl.setManifest.read")));
      if (rows[0]?.manifestJson != null) {
        return yield* new RlManifestAlreadySetError({ runId: input.runId });
      }
      yield* sql`
        UPDATE rl_runs
        SET manifest_json = ${
          // @effect-diagnostics-next-line preferSchemaOverJson:off - opaque blob column.
          JSON.stringify(input.manifest)
        }
        WHERE run_id = ${input.runId}
      `.pipe(Effect.mapError(toPersistenceSqlError("rl.setManifest.write")));
    });

  const appendMetrics: RunStoreShape["appendMetrics"] = (input) =>
    Effect.gen(function* () {
      yield* sql`
        INSERT INTO rl_run_metrics (run_id, seq, step, wall_clock_ms, values_json)
        VALUES (
          ${input.runId},
          ${input.seq},
          ${input.batch.step},
          ${input.batch.wallClockMs},
          ${
            // @effect-diagnostics-next-line preferSchemaOverJson:off - opaque blob column.
            JSON.stringify(input.batch.values)
          }
        )
      `;
      // Telemetry only advances the run's liveness marker. It never changes
      // lifecycle state: a busy metric stream is not evidence of progress.
      yield* sql`
        UPDATE rl_runs SET last_message_at = ${input.at} WHERE run_id = ${input.runId}
      `;
    }).pipe(Effect.mapError(toPersistenceSqlError("rl.appendMetrics")));

  const recordArtifact: RunStoreShape["recordArtifact"] = (input) =>
    Effect.gen(function* () {
      const uuid = yield* crypto.randomUUIDv4.pipe(Effect.orDie);
      const artifactId = `art_${uuid.replace(/-/g, "").slice(0, 24)}`;
      yield* sql`
        INSERT INTO rl_run_artifacts
          (artifact_id, run_id, kind, relative_path, bytes, content_type, produced_at)
        VALUES (
          ${artifactId},
          ${input.runId},
          ${input.kind},
          ${input.relativePath},
          ${input.bytes},
          ${input.contentType},
          ${input.producedAt}
        )
      `.pipe(Effect.mapError(toPersistenceSqlError("rl.recordArtifact")));
      yield* sql`
        UPDATE rl_runs SET last_message_at = ${input.producedAt} WHERE run_id = ${input.runId}
      `.pipe(Effect.mapError(toPersistenceSqlError("rl.recordArtifact.touch")));
      return decodeArtifact({
        artifactId,
        kind: input.kind,
        bytes: input.bytes,
        contentType: input.contentType,
        producedAt: input.producedAt,
      });
    });

  const listRuns: RunStoreShape["listRuns"] = (input) =>
    sql<RunRow>`
      SELECT ${sql.literal(SELECT_RUN_COLUMNS)}
      FROM rl_runs
      WHERE project_id = ${input.projectId}
      ORDER BY requested_at DESC
      LIMIT ${input.limit}
    `.pipe(
      Effect.map((rows) => rows.map(toSummary)),
      Effect.mapError(toPersistenceSqlError("rl.listRuns")),
    );

  const listMetrics: RunStoreShape["listMetrics"] = (input) =>
    sql<{
      readonly step: number;
      readonly wallClockMs: number;
      readonly valuesJson: string;
    }>`
      SELECT step, wall_clock_ms AS "wallClockMs", values_json AS "valuesJson"
      FROM (
        SELECT seq, step, wall_clock_ms, values_json
        FROM rl_run_metrics
        WHERE run_id = ${input.runId}
        ORDER BY seq DESC
        LIMIT ${input.limit}
      ) AS recent_metrics
      ORDER BY seq ASC
    `.pipe(
      Effect.map((rows) =>
        rows.map((row) =>
          decodeMetricBatch({
            step: row.step,
            wallClockMs: row.wallClockMs,
            values: JSON.parse(row.valuesJson),
          }),
        ),
      ),
      Effect.mapError(toPersistenceSqlError("rl.listMetrics")),
    );

  const listArtifacts: RunStoreShape["listArtifacts"] = (input) =>
    sql<{
      readonly artifactId: string;
      readonly kind: string;
      readonly bytes: number;
      readonly contentType: string;
      readonly producedAt: string;
    }>`
      SELECT
        artifact_id AS "artifactId",
        kind,
        bytes,
        content_type AS "contentType",
        produced_at AS "producedAt"
      FROM rl_run_artifacts
      WHERE run_id = ${input.runId}
      ORDER BY produced_at ASC
    `.pipe(
      Effect.map((rows) => rows.map((row) => decodeArtifact(row))),
      Effect.mapError(toPersistenceSqlError("rl.listArtifacts")),
    );

  const findArtifact: RunStoreShape["findArtifact"] = (input) =>
    sql<{
      readonly artifactId: string;
      readonly kind: string;
      readonly bytes: number;
      readonly contentType: string;
      readonly producedAt: string;
      readonly relativePath: string;
    }>`
      SELECT
        artifact_id AS "artifactId",
        kind,
        bytes,
        content_type AS "contentType",
        produced_at AS "producedAt",
        relative_path AS "relativePath"
      FROM rl_run_artifacts
      WHERE run_id = ${input.runId} AND artifact_id = ${input.artifactId}
    `.pipe(
      Effect.map((rows) => {
        const row = rows[0];
        if (row === undefined) return null;
        return {
          metadata: decodeArtifact({
            artifactId: row.artifactId,
            kind: row.kind,
            bytes: row.bytes,
            contentType: row.contentType,
            producedAt: row.producedAt,
          }),
          relativePath: row.relativePath,
        };
      }),
      Effect.mapError(toPersistenceSqlError("rl.findArtifact")),
    );

  const getRun: RunStoreShape["getRun"] = (input) =>
    Effect.gen(function* () {
      const rows = yield* sql<RunRow>`
        SELECT ${sql.literal(SELECT_RUN_COLUMNS)}
        FROM rl_runs
        WHERE run_id = ${input.runId}
      `.pipe(Effect.mapError(toPersistenceSqlError("rl.getRun")));
      const row = rows[0];
      if (row === undefined) {
        return yield* new RlRunNotFoundError({ runId: input.runId });
      }
      return {
        summary: toSummary(row),
        manifest:
          row.manifestJson === null
            ? null
            : decodeManifest(
                // @effect-diagnostics-next-line preferSchemaOverJson:off - opaque blob column.
                JSON.parse(row.manifestJson),
              ),
      };
    });

  const markActiveAsInterrupted: RunStoreShape["markActiveAsInterrupted"] = (input) =>
    Effect.gen(function* () {
      const before = yield* sql<{ readonly count: number }>`
        SELECT COUNT(*) AS count FROM rl_runs WHERE state IN ${sql.in(ACTIVE_STATES)}
      `;
      yield* sql`
        UPDATE rl_runs
        SET
          state = 'interrupted',
          error_code = 'ServerInterrupted',
          error_message = 'server restarted while run was active',
          ended_at = ${input.at},
          last_message_at = ${input.at}
        WHERE state IN ${sql.in(ACTIVE_STATES)}
      `;
      return before[0]?.count ?? 0;
    }).pipe(Effect.mapError(toPersistenceSqlError("rl.markActiveAsInterrupted")));

  return RunStore.of({
    insertRequested,
    updateState,
    setManifest,
    appendMetrics,
    recordArtifact,
    listRuns,
    listMetrics,
    listArtifacts,
    findArtifact,
    getRun,
    markActiveAsInterrupted,
  });
});

export const RunStoreLive = Layer.effect(RunStore, makeRunStore);
