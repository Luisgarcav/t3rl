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
  RlArtifactPage,
  RlArtifactEvidence,
  RlLineageEdge,
  RlMetricBatch,
  RlResolvedManifest,
  RlRunLineage,
  RlRunNotFoundError,
  RlRunSummary,
  RlStudy,
  RlStudyDefinition,
  RlSeedSet,
  RlStudyNotFoundError,
  isTerminalRlRunState,
  type RlArtifactKind,
  type RlArtifactState,
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
import type { ArtifactContentEntry } from "./ArtifactIdentity.ts";

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
  readonly lineage?: RlLineageEdge | undefined;
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
  readonly sha256: string;
  readonly logicalName: string;
  readonly format: string;
  readonly checkpointStep?: number | undefined;
  readonly fileCount: number;
  readonly contentManifest: ReadonlyArray<ArtifactContentEntry>;
  readonly evidence?: RlArtifactEvidence | undefined;
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
    readonly limit: number;
    readonly cursor?: string | undefined;
  }) => Effect.Effect<RlArtifactPage, RunStoreError>;
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
    {
      readonly metadata: RlArtifactMetadata;
      readonly relativePath: string;
      readonly contentManifest: ReadonlyArray<ArtifactContentEntry> | null;
    } | null,
    RunStoreError
  >;
  readonly getLineage: (input: {
    readonly runId: string;
  }) => Effect.Effect<RlRunLineage, RunStoreError>;
  readonly listReadyIntermediateCheckpoints: (input: {
    readonly runId: string;
  }) => Effect.Effect<
    ReadonlyArray<{ readonly metadata: RlArtifactMetadata; readonly relativePath: string }>,
    RunStoreError
  >;
  readonly setArtifactState: (input: {
    readonly artifactId: string;
    readonly state: RlArtifactState;
  }) => Effect.Effect<RlArtifactMetadata | null, RunStoreError>;
  readonly createStudy: (input: RlStudy) => Effect.Effect<void, RunStoreError>;
  readonly getStudy: (input: {
    readonly studyId: string;
  }) => Effect.Effect<RlStudy, RunStoreError | RlStudyNotFoundError>;
  readonly updateStudyRun: (input: {
    readonly studyId: string;
    readonly variantLabel: string;
    readonly trainingSeed: number;
    readonly runId?: string;
    readonly state: RlStudy["runs"][number]["state"];
    readonly studyState: RlStudy["state"];
    readonly at: string;
  }) => Effect.Effect<void, RunStoreError>;
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
const decodeLineageEdge = Schema.decodeUnknownSync(RlLineageEdge);
const decodeStudy = Schema.decodeUnknownSync(RlStudy);
const encodeStudyDefinition = Schema.encodeSync(Schema.fromJsonString(RlStudyDefinition));
const decodeStudyDefinition = Schema.decodeUnknownSync(Schema.fromJsonString(RlStudyDefinition));
const encodeSeedSet = Schema.encodeSync(Schema.fromJsonString(RlSeedSet));
const decodeSeedSet = Schema.decodeUnknownSync(Schema.fromJsonString(RlSeedSet));
const decodeArtifactContentManifest = Schema.decodeUnknownSync(
  Schema.Array(
    Schema.Struct({
      path: Schema.String,
      bytes: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
      sha256: Schema.String.check(Schema.isPattern(/^[0-9a-f]{64}$/)),
    }),
  ),
);
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
    sql
      .withTransaction(
        Effect.gen(function* () {
          const insertLineage = () =>
            input.lineage === undefined
              ? Effect.void
              : sql`
                INSERT INTO rl_run_lineage
                  (
                    child_run_id,
                    parent_run_id,
                    source_artifact_id,
                    source_artifact_sha256,
                    relation,
                    source_step,
                    created_at
                  )
                VALUES (
                  ${input.lineage.childRunId},
                  ${input.lineage.parentRunId},
                  ${input.lineage.sourceArtifactId},
                  ${input.lineage.sourceArtifactSha256},
                  ${input.lineage.relation},
                  ${input.lineage.sourceStep},
                  ${input.lineage.createdAt}
                )
              `.pipe(Effect.asVoid);

          if (input.requestId === undefined) {
            yield* sql`
            INSERT INTO rl_runs (run_id, project_id, experiment_id, state, requested_at)
            VALUES (${input.runId}, ${input.projectId}, ${input.experimentId}, 'requested', ${input.requestedAt})
          `;
            yield* insertLineage();
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
            yield* insertLineage();
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
        }),
      )
      .pipe(
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
          (
            artifact_id,
            run_id,
            kind,
            relative_path,
            bytes,
            content_type,
            produced_at,
            sha256,
            logical_name,
            format,
            state,
            checkpoint_step,
            file_count,
            content_manifest_json,
            evidence_json
          )
        VALUES (
          ${artifactId},
          ${input.runId},
          ${input.kind},
          ${input.relativePath},
          ${input.bytes},
          ${input.contentType},
          ${input.producedAt},
          ${input.sha256},
          ${input.logicalName},
          ${input.format},
          'ready',
          ${input.checkpointStep ?? null},
          ${input.fileCount},
          ${
            // @effect-diagnostics-next-line preferSchemaOverJson:off - bounded canonical evidence.
            JSON.stringify(input.contentManifest)
          },
          ${
            input.evidence === undefined
              ? null
              : // @effect-diagnostics-next-line preferSchemaOverJson:off - schema-decoded evidence.
                JSON.stringify(input.evidence)
          }
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
        sha256: input.sha256,
        logicalName: input.logicalName,
        format: input.format,
        state: "ready",
        checkpointStep: input.checkpointStep ?? null,
        fileCount: input.fileCount,
        evidence: input.evidence ?? null,
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

  interface ArtifactRow {
    readonly artifactId: string;
    readonly kind: string;
    readonly relativePath: string;
    readonly bytes: number;
    readonly contentType: string;
    readonly producedAt: string;
    readonly sha256: string | null;
    readonly logicalName: string | null;
    readonly format: string | null;
    readonly state: string;
    readonly checkpointStep: number | null;
    readonly fileCount: number | null;
    readonly contentManifestJson: string | null;
    readonly evidenceJson: string | null;
  }

  const artifactMetadata = (row: ArtifactRow): RlArtifactMetadata =>
    decodeArtifact({
      artifactId: row.artifactId,
      kind: row.kind,
      bytes: row.bytes,
      contentType: row.contentType,
      producedAt: row.producedAt,
      sha256: row.sha256,
      logicalName: row.logicalName ?? row.relativePath,
      format: row.format ?? "unknown",
      state: row.state,
      checkpointStep: row.checkpointStep,
      fileCount: row.fileCount ?? 0,
      evidence: row.evidenceJson === null ? null : JSON.parse(row.evidenceJson),
    });

  const listArtifacts: RunStoreShape["listArtifacts"] = (input) =>
    sql<ArtifactRow>`
      SELECT
        artifact_id AS "artifactId",
        kind,
        relative_path AS "relativePath",
        bytes,
        content_type AS "contentType",
        produced_at AS "producedAt",
        sha256,
        logical_name AS "logicalName",
        format,
        state,
        checkpoint_step AS "checkpointStep",
        file_count AS "fileCount",
        content_manifest_json AS "contentManifestJson",
        evidence_json AS "evidenceJson"
      FROM rl_run_artifacts
      WHERE run_id = ${input.runId}
        AND (
          ${input.cursor ?? null} IS NULL
          OR (produced_at, artifact_id) < (
            SELECT produced_at, artifact_id
            FROM rl_run_artifacts
            WHERE run_id = ${input.runId} AND artifact_id = ${input.cursor ?? null}
          )
        )
      ORDER BY produced_at DESC, artifact_id DESC
      LIMIT ${input.limit + 1}
    `.pipe(
      Effect.map((rows) => {
        const pageRows = rows.slice(0, input.limit);
        return {
          artifacts: pageRows.map(artifactMetadata),
          nextCursor: rows.length > input.limit ? (pageRows.at(-1)?.artifactId ?? null) : null,
        };
      }),
      Effect.mapError(toPersistenceSqlError("rl.listArtifacts")),
    );

  const findArtifact: RunStoreShape["findArtifact"] = (input) =>
    sql<ArtifactRow>`
      SELECT
        artifact_id AS "artifactId",
        kind,
        bytes,
        content_type AS "contentType",
        produced_at AS "producedAt",
        relative_path AS "relativePath",
        sha256,
        logical_name AS "logicalName",
        format,
        state,
        checkpoint_step AS "checkpointStep",
        file_count AS "fileCount",
        content_manifest_json AS "contentManifestJson",
        evidence_json AS "evidenceJson"
      FROM rl_run_artifacts
      WHERE run_id = ${input.runId} AND artifact_id = ${input.artifactId}
    `.pipe(
      Effect.map((rows) => {
        const row = rows[0];
        if (row === undefined) return null;
        return {
          metadata: artifactMetadata(row),
          relativePath: row.relativePath,
          contentManifest:
            row.contentManifestJson === null
              ? null
              : decodeArtifactContentManifest(JSON.parse(row.contentManifestJson)),
        };
      }),
      Effect.mapError(toPersistenceSqlError("rl.findArtifact")),
    );

  interface LineageRow {
    readonly childRunId: string;
    readonly parentRunId: string;
    readonly sourceArtifactId: string;
    readonly sourceArtifactSha256: string;
    readonly relation: string;
    readonly sourceStep: number;
    readonly createdAt: string;
    readonly depth: number;
  }

  const getLineage: RunStoreShape["getLineage"] = (input) =>
    sql<LineageRow>`
      WITH RECURSIVE ancestry (
        child_run_id,
        parent_run_id,
        source_artifact_id,
        source_artifact_sha256,
        relation,
        source_step,
        created_at,
        depth
      ) AS (
        SELECT
          child_run_id,
          parent_run_id,
          source_artifact_id,
          source_artifact_sha256,
          relation,
          source_step,
          created_at,
          0
        FROM rl_run_lineage
        WHERE child_run_id = ${input.runId}

        UNION ALL

        SELECT
          parent.child_run_id,
          parent.parent_run_id,
          parent.source_artifact_id,
          parent.source_artifact_sha256,
          parent.relation,
          parent.source_step,
          parent.created_at,
          ancestry.depth + 1
        FROM rl_run_lineage AS parent
        JOIN ancestry ON parent.child_run_id = ancestry.parent_run_id
        WHERE ancestry.depth < 64
      )
      SELECT
        child_run_id AS "childRunId",
        parent_run_id AS "parentRunId",
        source_artifact_id AS "sourceArtifactId",
        source_artifact_sha256 AS "sourceArtifactSha256",
        relation,
        source_step AS "sourceStep",
        created_at AS "createdAt",
        depth
      FROM ancestry
      ORDER BY depth ASC
      LIMIT 65
    `.pipe(
      Effect.map((rows) => ({
        edges: rows.slice(0, 64).map((row) => decodeLineageEdge(row)),
        truncated: rows.length > 64,
      })),
      Effect.mapError(toPersistenceSqlError("rl.getLineage")),
    );

  const listReadyIntermediateCheckpoints: RunStoreShape["listReadyIntermediateCheckpoints"] = (
    input,
  ) =>
    sql<ArtifactRow>`
      SELECT
        artifact_id AS "artifactId",
        kind,
        relative_path AS "relativePath",
        bytes,
        content_type AS "contentType",
        produced_at AS "producedAt",
        sha256,
        logical_name AS "logicalName",
        format,
        state,
        checkpoint_step AS "checkpointStep",
        file_count AS "fileCount",
        content_manifest_json AS "contentManifestJson",
        evidence_json AS "evidenceJson"
      FROM rl_run_artifacts
      WHERE run_id = ${input.runId}
        AND kind = 'checkpoint'
        AND state = 'ready'
        AND json_extract(evidence_json, '$._tag') = 'Checkpoint'
        AND json_extract(evidence_json, '$.checkpointClass') = 'intermediate'
      ORDER BY checkpoint_step DESC, produced_at DESC, artifact_id DESC
      LIMIT 65
    `.pipe(
      Effect.map((rows) =>
        rows.map((row) => ({ metadata: artifactMetadata(row), relativePath: row.relativePath })),
      ),
      Effect.mapError(toPersistenceSqlError("rl.listReadyIntermediateCheckpoints")),
    );

  const setArtifactState: RunStoreShape["setArtifactState"] = (input) =>
    Effect.gen(function* () {
      yield* sql`
        UPDATE rl_run_artifacts SET state = ${input.state} WHERE artifact_id = ${input.artifactId}
      `;
      const rows = yield* sql<ArtifactRow>`
        SELECT
          artifact_id AS "artifactId",
          kind,
          relative_path AS "relativePath",
          bytes,
          content_type AS "contentType",
          produced_at AS "producedAt",
          sha256,
          logical_name AS "logicalName",
          format,
          state,
          checkpoint_step AS "checkpointStep",
          file_count AS "fileCount",
          content_manifest_json AS "contentManifestJson",
          evidence_json AS "evidenceJson"
        FROM rl_run_artifacts
        WHERE artifact_id = ${input.artifactId}
      `;
      const row = rows[0];
      return row === undefined ? null : artifactMetadata(row);
    }).pipe(Effect.mapError(toPersistenceSqlError("rl.setArtifactState")));

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

  const createStudy: RunStoreShape["createStudy"] = (input) =>
    sql
      .withTransaction(
        Effect.gen(function* () {
          yield* sql`
          INSERT INTO rl_studies (study_id, project_id, state, definition_json, protocol_sha256, created_at, updated_at)
          VALUES (${input.studyId}, ${input.projectId}, ${input.state}, ${encodeStudyDefinition(input.definition)}, ${input.protocolSha256}, ${input.createdAt}, ${input.updatedAt})
        `;
          for (const run of input.runs) {
            yield* sql`
            INSERT INTO rl_study_runs (study_id, variant_label, training_seed, seeds_json, run_id, state)
            VALUES (${input.studyId}, ${run.variantLabel}, ${run.seeds.training}, ${encodeSeedSet(run.seeds)}, ${run.runId}, ${run.state})
          `;
          }
        }),
      )
      .pipe(Effect.mapError(toPersistenceSqlError("rl.createStudy")));

  interface StudyRow {
    readonly studyId: string;
    readonly projectId: string;
    readonly state: string;
    readonly definitionJson: string;
    readonly protocolSha256: string;
    readonly createdAt: string;
    readonly updatedAt: string;
  }
  interface StudyRunRow {
    readonly variantLabel: string;
    readonly seedsJson: string;
    readonly runId: string | null;
    readonly state: string;
  }
  const getStudy: RunStoreShape["getStudy"] = (input) =>
    Effect.gen(function* () {
      const rows =
        yield* sql<StudyRow>`SELECT study_id AS "studyId", project_id AS "projectId", state, definition_json AS "definitionJson", protocol_sha256 AS "protocolSha256", created_at AS "createdAt", updated_at AS "updatedAt" FROM rl_studies WHERE study_id = ${input.studyId}`;
      const row = rows[0];
      if (row === undefined) return yield* new RlStudyNotFoundError({ studyId: input.studyId });
      const runRows =
        yield* sql<StudyRunRow>`SELECT variant_label AS "variantLabel", seeds_json AS "seedsJson", run_id AS "runId", state FROM rl_study_runs WHERE study_id = ${input.studyId} ORDER BY training_seed, variant_label`;
      return decodeStudy({
        ...row,
        definition: decodeStudyDefinition(row.definitionJson),
        runs: runRows.map((run) => ({ ...run, seeds: decodeSeedSet(run.seedsJson) })),
      });
    }).pipe(
      Effect.mapError((error) =>
        Schema.is(RlStudyNotFoundError)(error)
          ? error
          : toPersistenceSqlError("rl.getStudy")(error),
      ),
    );

  const updateStudyRun: RunStoreShape["updateStudyRun"] = (input) =>
    sql
      .withTransaction(
        Effect.gen(function* () {
          yield* sql`UPDATE rl_study_runs SET run_id = COALESCE(${input.runId ?? null}, run_id), state = ${input.state} WHERE study_id = ${input.studyId} AND variant_label = ${input.variantLabel} AND training_seed = ${input.trainingSeed}`;
          yield* sql`UPDATE rl_studies SET state = ${input.studyState}, updated_at = ${input.at} WHERE study_id = ${input.studyId}`;
        }),
      )
      .pipe(Effect.mapError(toPersistenceSqlError("rl.updateStudyRun")));

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
    getLineage,
    listReadyIntermediateCheckpoints,
    setArtifactState,
    getRun,
    markActiveAsInterrupted,
    createStudy,
    getStudy,
    updateStudyRun,
  });
});

export const RunStoreLive = Layer.effect(RunStore, makeRunStore);
