import * as Effect from "effect/Effect";
import * as SqlClient from "effect/unstable/sql/SqlClient";

/**
 * T3RL run kernel storage. Metrics live in their own table so that no training
 * telemetry can ever reach the orchestration event log.
 */
export default Effect.gen(function* () {
  const sql = yield* SqlClient.SqlClient;

  yield* sql`
    CREATE TABLE IF NOT EXISTS rl_runs (
      run_id TEXT PRIMARY KEY,
      project_id TEXT NOT NULL,
      experiment_id TEXT NOT NULL,
      state TEXT NOT NULL,
      manifest_json TEXT,
      error_code TEXT,
      error_message TEXT,
      worker_pid INTEGER,
      requested_at TEXT NOT NULL,
      started_at TEXT,
      ended_at TEXT,
      last_message_at TEXT
    )
  `;

  yield* sql`
    CREATE INDEX IF NOT EXISTS idx_rl_runs_project_requested
    ON rl_runs(project_id, requested_at DESC)
  `;

  yield* sql`
    CREATE INDEX IF NOT EXISTS idx_rl_runs_state
    ON rl_runs(state)
  `;

  yield* sql`
    CREATE TABLE IF NOT EXISTS rl_run_metrics (
      run_id TEXT NOT NULL,
      seq INTEGER NOT NULL,
      step INTEGER NOT NULL,
      wall_clock_ms INTEGER NOT NULL,
      values_json TEXT NOT NULL,
      PRIMARY KEY (run_id, seq)
    )
  `;

  yield* sql`
    CREATE TABLE IF NOT EXISTS rl_run_artifacts (
      artifact_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      relative_path TEXT NOT NULL,
      bytes INTEGER NOT NULL,
      content_type TEXT NOT NULL,
      produced_at TEXT NOT NULL
    )
  `;

  yield* sql`
    CREATE INDEX IF NOT EXISTS idx_rl_run_artifacts_run
    ON rl_run_artifacts(run_id)
  `;
});
