import * as Effect from "effect/Effect";
import * as SqlClient from "effect/unstable/sql/SqlClient";

/** Immutable study definitions and their bounded run schedule. */
export default Effect.gen(function* () {
  const sql = yield* SqlClient.SqlClient;
  yield* sql`
    CREATE TABLE IF NOT EXISTS rl_studies (
      study_id TEXT PRIMARY KEY,
      project_id TEXT NOT NULL,
      state TEXT NOT NULL,
      definition_json TEXT NOT NULL,
      protocol_sha256 TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `;
  yield* sql`
    CREATE INDEX IF NOT EXISTS idx_rl_studies_project_created
    ON rl_studies(project_id, created_at DESC)
  `;
  yield* sql`
    CREATE TABLE IF NOT EXISTS rl_study_runs (
      study_id TEXT NOT NULL,
      variant_label TEXT NOT NULL,
      training_seed INTEGER NOT NULL,
      seeds_json TEXT NOT NULL,
      run_id TEXT,
      state TEXT NOT NULL,
      PRIMARY KEY (study_id, variant_label, training_seed)
    )
  `;
  yield* sql`
    CREATE INDEX IF NOT EXISTS idx_rl_study_runs_run ON rl_study_runs(run_id)
  `;
  yield* sql`
    CREATE TABLE IF NOT EXISTS rl_evaluation_samples (
      run_id TEXT NOT NULL,
      protocol_sha256 TEXT NOT NULL,
      sample_id TEXT NOT NULL,
      generation_seed INTEGER NOT NULL,
      values_json TEXT NOT NULL,
      PRIMARY KEY (run_id, protocol_sha256, sample_id)
    )
  `;
});
