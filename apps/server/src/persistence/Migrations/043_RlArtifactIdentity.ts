import * as Effect from "effect/Effect";
import * as SqlClient from "effect/unstable/sql/SqlClient";

/** Verified, content-addressed artifact metadata; null hashes preserve legacy rows. */
export default Effect.gen(function* () {
  const sql = yield* SqlClient.SqlClient;
  yield* sql`ALTER TABLE rl_run_artifacts ADD COLUMN sha256 TEXT`;
  yield* sql`ALTER TABLE rl_run_artifacts ADD COLUMN logical_name TEXT`;
  yield* sql`ALTER TABLE rl_run_artifacts ADD COLUMN format TEXT`;
  yield* sql`ALTER TABLE rl_run_artifacts ADD COLUMN state TEXT NOT NULL DEFAULT 'ready'`;
  yield* sql`ALTER TABLE rl_run_artifacts ADD COLUMN checkpoint_step INTEGER`;
  yield* sql`ALTER TABLE rl_run_artifacts ADD COLUMN file_count INTEGER`;
  yield* sql`ALTER TABLE rl_run_artifacts ADD COLUMN content_manifest_json TEXT`;
  yield* sql`
    CREATE INDEX IF NOT EXISTS idx_rl_run_artifacts_page
    ON rl_run_artifacts(run_id, produced_at, artifact_id)
  `;
});
