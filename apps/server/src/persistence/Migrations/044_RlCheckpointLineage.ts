import * as Effect from "effect/Effect";
import * as SqlClient from "effect/unstable/sql/SqlClient";

/** Protocol-v2 checkpoint evidence and immutable child-to-parent run lineage. */
export default Effect.gen(function* () {
  const sql = yield* SqlClient.SqlClient;

  yield* sql`ALTER TABLE rl_run_artifacts ADD COLUMN evidence_json TEXT`;

  yield* sql`
    CREATE TABLE IF NOT EXISTS rl_run_lineage (
      child_run_id TEXT PRIMARY KEY,
      parent_run_id TEXT NOT NULL,
      source_artifact_id TEXT NOT NULL,
      source_artifact_sha256 TEXT NOT NULL,
      relation TEXT NOT NULL,
      source_step INTEGER NOT NULL,
      created_at TEXT NOT NULL
    )
  `;

  yield* sql`
    CREATE INDEX IF NOT EXISTS idx_rl_run_lineage_parent
    ON rl_run_lineage(parent_run_id, created_at)
  `;
});
