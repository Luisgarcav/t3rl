import * as Effect from "effect/Effect";
import * as SqlClient from "effect/unstable/sql/SqlClient";

/** Transport idempotency for rl.startRun; null preserves pre-contract rows. */
export default Effect.gen(function* () {
  const sql = yield* SqlClient.SqlClient;
  yield* sql`ALTER TABLE rl_runs ADD COLUMN client_request_id TEXT`;
  yield* sql`
    CREATE UNIQUE INDEX IF NOT EXISTS idx_rl_runs_project_request
    ON rl_runs(project_id, client_request_id)
    WHERE client_request_id IS NOT NULL
  `;
});
