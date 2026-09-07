import * as Effect from "effect/Effect";
import * as SqlClient from "effect/unstable/sql/SqlClient";

/** Deterministic evaluations have no generation seed; indexed rows retain their source identity. */
export default Effect.gen(function* () {
  const sql = yield* SqlClient.SqlClient;
  yield* sql`ALTER TABLE rl_evaluation_samples RENAME TO rl_evaluation_samples_legacy`;
  yield* sql`
    CREATE TABLE rl_evaluation_samples (
      run_id TEXT NOT NULL,
      protocol_sha256 TEXT NOT NULL,
      sample_id TEXT NOT NULL,
      generation_seed INTEGER,
      values_json TEXT NOT NULL,
      source_artifact_id TEXT,
      source_artifact_sha256 TEXT,
      PRIMARY KEY (run_id, protocol_sha256, sample_id)
    )
  `;
  yield* sql`
    INSERT INTO rl_evaluation_samples (run_id, protocol_sha256, sample_id, generation_seed, values_json)
    SELECT run_id, protocol_sha256, sample_id, generation_seed, values_json
    FROM rl_evaluation_samples_legacy
  `;
  yield* sql`DROP TABLE rl_evaluation_samples_legacy`;
});
