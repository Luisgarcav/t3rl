import { assert, it } from "@effect/vitest";
import * as Effect from "effect/Effect";
import * as SqlClient from "effect/unstable/sql/SqlClient";

import { runMigrations } from "../Migrations.ts";
import * as NodeSqliteClient from "../NodeSqliteClient.ts";

it.effect("preserves legacy samples while allowing deterministic samples linked to evidence", () =>
  Effect.gen(function* () {
    const sql = yield* SqlClient.SqlClient;
    yield* runMigrations({ toMigrationInclusive: 45 });
    yield* sql`INSERT INTO rl_evaluation_samples (run_id, protocol_sha256, sample_id, generation_seed, values_json) VALUES ('run_old', 'protocol_old', 'sample_old', 7, '{"eval/loss":1}')`;
    yield* runMigrations({ toMigrationInclusive: 46 });
    yield* sql`INSERT INTO rl_evaluation_samples (run_id, protocol_sha256, sample_id, generation_seed, values_json, source_artifact_id, source_artifact_sha256) VALUES ('run_new', 'protocol_new', 'sample_new', NULL, '{"eval/loss":2}', 'artifact_new', 'verified_hash')`;
    const rows = yield* sql<{
      readonly run_id: string;
      readonly generation_seed: number | null;
      readonly source_artifact_id: string | null;
    }>`SELECT run_id, generation_seed, source_artifact_id FROM rl_evaluation_samples ORDER BY run_id`;
    assert.deepEqual(rows, [
      { run_id: "run_new", generation_seed: null, source_artifact_id: "artifact_new" },
      { run_id: "run_old", generation_seed: 7, source_artifact_id: null },
    ]);
  }).pipe(Effect.provide(NodeSqliteClient.layerMemory())),
);
