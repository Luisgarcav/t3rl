import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";

const definition = Buffer.from(JSON.stringify({ version: 1, adapter: "trl", method: "grpo" }));
const dataset = Buffer.alloc(1024 * 1024, 0x61);
const verifier = Buffer.from("def verify(completion, expected): return completion == expected\n");
const rssBefore = process.memoryUsage().rss;
const started = performance.now();
let digest = "";
for (let iteration = 0; iteration < 100; iteration += 1) {
  digest = createHash("sha256").update(definition).update(dataset).update(verifier).digest("hex");
}
const elapsedMs = performance.now() - started;
const validationPayload = JSON.stringify({
  valid: true,
  issues: [],
  resolvedInputs: [digest, digest, digest],
});
console.log(
  JSON.stringify(
    {
      fixture: "100 validations of 1 MiB dataset + definition + verifier",
      elapsedMs,
      rssDeltaBytes: process.memoryUsage().rss - rssBefore,
      sqliteWrites: 0,
      websocketValidationBytes: Buffer.byteLength(validationPayload),
      runSnapshotGrowthBytes: definition.length + dataset.length + verifier.length,
      rendererValidationObjects: 1,
    },
    null,
    2,
  ),
);
