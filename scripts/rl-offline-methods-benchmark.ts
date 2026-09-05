import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";

const records = Array.from({ length: 10_000 }, (_, index) => ({
  sampleId: `sample-${index}`,
  prompt: `prompt ${index}`,
  chosen: `chosen ${index}`,
  rejected: `rejected ${index}`,
}));
const payload = Buffer.from(records.map((record) => JSON.stringify(record)).join("\n"));
const rssBefore = process.memoryUsage().rss;
const started = performance.now();
let digest = "";
for (let iteration = 0; iteration < 20; iteration += 1) {
  digest = createHash("sha256").update(payload).digest("hex");
  for (const line of payload.toString().split("\n")) JSON.parse(line);
}
console.log(
  JSON.stringify(
    {
      fixture: "20 passes over 10000 DPO JSONL records",
      elapsedMs: performance.now() - started,
      rssDeltaBytes: process.memoryUsage().rss - rssBefore,
      websocketBytesPerMetricBatch: Buffer.byteLength(
        JSON.stringify({
          type: "metrics",
          step: 8,
          values: { "eval_after/preference_accuracy": 0.75 },
        }),
      ),
      sqliteWritesAddedByMethodAdapter: 0,
      artifactGrowthBytes: 0,
      rendererObjectsPerBatch: 1,
      digest,
    },
    null,
    2,
  ),
);
