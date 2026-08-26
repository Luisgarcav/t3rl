import type { RlRunProjection } from "@t3tools/client-runtime/state/rl";

const MAX_CONFIG_CHARACTERS = 4_000;

function boundedJson(value: unknown): string {
  const encoded = JSON.stringify(value, null, 2);
  if (encoded === undefined) return "null";
  return encoded.length <= MAX_CONFIG_CHARACTERS
    ? encoded
    : `${encoded.slice(0, MAX_CONFIG_CHARACTERS)}\n… configuration truncated`;
}

function metricLines(projection: RlRunProjection): ReadonlyArray<string> {
  const sorted = [...projection.metrics].sort(
    (left, right) => left.step - right.step || left.wallClockMs - right.wallClockMs,
  );
  const keys = [...new Set(sorted.flatMap((batch) => Object.keys(batch.values)))].toSorted();
  return keys.map((key) => {
    const observed = sorted.flatMap((batch) => {
      const value = batch.values[key];
      return value === undefined ? [] : [{ step: batch.step, value }];
    });
    const numeric = observed.filter(
      (entry): entry is { readonly step: number; readonly value: number } =>
        typeof entry.value === "number" && Number.isFinite(entry.value),
    );
    const nonFinite = observed.filter(
      (entry) => typeof entry.value === "string" && ["nan", "+inf", "-inf"].includes(entry.value),
    ).length;
    if (numeric.length === 0) {
      return `- ${key}: numeric points=0, non-finite=${nonFinite}, observed=${observed.length}`;
    }
    const values = numeric.map((entry) => entry.value);
    const first = numeric[0]!;
    const last = numeric.at(-1)!;
    return `- ${key}: points=${numeric.length}, first=${first.value} @${first.step}, last=${last.value} @${last.step}, min=${Math.min(...values)}, max=${Math.max(...values)}, non-finite=${nonFinite}`;
  });
}

/** Produces a bounded evidence block suitable for a user-visible agent prompt. */
export function formatResearchRunEvidence(projection: RlRunProjection): string {
  const { summary, manifest } = projection;
  return [
    `state: ${summary.state}`,
    `experiment: ${summary.experimentId}`,
    `requested: ${summary.requestedAt}`,
    `started: ${summary.startedAt ?? "not started"}`,
    `ended: ${summary.endedAt ?? "not ended"}`,
    `failure: ${summary.errorCode ?? "none"}${summary.errorMessage ? ` — ${summary.errorMessage}` : ""}`,
    "",
    "manifest:",
    manifest === null
      ? "- unavailable"
      : [
          `- runner: ${manifest.runnerId} ${manifest.runnerVersion}`,
          `- seed: ${manifest.seed}`,
          `- source: ${manifest.sourceRevision ?? "unavailable"}; dirty=${manifest.sourceDirty === null ? "unknown" : manifest.sourceDirty}`,
          `- environment fingerprint: ${manifest.environmentFingerprint}`,
          `- instrumentation: ${manifest.instrumentationLevel}`,
          `- hardware: ${manifest.hardwareSummary}`,
          `- effective config: ${boundedJson(manifest.effectiveConfig)}`,
        ].join("\n"),
    "",
    `metric batches: ${projection.metrics.length}`,
    ...metricLines(projection),
    "",
    `artifacts: ${
      projection.artifacts.length === 0
        ? "none"
        : projection.artifacts
            .map((artifact) => `${artifact.artifactId} (${artifact.kind}, ${artifact.bytes} bytes)`)
            .join(", ")
    }`,
  ].join("\n");
}
