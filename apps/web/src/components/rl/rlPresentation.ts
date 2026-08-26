import type { RlMetricBatch, RlRunState } from "@t3tools/contracts";

export interface RlMetricDefinition {
  readonly key: string;
  readonly label: string;
  readonly description: string;
  readonly color: string;
}

export const RL_METRIC_DEFINITIONS: ReadonlyArray<RlMetricDefinition> = [
  {
    key: "train/return",
    label: "Training return",
    description: "Mean episodic reward",
    color: "var(--color-info)",
  },
  {
    key: "train/episode_length",
    label: "Episode length",
    description: "Mean training episode length",
    color: "var(--color-success)",
  },
  {
    key: "eval/return",
    label: "Evaluation return",
    description: "Deterministic policy evaluation",
    color: "var(--color-primary)",
  },
  {
    key: "eval/episode_length",
    label: "Evaluation length",
    description: "Mean evaluation episode length",
    color: "var(--color-warning)",
  },
  {
    key: "train/policy_loss",
    label: "Policy loss",
    description: "PPO policy objective",
    color: "var(--color-chart-1, var(--color-info))",
  },
  {
    key: "train/value_loss",
    label: "Value loss",
    description: "Value function error",
    color: "var(--color-chart-2, var(--color-warning))",
  },
  {
    key: "train/entropy",
    label: "Entropy",
    description: "Policy exploration",
    color: "var(--color-chart-3, var(--color-success))",
  },
  {
    key: "train/approx_kl",
    label: "Approximate KL",
    description: "Policy update distance",
    color: "var(--color-chart-4, var(--color-destructive))",
  },
];

export type RlStatusVariant = "error" | "info" | "outline" | "secondary" | "success" | "warning";

export function rlStatusVariant(state: RlRunState): RlStatusVariant {
  switch (state) {
    case "running":
    case "completed":
      return "success";
    case "failed":
      return "error";
    case "cancelling":
    case "interrupted":
      return "warning";
    case "requested":
    case "preparing":
      return "info";
    case "cancelled":
      return "outline";
  }
}

export function formatRlState(state: RlRunState): string {
  return state.charAt(0).toUpperCase() + state.slice(1);
}

export interface RlMetricPoint {
  readonly step: number;
  readonly value: number;
}

export function selectRlMetricPoints(
  metrics: ReadonlyArray<RlMetricBatch>,
  key: string,
  maxPoints = 120,
): ReadonlyArray<RlMetricPoint> {
  const points = metrics.flatMap((batch) => {
    const value = batch.values[key];
    return typeof value === "number" && Number.isFinite(value) ? [{ step: batch.step, value }] : [];
  });
  if (points.length <= maxPoints) return points;

  const stride = Math.ceil(points.length / maxPoints);
  const sampled = points.filter((_, index) => index % stride === 0);
  const last = points.at(-1);
  if (last !== undefined && sampled.at(-1) !== last) {
    if (sampled.length >= maxPoints) sampled[sampled.length - 1] = last;
    else sampled.push(last);
  }
  return sampled;
}

export function latestRlMetricValue(
  metrics: ReadonlyArray<RlMetricBatch>,
  key: string,
): RlMetricBatch["values"][string] | undefined {
  for (let index = metrics.length - 1; index >= 0; index -= 1) {
    const value = metrics[index]?.values[key];
    if (value !== undefined) return value;
  }
  return undefined;
}

export function formatRlMetricValue(value: RlMetricBatch["values"][string] | undefined): string {
  if (value === undefined || value === null) return "—";
  if (typeof value === "string") return value.toUpperCase();
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

export function formatRlDuration(
  startedAt: string | null,
  endedAt: string | null,
  now: number,
): string {
  if (startedAt === null) return "Not started";
  const start = Date.parse(startedAt);
  const end = endedAt === null ? now : Date.parse(endedAt);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return "Unknown";
  const totalSeconds = Math.max(0, Math.floor((end - start) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return hours > 0
    ? `${hours}h ${String(minutes).padStart(2, "0")}m`
    : `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

export function formatRlBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
