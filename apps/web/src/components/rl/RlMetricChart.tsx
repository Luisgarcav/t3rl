import type { RlMetricBatch } from "@t3tools/contracts";

import { Card, CardDescription, CardHeader, CardPanel, CardTitle } from "../ui/card";
import {
  formatRlMetricValue,
  latestRlMetricValue,
  type RlMetricDefinition,
  selectRlMetricPoints,
} from "./rlPresentation";

const CHART_WIDTH = 320;
const CHART_HEIGHT = 104;
const CHART_PADDING = 8;

interface ChartGeometry {
  readonly path: string;
  readonly pointCount: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly latestPoint: {
    readonly x: number;
    readonly y: number;
  };
}

function buildChartGeometry(points: ReturnType<typeof selectRlMetricPoints>): ChartGeometry | null {
  if (points.length === 0) return null;
  const steps = points.map((point) => point.step);
  const values = points.map((point) => point.value);
  const minimumStep = Math.min(...steps);
  const maximumStep = Math.max(...steps);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const stepRange = maximumStep - minimumStep;
  const valueRange = maximum - minimum;
  const valueMagnitude = Math.max(Math.abs(minimum), Math.abs(maximum), 1e-12);
  const effectivelyConstant = valueRange <= Math.max(1e-12, valueMagnitude * 1e-6);
  const width = CHART_WIDTH - CHART_PADDING * 2;
  const height = CHART_HEIGHT - CHART_PADDING * 2;
  const chartPoints = points.map((point) => ({
    x:
      stepRange === 0
        ? CHART_WIDTH / 2
        : CHART_PADDING + ((point.step - minimumStep) / stepRange) * width,
    y: effectivelyConstant
      ? CHART_HEIGHT / 2
      : CHART_HEIGHT - CHART_PADDING - ((point.value - minimum) / valueRange) * height,
  }));
  const path = chartPoints
    .map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`)
    .join(" ");
  return {
    path,
    pointCount: chartPoints.length,
    minimum,
    maximum,
    latestPoint: chartPoints.at(-1)!,
  };
}

export function RlMetricChart({
  definition,
  metrics,
}: {
  readonly definition: RlMetricDefinition;
  readonly metrics: ReadonlyArray<RlMetricBatch>;
}) {
  const points = selectRlMetricPoints(metrics, definition.key);
  const geometry = buildChartGeometry(points);
  const latest = latestRlMetricValue(metrics, definition.key);

  return (
    <Card className="min-w-0 overflow-hidden rounded-xl">
      <CardHeader className="grid grid-cols-[1fr_auto] gap-x-3 p-4 pb-2">
        <div className="min-w-0">
          <CardTitle className="truncate text-sm">{definition.label}</CardTitle>
          <CardDescription className="mt-1 truncate text-xs">
            {definition.description}
          </CardDescription>
        </div>
        <span className="font-mono text-sm font-semibold tabular-nums text-foreground">
          {formatRlMetricValue(latest)}
        </span>
      </CardHeader>
      <CardPanel className="px-3 pb-3 pt-0">
        {geometry === null ? (
          <div className="flex h-26 items-center justify-center rounded-lg bg-muted/24 text-xs text-muted-foreground">
            Waiting for metric data
          </div>
        ) : (
          <div className="relative h-26 overflow-hidden rounded-lg bg-muted/18">
            <svg
              aria-label={`${definition.label} across training steps`}
              className="size-full"
              preserveAspectRatio="none"
              role="img"
              viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
            >
              <path
                d={`M${CHART_PADDING},${CHART_HEIGHT / 2} H${CHART_WIDTH - CHART_PADDING}`}
                fill="none"
                stroke="var(--color-border)"
                strokeDasharray="3 4"
                vectorEffect="non-scaling-stroke"
              />
              {geometry.pointCount > 1 ? (
                <path
                  d={geometry.path}
                  data-slot="rl-metric-chart-line"
                  fill="none"
                  stroke={definition.color}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                />
              ) : null}
            </svg>
            <span
              aria-hidden
              className="pointer-events-none absolute size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-background"
              data-slot="rl-metric-chart-latest-point"
              style={{
                backgroundColor: definition.color,
                left: `${(geometry.latestPoint.x / CHART_WIDTH) * 100}%`,
                top: `${(geometry.latestPoint.y / CHART_HEIGHT) * 100}%`,
              }}
            />
            <span className="absolute left-2 top-1 font-mono text-[9px] tabular-nums text-muted-foreground/70">
              {formatRlMetricValue(geometry.maximum)}
            </span>
            <span className="absolute bottom-1 left-2 font-mono text-[9px] tabular-nums text-muted-foreground/70">
              {formatRlMetricValue(geometry.minimum)}
            </span>
          </div>
        )}
      </CardPanel>
    </Card>
  );
}
