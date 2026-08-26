import { formatRlMetricValue, type RlMetricDefinition } from "../rlPresentation";
import { sampleAggregateSeries, type RlAggregatePoint } from "./rlComparison";

const CHART_WIDTH = 720;
const CHART_HEIGHT = 230;
const PADDING_LEFT = 52;
const PADDING_RIGHT = 14;
const PADDING_TOP = 14;
const PADDING_BOTTOM = 28;

export type RlCenterStatistic = "mean" | "median";
export type RlBandStatistic = "range" | "interquartile";

interface ChartGeometry {
  readonly centerPath: string;
  readonly bandPath: string;
  readonly minimumStep: number;
  readonly maximumStep: number;
  readonly minimumValue: number;
  readonly maximumValue: number;
  readonly displayedPointCount: number;
}

function pointCenter(point: RlAggregatePoint, statistic: RlCenterStatistic): number {
  return statistic === "mean" ? point.mean : point.median;
}

function pointBand(point: RlAggregatePoint, statistic: RlBandStatistic): readonly [number, number] {
  return statistic === "range"
    ? [point.minimum, point.maximum]
    : [point.firstQuartile, point.thirdQuartile];
}

function buildChartGeometry(
  series: ReadonlyArray<RlAggregatePoint>,
  centerStatistic: RlCenterStatistic,
  bandStatistic: RlBandStatistic,
): ChartGeometry | null {
  const points = sampleAggregateSeries(series);
  if (points.length === 0) return null;

  const minimumStep = points[0]?.step ?? 0;
  const maximumStep = points.at(-1)?.step ?? minimumStep;
  const bandValues = points.flatMap((point) => pointBand(point, bandStatistic));
  const centerValues = points.map((point) => pointCenter(point, centerStatistic));
  const minimumValue = Math.min(...bandValues, ...centerValues);
  const maximumValue = Math.max(...bandValues, ...centerValues);
  const stepRange = Math.max(1, maximumStep - minimumStep);
  const valueRange = Math.max(1e-9, maximumValue - minimumValue);
  const innerWidth = CHART_WIDTH - PADDING_LEFT - PADDING_RIGHT;
  const innerHeight = CHART_HEIGHT - PADDING_TOP - PADDING_BOTTOM;
  const x = (step: number) => PADDING_LEFT + ((step - minimumStep) / stepRange) * innerWidth;
  const y = (value: number) =>
    CHART_HEIGHT - PADDING_BOTTOM - ((value - minimumValue) / valueRange) * innerHeight;
  const pathFor = (values: ReadonlyArray<readonly [number, number]>) =>
    values
      .map(
        ([step, value], index) =>
          `${index === 0 ? "M" : "L"}${x(step).toFixed(2)},${y(value).toFixed(2)}`,
      )
      .join(" ");

  const upper = points.map((point) => [point.step, pointBand(point, bandStatistic)[1]] as const);
  const lower = points
    .toReversed()
    .map((point) => [point.step, pointBand(point, bandStatistic)[0]] as const);
  const center = points.map((point) => [point.step, pointCenter(point, centerStatistic)] as const);
  return {
    centerPath: pathFor(center),
    bandPath: `${pathFor(upper)} ${pathFor(lower).replace(/^M/, "L")} Z`,
    minimumStep,
    maximumStep,
    minimumValue,
    maximumValue,
    displayedPointCount: points.length,
  };
}

export function RlAggregateMetricChart({
  definition,
  series,
  centerStatistic,
  bandStatistic,
}: {
  readonly definition: RlMetricDefinition;
  readonly series: ReadonlyArray<RlAggregatePoint>;
  readonly centerStatistic: RlCenterStatistic;
  readonly bandStatistic: RlBandStatistic;
}) {
  const geometry = buildChartGeometry(series, centerStatistic, bandStatistic);
  if (geometry === null) {
    return (
      <div className="flex h-58 items-center justify-center rounded-xl border border-dashed bg-muted/12 px-6 text-center text-sm text-muted-foreground">
        No finite observations are available for this metric in the selected snapshots.
      </div>
    );
  }

  const counts = series.map((point) => point.count);
  const minimumCount = Math.min(...counts);
  const maximumCount = Math.max(...counts);
  const centerLabel = centerStatistic === "mean" ? "Mean" : "Median";
  const bandLabel = bandStatistic === "range" ? "Observed min–max" : "Observed Q1–Q3";

  return (
    <div className="space-y-2">
      <div className="relative h-58 overflow-hidden rounded-xl border bg-muted/10">
        <svg
          aria-label={`${definition.label}: ${centerLabel.toLowerCase()} with ${bandLabel.toLowerCase()} band across exact training steps`}
          className="size-full"
          preserveAspectRatio="none"
          role="img"
          viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        >
          {[0, 0.5, 1].map((fraction) => {
            const y = PADDING_TOP + fraction * (CHART_HEIGHT - PADDING_TOP - PADDING_BOTTOM);
            return (
              <path
                d={`M${PADDING_LEFT},${y} H${CHART_WIDTH - PADDING_RIGHT}`}
                fill="none"
                key={fraction}
                stroke="var(--color-border)"
                strokeDasharray={fraction === 0.5 ? "3 4" : undefined}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
          <path d={geometry.bandPath} fill={definition.color} fillOpacity="0.16" stroke="none" />
          <path
            d={geometry.centerPath}
            fill="none"
            stroke={definition.color}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2.5"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
        <span className="absolute left-2 top-2 font-mono text-[10px] tabular-nums text-muted-foreground">
          {formatRlMetricValue(geometry.maximumValue)}
        </span>
        <span className="absolute bottom-7 left-2 font-mono text-[10px] tabular-nums text-muted-foreground">
          {formatRlMetricValue(geometry.minimumValue)}
        </span>
        <span className="absolute bottom-1 left-13 font-mono text-[10px] tabular-nums text-muted-foreground">
          step {geometry.minimumStep.toLocaleString()}
        </span>
        <span className="absolute bottom-1 right-3 font-mono text-[10px] tabular-nums text-muted-foreground">
          step {geometry.maximumStep.toLocaleString()}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span>
          <span
            aria-hidden="true"
            className="mr-1.5 inline-block h-0.5 w-4 align-middle"
            style={{ backgroundColor: definition.color }}
          />
          {centerLabel}
        </span>
        <span>{bandLabel} band</span>
        <span>
          n={minimumCount === maximumCount ? minimumCount : `${minimumCount}–${maximumCount}`} seeds
          per step
        </span>
        {geometry.displayedPointCount < series.length ? (
          <span>
            Displaying {geometry.displayedPointCount.toLocaleString()} of{" "}
            {series.length.toLocaleString()} exact-step points
          </span>
        ) : null}
      </div>
    </div>
  );
}
