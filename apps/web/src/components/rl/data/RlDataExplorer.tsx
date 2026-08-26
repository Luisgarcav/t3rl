import type { RlMetricBatch } from "@t3tools/contracts";
import { DatabaseIcon } from "lucide-react";
import { useState } from "react";

import { Badge } from "../../ui/badge";
import { Card, CardAction, CardDescription, CardHeader, CardPanel, CardTitle } from "../../ui/card";
import { Select, SelectItem, SelectPopup, SelectTrigger, SelectValue } from "../../ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../../ui/table";
import { formatRlMetricValue, RL_METRIC_DEFINITIONS } from "../rlPresentation";
import {
  type VisualizationMode,
  VisualizationModeTabs,
} from "../visualization/VisualizationModeTabs";
import {
  collectRlMetricKeys,
  createRlDataVisualizationSpec,
  type RlMetricDataSelection,
  selectRlMetricData,
} from "./rlDataExplorer";

const DATA_CHART_WIDTH = 720;
const DATA_CHART_HEIGHT = 260;
const DATA_CHART_PADDING_X = 48;
const DATA_CHART_PADDING_Y = 28;

interface DataChartGeometry {
  readonly path: string;
  readonly minimumStep: number;
  readonly maximumStep: number;
  readonly minimumValue: number;
  readonly maximumValue: number;
}

function metricLabel(metricKey: string): string {
  return (
    RL_METRIC_DEFINITIONS.find((definition) => definition.key === metricKey)?.label ?? metricKey
  );
}

function metricColor(metricKey: string): string {
  return (
    RL_METRIC_DEFINITIONS.find((definition) => definition.key === metricKey)?.color ??
    "var(--color-primary)"
  );
}

function buildDataChartGeometry(selection: RlMetricDataSelection): DataChartGeometry | null {
  const points = selection.rows.flatMap((row) =>
    typeof row.value === "number" && Number.isFinite(row.value)
      ? [{ step: row.step, value: row.value }]
      : [],
  );
  if (points.length === 0) return null;
  const steps = points.map((point) => point.step);
  const values = points.map((point) => point.value);
  const minimumStep = Math.min(...steps);
  const maximumStep = Math.max(...steps);
  const minimumValue = Math.min(...values);
  const maximumValue = Math.max(...values);
  const stepRange = Math.max(1, maximumStep - minimumStep);
  const valueRange = Math.max(1e-9, maximumValue - minimumValue);
  const width = DATA_CHART_WIDTH - DATA_CHART_PADDING_X * 2;
  const height = DATA_CHART_HEIGHT - DATA_CHART_PADDING_Y * 2;
  const path = points
    .map((point, index) => {
      const x = DATA_CHART_PADDING_X + ((point.step - minimumStep) / stepRange) * width;
      const y =
        DATA_CHART_HEIGHT -
        DATA_CHART_PADDING_Y -
        ((point.value - minimumValue) / valueRange) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return { path, minimumStep, maximumStep, minimumValue, maximumValue };
}

function DataChart({
  metricKey,
  selection,
}: {
  readonly metricKey: string;
  readonly selection: RlMetricDataSelection;
}) {
  const geometry = buildDataChartGeometry(selection);
  if (geometry === null) {
    return (
      <div className="flex min-h-72 items-center justify-center rounded-xl border border-dashed text-sm text-muted-foreground">
        This metric has no finite values to plot yet.
      </div>
    );
  }
  return (
    <div className="relative min-h-72 overflow-hidden rounded-xl border border-border/70 bg-muted/12">
      <svg
        aria-label={`${metricLabel(metricKey)} across training steps`}
        className="absolute inset-0 size-full"
        preserveAspectRatio="none"
        role="img"
        viewBox={`0 0 ${DATA_CHART_WIDTH} ${DATA_CHART_HEIGHT}`}
      >
        {[0, 0.5, 1].map((fraction) => {
          const y =
            DATA_CHART_PADDING_Y + fraction * (DATA_CHART_HEIGHT - DATA_CHART_PADDING_Y * 2);
          return (
            <path
              d={`M${DATA_CHART_PADDING_X},${y} H${DATA_CHART_WIDTH - DATA_CHART_PADDING_X}`}
              fill="none"
              key={fraction}
              stroke="var(--color-border)"
              strokeDasharray="3 5"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
        <path
          d={geometry.path}
          fill="none"
          stroke={metricColor(metricKey)}
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2.5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className="pointer-events-none absolute inset-0 flex flex-col justify-between p-3 font-mono text-[10px] tabular-nums text-muted-foreground">
        <span>{formatRlMetricValue(geometry.maximumValue)}</span>
        <div className="flex items-end justify-between gap-4">
          <span>{formatRlMetricValue(geometry.minimumValue)}</span>
          <span>
            step {geometry.minimumStep.toLocaleString()} → {geometry.maximumStep.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}

function DataTable({ selection }: { readonly selection: RlMetricDataSelection }) {
  return (
    <div className="max-h-[32rem] overflow-auto rounded-xl border border-border/70">
      <Table>
        <TableHeader className="sticky top-0 z-10 bg-card">
          <TableRow>
            <TableHead>Step</TableHead>
            <TableHead>Elapsed</TableHead>
            <TableHead className="text-right">Value</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {selection.rows.toReversed().map((row) => (
            <TableRow key={`${row.step}:${row.wallClockMs}`}>
              <TableCell className="font-mono tabular-nums">{row.step.toLocaleString()}</TableCell>
              <TableCell className="font-mono tabular-nums">
                {(row.wallClockMs / 1_000).toLocaleString(undefined, {
                  maximumFractionDigits: 2,
                })}
                s
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">
                {formatRlMetricValue(row.value)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function RlDataExplorer({ metrics }: { readonly metrics: ReadonlyArray<RlMetricBatch> }) {
  const metricKeys = collectRlMetricKeys(metrics);
  const [selectedMetricOverride, setSelectedMetricOverride] = useState<string | null>(null);
  const [mode, setMode] = useState<VisualizationMode>("visual");
  const metricKey =
    selectedMetricOverride !== null && metricKeys.includes(selectedMetricOverride)
      ? selectedMetricOverride
      : (metricKeys[0] ?? null);
  const selection = metricKey === null ? null : selectRlMetricData(metrics, metricKey);
  const label = metricKey === null ? "Metric" : metricLabel(metricKey);
  const source = selection === null ? null : createRlDataVisualizationSpec(label, selection);

  return (
    <Card className="min-w-0 overflow-hidden">
      <CardHeader>
        <div className="flex items-center gap-2">
          <DatabaseIcon className="size-4 text-muted-foreground" />
          <CardTitle className="text-base">Data explorer</CardTitle>
        </div>
        <CardDescription>
          Inspect every metric retained by the run, including custom worker keys.
        </CardDescription>
        <CardAction>
          <Badge variant="secondary">{metricKeys.length.toLocaleString()} metrics</Badge>
        </CardAction>
      </CardHeader>
      <CardPanel>
        {metricKey === null || selection === null || source === null ? (
          <div className="rounded-xl border border-dashed p-10 text-center text-sm text-muted-foreground">
            Metric data will appear here after the worker emits its first batch.
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <label className="min-w-56 flex-1 space-y-1.5 sm:max-w-sm">
                <span className="text-xs font-medium text-muted-foreground">Metric</span>
                <Select
                  items={metricKeys.map((key) => ({ label: metricLabel(key), value: key }))}
                  value={metricKey}
                  onValueChange={(value) => {
                    if (typeof value === "string") setSelectedMetricOverride(value);
                  }}
                >
                  <SelectTrigger aria-label="Data explorer metric" className="w-full">
                    <SelectValue>{label}</SelectValue>
                  </SelectTrigger>
                  <SelectPopup>
                    {metricKeys.map((key) => (
                      <SelectItem key={key} value={key}>
                        <span className="flex min-w-0 items-center justify-between gap-4">
                          <span className="truncate">{metricLabel(key)}</span>
                          <span className="font-mono text-[10px] text-muted-foreground">{key}</span>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectPopup>
                </Select>
              </label>
              <VisualizationModeTabs
                label="Data explorer display"
                value={mode}
                onChange={setMode}
              />
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <Badge variant="outline">{metricKey}</Badge>
              <span>{selection.totalRows.toLocaleString()} recorded values</span>
              {selection.truncated ? <span>· showing the latest 200 rows</span> : null}
            </div>

            <div aria-label={`${label} ${mode}`} role="tabpanel">
              {mode === "visual" ? (
                <DataChart metricKey={metricKey} selection={selection} />
              ) : mode === "data" ? (
                <DataTable selection={selection} />
              ) : (
                <pre className="max-h-[32rem] overflow-auto rounded-xl border border-border/70 bg-muted/18 p-4 font-mono text-[11px] leading-relaxed">
                  {JSON.stringify(source, null, 2)}
                </pre>
              )}
            </div>
          </div>
        )}
      </CardPanel>
    </Card>
  );
}
