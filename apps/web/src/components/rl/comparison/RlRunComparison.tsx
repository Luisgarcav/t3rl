"use client";

import { useAtomValue } from "@effect/atom-react";
import type { RlRunProjection } from "@t3tools/client-runtime/state/rl";
import type {
  EnvironmentId,
  RlExperimentId,
  RlRunId,
  RlRunSummary,
  RlStudyComparison,
} from "@t3tools/contracts";
import * as Option from "effect/Option";
import { AsyncResult, Atom } from "effect/unstable/reactivity";
import { AlertTriangleIcon, BarChart3Icon, InfoIcon } from "lucide-react";
import { useState } from "react";

import { cn } from "~/lib/utils";
import { formatEnvironmentQueryError } from "~/state/query";
import { rlEnvironment } from "~/state/rl";
import { useAtomCommand } from "~/state/use-atom-command";

import { Alert, AlertDescription, AlertTitle } from "../../ui/alert";
import { Badge } from "../../ui/badge";
import { Button } from "../../ui/button";
import { Card, CardDescription, CardHeader, CardPanel, CardTitle } from "../../ui/card";
import { Checkbox } from "../../ui/checkbox";
import { Input } from "../../ui/input";
import { Select, SelectItem, SelectPopup, SelectTrigger, SelectValue } from "../../ui/select";
import { Skeleton } from "../../ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../../ui/table";
import { Tooltip, TooltipPopup, TooltipTrigger } from "../../ui/tooltip";
import {
  formatRlMetricValue,
  formatRlState,
  RL_METRIC_DEFINITIONS,
  rlStatusVariant,
  type RlMetricDefinition,
} from "../rlPresentation";
import {
  RlAggregateMetricChart,
  type RlBandStatistic,
  type RlCenterStatistic,
} from "./RlAggregateMetricChart";
import {
  analyzeRlComparison,
  compatibleRunsForExperiment,
  listObservedMetricKeys,
  type RlComparisonAnalysis,
} from "./rlComparison";

const DEFAULT_SELECTION_LIMIT = 5;
const MAX_SELECTION_LIMIT = 12;
const requestedAtFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

interface DetailSnapshot {
  readonly data: RlRunProjection | null;
  readonly error: string | null;
  readonly isPending: boolean;
}

function parseComparisonDetailsKey(
  key: string,
): readonly [EnvironmentId, ReadonlyArray<RlRunId>] | null {
  try {
    const parsed: unknown = JSON.parse(key);
    if (
      !Array.isArray(parsed) ||
      parsed.length !== 2 ||
      typeof parsed[0] !== "string" ||
      !Array.isArray(parsed[1]) ||
      !parsed[1].every((runId) => typeof runId === "string")
    ) {
      return null;
    }
    return [parsed[0] as EnvironmentId, parsed[1] as ReadonlyArray<RlRunId>];
  } catch {
    return null;
  }
}

const comparisonDetailsAtom = Atom.family((key: string) =>
  Atom.make((get): Readonly<Record<string, DetailSnapshot>> => {
    const target = parseComparisonDetailsKey(key);
    if (target === null) return {};
    const [environmentId, runIds] = target;
    const details: Record<string, DetailSnapshot> = {};
    for (const runId of runIds) {
      const result = get(rlEnvironment.detail({ environmentId, input: { runId } }));
      details[runId] = {
        data: Option.getOrNull(AsyncResult.value(result)),
        error: result._tag === "Failure" ? formatEnvironmentQueryError(result.cause) : null,
        isPending: result.waiting,
      };
    }
    return details;
  }).pipe(Atom.withLabel(`rl-comparison-details:${key}`)),
);

const emptyComparisonDetailsAtom = Atom.make<Readonly<Record<string, DetailSnapshot>>>({}).pipe(
  Atom.withLabel("rl-comparison-details:empty"),
);

function useComparisonDetails(
  environmentId: EnvironmentId,
  runIds: ReadonlyArray<RlRunId>,
): Readonly<Record<string, DetailSnapshot>> {
  const key = JSON.stringify([environmentId, runIds]);
  return useAtomValue(
    runIds.length === 0 ? emptyComparisonDetailsAtom : comparisonDetailsAtom(key),
  );
}

export interface RlRunComparisonProps {
  readonly environmentId: EnvironmentId;
  readonly runs: ReadonlyArray<RlRunSummary>;
  readonly defaultExperimentId?: RlExperimentId | null;
  readonly maxSelectedRuns?: number;
  readonly className?: string;
}

function formatRequestedAt(value: string): string {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return "Unknown time";
  return requestedAtFormatter.format(timestamp);
}

function shortRunId(runId: RlRunId): string {
  return runId.length > 16 ? `${runId.slice(0, 8)}…${runId.slice(-5)}` : runId;
}

function RunIdLabel({ runId }: { readonly runId: RlRunId }) {
  return (
    <Tooltip>
      <TooltipTrigger render={<span className="font-mono text-xs" />}>
        {shortRunId(runId)}
      </TooltipTrigger>
      <TooltipPopup className="max-w-80 break-all">{runId}</TooltipPopup>
    </Tooltip>
  );
}

function metricDefinition(key: string): RlMetricDefinition {
  return (
    RL_METRIC_DEFINITIONS.find((definition) => definition.key === key) ?? {
      key,
      label: key,
      description: "Runner-provided metric",
      color: "var(--color-info)",
    }
  );
}

function defaultSelectedRunIds(
  runs: ReadonlyArray<RlRunSummary>,
  limit: number,
): ReadonlyArray<RlRunId> {
  return runs.slice(0, Math.min(DEFAULT_SELECTION_LIMIT, limit)).map((run) => run.runId);
}

function normalizeSelectionLimit(value: number): number {
  if (!Number.isFinite(value)) return MAX_SELECTION_LIMIT;
  return Math.max(1, Math.min(MAX_SELECTION_LIMIT, Math.floor(value)));
}

function ComparisonAnalysis({
  analysis,
  bandStatistic,
  centerStatistic,
  metricOptions,
  selectedMetric,
  onBandStatisticChange,
  onCenterStatisticChange,
  onMetricChange,
}: {
  readonly analysis: RlComparisonAnalysis;
  readonly bandStatistic: RlBandStatistic;
  readonly centerStatistic: RlCenterStatistic;
  readonly metricOptions: ReadonlyArray<RlMetricDefinition>;
  readonly selectedMetric: RlMetricDefinition;
  readonly onBandStatisticChange: (statistic: RlBandStatistic) => void;
  readonly onCenterStatisticChange: (statistic: RlCenterStatistic) => void;
  readonly onMetricChange: (metricKey: string) => void;
}) {
  const hasWarning = analysis.warnings.some((entry) => entry.severity === "warning");
  return (
    <>
      <div className="grid gap-3 md:grid-cols-3">
        <label className="space-y-1.5">
          <span className="text-sm font-medium">Metric</span>
          <Select
            items={metricOptions.map((definition) => ({
              label: definition.label,
              value: definition.key,
            }))}
            value={selectedMetric.key}
            onValueChange={(value) => {
              if (typeof value === "string") onMetricChange(value);
            }}
          >
            <SelectTrigger aria-label="Comparison metric" className="w-full">
              <SelectValue>{selectedMetric.label}</SelectValue>
            </SelectTrigger>
            <SelectPopup>
              {metricOptions.map((definition) => (
                <SelectItem key={definition.key} value={definition.key}>
                  {definition.label}
                </SelectItem>
              ))}
            </SelectPopup>
          </Select>
        </label>
        <label className="space-y-1.5">
          <span className="text-sm font-medium">Center</span>
          <Select
            items={[
              { label: "Mean", value: "mean" },
              { label: "Median", value: "median" },
            ]}
            value={centerStatistic}
            onValueChange={(value) => {
              if (value === "mean" || value === "median") onCenterStatisticChange(value);
            }}
          >
            <SelectTrigger aria-label="Center statistic" className="w-full">
              <SelectValue>{centerStatistic === "mean" ? "Mean" : "Median"}</SelectValue>
            </SelectTrigger>
            <SelectPopup>
              <SelectItem value="mean">Mean</SelectItem>
              <SelectItem value="median">Median</SelectItem>
            </SelectPopup>
          </Select>
        </label>
        <label className="space-y-1.5">
          <span className="text-sm font-medium">Band</span>
          <Select
            items={[
              { label: "Observed min–max", value: "range" },
              { label: "Observed Q1–Q3", value: "interquartile" },
            ]}
            value={bandStatistic}
            onValueChange={(value) => {
              if (value === "range" || value === "interquartile") {
                onBandStatisticChange(value);
              }
            }}
          >
            <SelectTrigger aria-label="Band statistic" className="w-full">
              <SelectValue>
                {bandStatistic === "range" ? "Observed min–max" : "Observed Q1–Q3"}
              </SelectValue>
            </SelectTrigger>
            <SelectPopup>
              <SelectItem value="range">Observed min–max</SelectItem>
              <SelectItem value="interquartile">Observed Q1–Q3</SelectItem>
            </SelectPopup>
          </Select>
        </label>
      </div>

      <RlAggregateMetricChart
        bandStatistic={bandStatistic}
        centerStatistic={centerStatistic}
        definition={selectedMetric}
        series={analysis.series}
      />

      {analysis.warnings.length > 0 ? (
        <Alert controlAlignment="first-line" variant={hasWarning ? "warning" : "info"}>
          {hasWarning ? <AlertTriangleIcon /> : <InfoIcon />}
          <AlertTitle>Interpretation guardrails</AlertTitle>
          <AlertDescription>
            <ul className="list-disc space-y-1 pl-4">
              {analysis.warnings.map((entry) => (
                <li key={entry.code}>{entry.message}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      ) : null}

      <div className="space-y-2">
        <div>
          <h3 className="text-sm font-semibold">Per-seed summary</h3>
          <p className="text-xs text-muted-foreground">
            Latest finite {selectedMetric.label.toLowerCase()} observation in each representative
            snapshot. Blank values stay blank.
          </p>
        </div>
        <div className="rounded-xl border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Seed</TableHead>
                <TableHead>Representative run</TableHead>
                <TableHead>State</TableHead>
                <TableHead className="text-right">Latest step</TableHead>
                <TableHead className="text-right">Latest value</TableHead>
                <TableHead className="text-right">Metric batches</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {analysis.seeds.map((seed) => (
                <TableRow key={seed.seed}>
                  <TableCell className="font-mono tabular-nums">{seed.seed}</TableCell>
                  <TableCell>
                    <RunIdLabel runId={seed.representative.summary.runId} />
                    {seed.runCount > 1 ? (
                      <span className="ml-2 text-muted-foreground">newest of {seed.runCount}</span>
                    ) : null}
                  </TableCell>
                  <TableCell>
                    <Badge variant={rlStatusVariant(seed.representative.summary.state)}>
                      {formatRlState(seed.representative.summary.state)}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {seed.latestMetricStep?.toLocaleString() ?? "—"}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {seed.latestMetricValue === null
                      ? "—"
                      : formatRlMetricValue(seed.latestMetricValue)}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {seed.metricBatchCount.toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </>
  );
}

export function RlRunComparison({
  environmentId,
  runs,
  defaultExperimentId = null,
  maxSelectedRuns = MAX_SELECTION_LIMIT,
  className,
}: RlRunComparisonProps) {
  const experimentIds = Array.from(new Set(runs.map((run) => run.experimentId))).toSorted();
  const runCountByExperiment = new Map<RlExperimentId, number>();
  for (const run of runs) {
    runCountByExperiment.set(
      run.experimentId,
      (runCountByExperiment.get(run.experimentId) ?? 0) + 1,
    );
  }
  const [experimentOverride, setExperimentOverride] = useState<RlExperimentId | null>(null);
  const [selectionByExperiment, setSelectionByExperiment] = useState<
    Readonly<Record<string, ReadonlyArray<RlRunId>>>
  >({});
  const [selectionWarning, setSelectionWarning] = useState<string | null>(null);
  const [metricOverride, setMetricOverride] = useState<string | null>(null);
  const [centerStatistic, setCenterStatistic] = useState<RlCenterStatistic>("mean");
  const [bandStatistic, setBandStatistic] = useState<RlBandStatistic>("range");
  const [studyId, setStudyId] = useState("");
  const [baselineLabel, setBaselineLabel] = useState("baseline");
  const [candidateLabel, setCandidateLabel] = useState("candidate");
  const [studyComparison, setStudyComparison] = useState<RlStudyComparison | null>(null);
  const [studyError, setStudyError] = useState<string | null>(null);
  const compareStudy = useAtomCommand(rlEnvironment.compareStudy, { reportFailure: false });
  const effectiveLimit = normalizeSelectionLimit(maxSelectedRuns);
  const preferredExperiment =
    defaultExperimentId !== null && experimentIds.includes(defaultExperimentId)
      ? defaultExperimentId
      : null;
  const selectedExperimentId =
    (experimentOverride !== null && experimentIds.includes(experimentOverride)
      ? experimentOverride
      : null) ??
    preferredExperiment ??
    experimentIds[0] ??
    null;
  const compatibleRuns =
    selectedExperimentId === null ? [] : compatibleRunsForExperiment(runs, selectedExperimentId);
  const selectedRunIds =
    selectedExperimentId === null
      ? []
      : (selectionByExperiment[selectedExperimentId] ??
        defaultSelectedRunIds(compatibleRuns, effectiveLimit));
  const availableRunIds = new Set(compatibleRuns.map((run) => run.runId));
  const activeSelectedRunIds = selectedRunIds.filter((runId) => availableRunIds.has(runId));
  const activeSelectedRunIdSet = new Set(activeSelectedRunIds);
  const detailsByRunId = useComparisonDetails(environmentId, activeSelectedRunIds);

  const selectedProjections = activeSelectedRunIds.flatMap((runId) => {
    const projection = detailsByRunId[runId]?.data;
    return projection === undefined || projection === null ? [] : [projection];
  });
  const observedMetricKeys = listObservedMetricKeys(selectedProjections);
  const metricOptions = (
    observedMetricKeys.length > 0 ? observedMetricKeys.map(metricDefinition) : RL_METRIC_DEFINITIONS
  ).toSorted((left, right) => left.label.localeCompare(right.label));
  const overriddenMetric = metricOptions.find((definition) => definition.key === metricOverride);
  const selectedMetric =
    overriddenMetric ??
    metricOptions.find((definition) => definition.key === "train/return") ??
    metricOptions[0] ??
    metricDefinition("train/return");
  const analysis =
    selectedExperimentId === null
      ? null
      : analyzeRlComparison(selectedProjections, selectedExperimentId, selectedMetric.key);
  const pendingCount = activeSelectedRunIds.filter(
    (runId) => detailsByRunId[runId]?.isPending !== false,
  ).length;
  const detailErrors = activeSelectedRunIds.flatMap((runId) => {
    const error = detailsByRunId[runId]?.error;
    return error === undefined || error === null ? [] : [{ runId, error }];
  });

  const toggleRun = (runId: RlRunId, checked: boolean) => {
    if (selectedExperimentId === null) return;
    setSelectionWarning(null);
    const current = activeSelectedRunIds;
    if (!checked) {
      setSelectionByExperiment((selections) => ({
        ...selections,
        [selectedExperimentId]: current.filter((selected) => selected !== runId),
      }));
      return;
    }
    if (current.includes(runId)) return;
    if (current.length >= effectiveLimit) {
      setSelectionWarning(
        `Select at most ${effectiveLimit} runs at once. This bounds snapshot traffic and chart work.`,
      );
      return;
    }
    setSelectionByExperiment((selections) => ({
      ...selections,
      [selectedExperimentId]: [...current, runId],
    }));
  };

  const runStudyComparison = async () => {
    setStudyError(null);
    const result = await compareStudy({
      environmentId,
      input: {
        studyId,
        baselineLabel,
        candidateLabel,
        metricKey: selectedMetric.key,
        estimator: {
          version: 1,
          statistic: "paired-mean-delta",
          statisticalUnit: "paired-sample-within-run-seed",
          confidenceLevel: 0.95,
          resamplingSeed: 17,
          resampleCount: 10_000,
          missingPairPolicy: "exclude",
        },
      },
    });
    if (result._tag === "Success") setStudyComparison(result.value);
    else setStudyError("The study could not be compared in this environment.");
  };

  if (selectedExperimentId === null) {
    return (
      <Card className={cn("overflow-hidden", className)}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <BarChart3Icon className="size-4 text-muted-foreground" />
            Multi-seed comparison
          </CardTitle>
          <CardDescription>Run an experiment at least once to compare snapshots.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader className="border-b">
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          <BarChart3Icon className="size-4 text-muted-foreground" />
          Multi-seed comparison
          <Badge variant="secondary">{activeSelectedRunIds.length} selected</Badge>
          {analysis !== null ? (
            <Badge variant="outline">{analysis.seeds.length} seeds</Badge>
          ) : null}
        </CardTitle>
        <CardDescription>
          Compare compatible runs from one experiment using resolved manifests and bounded detail
          snapshots.
        </CardDescription>
      </CardHeader>

      <CardPanel className="space-y-6 p-4 sm:p-6">
        <section className="space-y-3 rounded-xl border p-4">
          <div>
            <div className="text-sm font-medium">Authoritative paired study</div>
            <div className="text-xs text-muted-foreground">
              Server-computed hierarchical bootstrap with explicit N, seeds, failures, and interval.
            </div>
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            <Input
              aria-label="Study ID"
              placeholder="study_…"
              value={studyId}
              onChange={(event) => setStudyId(event.currentTarget.value)}
            />
            <Input
              aria-label="Baseline label"
              value={baselineLabel}
              onChange={(event) => setBaselineLabel(event.currentTarget.value)}
            />
            <Input
              aria-label="Candidate label"
              value={candidateLabel}
              onChange={(event) => setCandidateLabel(event.currentTarget.value)}
            />
          </div>
          <Button disabled={studyId.trim().length === 0} size="sm" onClick={runStudyComparison}>
            Compare study
          </Button>
          {studyError !== null ? <p className="text-xs text-destructive">{studyError}</p> : null}
          {studyComparison !== null ? (
            <div className="grid gap-2 text-xs sm:grid-cols-3">
              <span>N {studyComparison.n}</span>
              <span>Seeds {studyComparison.seedSet.join(", ") || "none"}</span>
              <span>Method hierarchical bootstrap v{studyComparison.estimator.version}</span>
              <span>
                Delta{" "}
                {studyComparison.pairedDelta === null
                  ? "—"
                  : formatRlMetricValue(studyComparison.pairedDelta)}
              </span>
              <span>
                Interval{" "}
                {studyComparison.interval === null
                  ? "not enough evidence"
                  : `${formatRlMetricValue(studyComparison.interval[0])}–${formatRlMetricValue(studyComparison.interval[1])}`}
              </span>
              <span>
                {studyComparison.unmatchedRuns} unmatched · {studyComparison.failedRuns} failed
              </span>
            </div>
          ) : null}
        </section>
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
          <label className="space-y-1.5">
            <span className="text-sm font-medium">Experiment</span>
            <Select
              items={experimentIds.map((experimentId) => ({
                label: experimentId,
                value: experimentId,
              }))}
              value={selectedExperimentId}
              onValueChange={(value) => {
                if (typeof value !== "string") return;
                setExperimentOverride(value);
                setSelectionWarning(null);
              }}
            >
              <SelectTrigger aria-label="Comparison experiment" className="w-full">
                <SelectValue>{selectedExperimentId}</SelectValue>
              </SelectTrigger>
              <SelectPopup>
                {experimentIds.map((experimentId) => (
                  <SelectItem key={experimentId} value={experimentId}>
                    {experimentId} ({runCountByExperiment.get(experimentId) ?? 0})
                  </SelectItem>
                ))}
              </SelectPopup>
            </Select>
          </label>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm font-medium">Compatible runs</span>
              <span className="text-xs text-muted-foreground">
                One snapshot per selected run · max {effectiveLimit}
              </span>
            </div>
            <div className="max-h-52 overflow-y-auto rounded-xl border">
              {compatibleRuns.map((run) => {
                const checked = activeSelectedRunIdSet.has(run.runId);
                const seed = detailsByRunId[run.runId]?.data?.manifest?.seed;
                return (
                  <label
                    className="flex cursor-pointer items-center gap-3 border-b px-3 py-2.5 last:border-b-0 hover:bg-muted/30"
                    key={run.runId}
                  >
                    <Checkbox
                      aria-label={`Compare run ${run.runId}`}
                      checked={checked}
                      onCheckedChange={(value) => toggleRun(run.runId, value === true)}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="flex flex-wrap items-center gap-2">
                        <RunIdLabel runId={run.runId} />
                        <Badge size="sm" variant={rlStatusVariant(run.state)}>
                          {formatRlState(run.state)}
                        </Badge>
                        {seed !== undefined ? (
                          <span className="font-mono text-xs text-muted-foreground">
                            seed {seed}
                          </span>
                        ) : null}
                      </span>
                      <span className="mt-0.5 block text-xs text-muted-foreground">
                        {formatRequestedAt(run.requestedAt)}
                      </span>
                    </span>
                  </label>
                );
              })}
            </div>
          </div>
        </div>

        {selectionWarning !== null ? (
          <Alert variant="warning">
            <AlertTriangleIcon />
            <AlertDescription>{selectionWarning}</AlertDescription>
          </Alert>
        ) : null}

        {detailErrors.length > 0 ? (
          <Alert variant="warning">
            <AlertTriangleIcon />
            <AlertTitle>Some snapshots could not be loaded</AlertTitle>
            <AlertDescription>
              <ul className="list-disc space-y-1 pl-4">
                {detailErrors.map(({ runId, error }) => (
                  <li key={runId}>
                    {shortRunId(runId)}: {error}
                  </li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        ) : null}

        {pendingCount > 0 ? (
          <div className="space-y-3">
            <div className="text-sm text-muted-foreground">
              Loading {pendingCount} run snapshot{pendingCount === 1 ? "" : "s"}…
            </div>
            <Skeleton className="h-58 rounded-xl" />
          </div>
        ) : activeSelectedRunIds.length === 0 ? (
          <div className="rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">
            Select at least one run. Two or more distinct verified seeds are needed for a multi-seed
            aggregate.
          </div>
        ) : analysis !== null ? (
          <ComparisonAnalysis
            analysis={analysis}
            bandStatistic={bandStatistic}
            centerStatistic={centerStatistic}
            metricOptions={metricOptions}
            selectedMetric={selectedMetric}
            onBandStatisticChange={setBandStatistic}
            onCenterStatisticChange={setCenterStatistic}
            onMetricChange={setMetricOverride}
          />
        ) : null}
      </CardPanel>
    </Card>
  );
}
