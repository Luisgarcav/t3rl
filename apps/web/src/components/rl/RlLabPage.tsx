import { scopeProjectRef } from "@t3tools/client-runtime/environment";
import type { RlRunProjection } from "@t3tools/client-runtime/state/rl";
import { squashAtomCommandFailure } from "@t3tools/client-runtime/state/runtime";
import {
  type EnvironmentId,
  isTerminalRlRunState,
  type ProjectId,
  type RlArtifactMetadata,
  type RlCapabilityReport,
  type RlExperimentSummary,
  type RlResolvedManifest,
  RlRunRequestId,
  type RlRunId,
  type RlRunSummary,
} from "@t3tools/contracts";
import {
  ActivityIcon,
  AlertCircleIcon,
  ArchiveIcon,
  BarChart3Icon,
  BoxIcon,
  CheckCircle2Icon,
  Clock3Icon,
  ExternalLinkIcon,
  FlaskConicalIcon,
  GaugeIcon,
  LayoutDashboardIcon,
  PlayIcon,
  RefreshCcwIcon,
  RotateCcwIcon,
  RouteIcon,
  SquareIcon,
} from "lucide-react";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";

import { useAssetUrlState } from "~/assets/assetUrls";
import { cn, randomUUID } from "~/lib/utils";
import { useProject } from "~/state/entities";
import { useEnvironmentQuery } from "~/state/query";
import { rlEnvironment } from "~/state/rl";
import { useAtomCommand } from "~/state/use-atom-command";
import { COLLAPSED_SIDEBAR_TITLEBAR_INSET_CLASS } from "~/workspaceTitlebar";

import { Alert, AlertDescription, AlertTitle } from "../ui/alert";
import { Badge } from "../ui/badge";
import { Button, buttonVariants } from "../ui/button";
import { Card, CardAction, CardDescription, CardHeader, CardPanel, CardTitle } from "../ui/card";
import { Input } from "../ui/input";
import { Select, SelectItem, SelectPopup, SelectTrigger, SelectValue } from "../ui/select";
import { Skeleton } from "../ui/skeleton";
import { Spinner } from "../ui/spinner";
import { RlRunComparison } from "./comparison/RlRunComparison";
import { RlDiagnosticsPanel } from "./diagnostics/RlDiagnosticsPanel";
import { analyzeRlDiagnostics } from "./diagnostics/rlDiagnostics";
import { RlMetricChart } from "./RlMetricChart";
import {
  formatRlBytes,
  formatRlDuration,
  formatRlState,
  RL_METRIC_DEFINITIONS,
  rlStatusVariant,
} from "./rlPresentation";
import { TrajectoryViewer } from "./trajectory/TrajectoryViewer";

export type RlLabView = "overview" | "behavior" | "compare" | "diagnostics";
type RunScopedRlLabView = Exclude<RlLabView, "compare">;

const RL_LAB_VIEWS: ReadonlyArray<{
  readonly id: RlLabView;
  readonly label: string;
  readonly requiresRun: boolean;
  readonly icon: ReactNode;
}> = [
  { id: "overview", label: "Overview", requiresRun: false, icon: <LayoutDashboardIcon /> },
  { id: "behavior", label: "Behavior", requiresRun: true, icon: <RouteIcon /> },
  { id: "compare", label: "Compare", requiresRun: false, icon: <BarChart3Icon /> },
  { id: "diagnostics", label: "Diagnostics", requiresRun: true, icon: <ActivityIcon /> },
];

const runDateFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

function formatRunTimestamp(timestamp: string | null): string {
  if (timestamp === null) return "—";
  const value = Date.parse(timestamp);
  return Number.isFinite(value) ? runDateFormatter.format(value) : "Unknown";
}

function commandFailureMessage(result: Parameters<typeof squashAtomCommandFailure>[0]): string {
  const failure = squashAtomCommandFailure(result);
  return failure instanceof Error && failure.message.trim().length > 0
    ? failure.message
    : "The request failed.";
}

export function RlLabPage({
  environmentId,
  projectId,
  selectedRunId,
  activeView,
  chromeVariant = "page",
  baselineRunId = null,
  baselineActionPending = false,
  onSelectRun,
  onViewChange,
  onUseRunAsBaseline,
}: {
  readonly environmentId: EnvironmentId;
  readonly projectId: ProjectId;
  readonly selectedRunId: RlRunId | null;
  readonly activeView: RlLabView;
  readonly chromeVariant?: "page" | "embedded";
  readonly baselineRunId?: RlRunId | null;
  readonly baselineActionPending?: boolean;
  readonly onSelectRun: (runId: RlRunId | null) => void;
  readonly onViewChange: (view: RlLabView) => void;
  readonly onUseRunAsBaseline?: (runId: RlRunId) => void;
}) {
  const project = useProject(scopeProjectRef(environmentId, projectId));
  const capabilitiesAtom = rlEnvironment.capabilities({ environmentId, input: {} });
  const runsAtom = rlEnvironment.runs({ environmentId, input: { projectId, limit: 200 } });
  const runAtom =
    selectedRunId === null
      ? null
      : rlEnvironment.run({ environmentId, input: { runId: selectedRunId } });
  const capabilities = useEnvironmentQuery(capabilitiesAtom);
  const runs = useEnvironmentQuery(runsAtom);
  const liveRun = useEnvironmentQuery(runAtom);
  const selectedSummary = liveRun.data?.summary ?? null;
  const displayedRuns = (runs.data?.runs ?? []).map((run) =>
    selectedSummary !== null && selectedSummary.runId === run.runId ? selectedSummary : run,
  );
  const effectiveView =
    selectedRunId === null && (activeView === "behavior" || activeView === "diagnostics")
      ? "overview"
      : activeView;
  const runView: RunScopedRlLabView = effectiveView === "compare" ? "overview" : effectiveView;

  return (
    <div className="@container/rl-lab flex min-h-0 min-w-0 flex-1 flex-col bg-background">
      <header
        className={cn(
          "flex shrink-0 items-center gap-3 border-b border-border px-3",
          chromeVariant === "page"
            ? [
                "drag-region h-[var(--workspace-topbar-height)] min-h-[var(--workspace-topbar-height)] @[36rem]/rl-lab:px-5",
                "wco:pr-[var(--workspace-native-controls-inset)]",
                COLLAPSED_SIDEBAR_TITLEBAR_INSET_CLASS,
              ]
            : "h-11 min-h-11",
        )}
      >
        <FlaskConicalIcon className="size-4 text-muted-foreground" />
        <div className="flex min-w-0 items-baseline gap-2">
          <h1 className="shrink-0 text-sm font-semibold">RL Lab</h1>
          <span className="truncate text-xs text-muted-foreground">
            {project?.title ?? projectId}
          </span>
        </div>
        <Button
          className={cn("ml-auto", chromeVariant === "page" && "no-drag")}
          size="xs"
          onClick={() => onSelectRun(null)}
        >
          <PlayIcon />
          New run
        </Button>
      </header>

      <RlLabNavigation
        activeView={effectiveView}
        hasSelectedRun={selectedRunId !== null}
        onViewChange={onViewChange}
      />

      <div className="flex min-h-0 flex-1 flex-col @[52rem]/rl-lab:flex-row">
        <RunHistory
          error={runs.error}
          isPending={runs.isPending}
          runs={displayedRuns}
          selectedRunId={selectedRunId}
          onRefresh={runs.refresh}
          onSelectRun={onSelectRun}
        />
        <main className="min-h-0 min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-7xl p-4 @[42rem]/rl-lab:p-6 @[64rem]/rl-lab:p-8">
            {effectiveView === "compare" ? (
              <RlRunComparison
                defaultExperimentId={selectedSummary?.experimentId ?? null}
                environmentId={environmentId}
                runs={displayedRuns}
              />
            ) : selectedRunId === null ? (
              <RunLauncher
                capabilities={capabilities.data}
                error={capabilities.error}
                isPending={capabilities.isPending}
                environmentId={environmentId}
                projectId={projectId}
                onRefresh={capabilities.refresh}
                onRunsRefresh={runs.refresh}
                onStarted={onSelectRun}
              />
            ) : (
              <RunDetail
                environmentId={environmentId}
                error={liveRun.error}
                isPending={liveRun.isPending}
                projection={liveRun.data}
                runId={selectedRunId}
                view={runView}
                baselineActionPending={baselineActionPending}
                isBaseline={selectedRunId === baselineRunId}
                onRefresh={liveRun.refresh}
                onRunsRefresh={runs.refresh}
                {...(onUseRunAsBaseline === undefined
                  ? {}
                  : { onUseAsBaseline: onUseRunAsBaseline })}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

function RlLabNavigation({
  activeView,
  hasSelectedRun,
  onViewChange,
}: {
  readonly activeView: RlLabView;
  readonly hasSelectedRun: boolean;
  readonly onViewChange: (view: RlLabView) => void;
}) {
  return (
    <nav
      aria-label="RL Lab views"
      className="flex shrink-0 gap-1 overflow-x-auto border-b border-border bg-muted/8 px-3 py-2 @[36rem]/rl-lab:px-5"
    >
      {RL_LAB_VIEWS.map((view) => {
        const disabled = view.requiresRun && !hasSelectedRun;
        return (
          <Button
            aria-current={activeView === view.id ? "page" : undefined}
            disabled={disabled}
            key={view.id}
            size="xs"
            title={disabled ? "Select a run to open this view" : undefined}
            variant={activeView === view.id ? "secondary" : "ghost-muted"}
            onClick={() => onViewChange(view.id)}
          >
            {view.icon}
            {view.label}
          </Button>
        );
      })}
    </nav>
  );
}

function RunHistory({
  error,
  isPending,
  runs,
  selectedRunId,
  onRefresh,
  onSelectRun,
}: {
  readonly error: string | null;
  readonly isPending: boolean;
  readonly runs: ReadonlyArray<RlRunSummary>;
  readonly selectedRunId: RlRunId | null;
  readonly onRefresh: () => void;
  readonly onSelectRun: (runId: RlRunId | null) => void;
}) {
  return (
    <aside className="flex max-h-52 shrink-0 flex-col border-b border-border bg-muted/8 @[52rem]/rl-lab:max-h-none @[52rem]/rl-lab:w-72 @[52rem]/rl-lab:border-r @[52rem]/rl-lab:border-b-0">
      <div className="flex h-12 shrink-0 items-center border-b border-border/70 px-3">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Run history
        </span>
        <Button
          aria-label="Refresh run history"
          className="ml-auto"
          disabled={isPending}
          size="icon-xs"
          title="Refresh run history"
          variant="ghost-muted"
          onClick={onRefresh}
        >
          <RefreshCcwIcon className={cn(isPending && "animate-spin")} />
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {isPending && runs.length === 0 ? (
          <div className="space-y-2 p-1">
            {Array.from({ length: 4 }, (_, index) => (
              <Skeleton className="h-15 rounded-lg" key={index} />
            ))}
          </div>
        ) : null}
        {error !== null && runs.length === 0 ? (
          <div className="p-3 text-xs text-destructive">{error}</div>
        ) : null}
        {!isPending && error === null && runs.length === 0 ? (
          <div className="flex flex-col items-center px-4 py-10 text-center">
            <ArchiveIcon className="mb-3 size-5 text-muted-foreground" />
            <span className="text-sm font-medium">No runs yet</span>
            <span className="mt-1 text-xs text-muted-foreground">
              Start an experiment to build history.
            </span>
          </div>
        ) : null}
        <div className="space-y-1">
          {runs.map((run) => (
            <button
              aria-current={selectedRunId === run.runId ? "page" : undefined}
              className={cn(
                "flex w-full min-w-0 flex-col gap-1 rounded-lg px-3 py-2.5 text-left outline-none transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring",
                selectedRunId === run.runId && "bg-accent",
              )}
              key={run.runId}
              type="button"
              onClick={() => onSelectRun(run.runId)}
            >
              <span className="flex w-full min-w-0 items-center gap-2">
                <span className="min-w-0 flex-1 truncate font-medium text-sm">
                  {run.experimentId}
                </span>
                <Badge size="sm" variant={rlStatusVariant(run.state)}>
                  {formatRlState(run.state)}
                </Badge>
              </span>
              <span className="flex w-full items-center justify-between gap-2 font-mono text-[10px] text-muted-foreground">
                <span className="truncate">{run.runId}</span>
                <span className="shrink-0">{formatRunTimestamp(run.requestedAt)}</span>
              </span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}

function RunLauncher({
  capabilities,
  error,
  isPending,
  environmentId,
  projectId,
  onRefresh,
  onRunsRefresh,
  onStarted,
}: {
  readonly capabilities: RlCapabilityReport | null;
  readonly error: string | null;
  readonly isPending: boolean;
  readonly environmentId: EnvironmentId;
  readonly projectId: ProjectId;
  readonly onRefresh: () => void;
  readonly onRunsRefresh: () => void;
  readonly onStarted: (runId: RlRunId) => void;
}) {
  const [experimentOverride, setExperimentOverride] = useState<string | null>(null);
  const [seedOverride, setSeedOverride] = useState<{
    readonly experimentId: string;
    readonly value: string;
  } | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const startRun = useAtomCommand(rlEnvironment.start, { reportFailure: false });
  const experiments = capabilities?.experiments ?? [];
  const selectedExperiment =
    experiments.find((entry) => entry.experimentId === experimentOverride) ??
    experiments[0] ??
    null;
  const selectedRunner =
    selectedExperiment === null
      ? null
      : (capabilities?.runners.find((entry) => entry.runnerId === selectedExperiment.runnerId) ??
        null);
  const seed =
    selectedExperiment !== null && seedOverride?.experimentId === selectedExperiment.experimentId
      ? seedOverride.value
      : String(selectedExperiment?.defaultSeed ?? 0);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (selectedExperiment === null || selectedRunner?.available !== true) return;
    if (seed.trim().length === 0) {
      setStartError("Seed is required.");
      return;
    }
    const parsedSeed = Number(seed);
    if (!Number.isSafeInteger(parsedSeed)) {
      setStartError("Seed must be a safe integer.");
      return;
    }

    setIsStarting(true);
    setStartError(null);
    const result = await startRun({
      environmentId,
      input: {
        projectId,
        experimentId: selectedExperiment.experimentId,
        seed: parsedSeed,
        requestId: RlRunRequestId.make(randomUUID()),
      },
    });
    setIsStarting(false);
    if (result._tag === "Success") {
      onRunsRefresh();
      onStarted(result.value.runId);
      return;
    }
    setStartError(commandFailureMessage(result));
  };

  if (isPending && capabilities === null) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-72 rounded-2xl" />
      </div>
    );
  }

  if (error !== null && capabilities === null) {
    return (
      <Alert variant="error">
        <AlertCircleIcon />
        <AlertTitle>Couldn’t inspect the RL environment</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
        <Button size="xs" variant="outline" onClick={onRefresh}>
          <RotateCcwIcon />
          Try again
        </Button>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
          <GaugeIcon className="size-3.5" />
          Reproducible training
        </div>
        <h2 className="text-2xl font-semibold tracking-tight">Launch an experiment</h2>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
          The server resolves the runner, environment evidence, and effective configuration before
          training begins. Runs continue even if this page disconnects.
        </p>
      </div>

      {selectedExperiment !== null && selectedRunner?.available !== true ? (
        <Alert variant="warning">
          <AlertCircleIcon />
          <AlertTitle>
            {selectedRunner === null
              ? `${selectedExperiment.runnerId} capability is missing`
              : `${selectedRunner.runnerId} is unavailable`}
          </AlertTitle>
          <AlertDescription>
            {selectedRunner?.remedy ??
              "Verify the runner catalog and dependencies in the server environment."}
          </AlertDescription>
        </Alert>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Run configuration</CardTitle>
          <CardDescription>
            Choose a catalog experiment and an explicit random seed.
          </CardDescription>
          {selectedRunner?.available ? (
            <CardAction>
              <Badge variant="success">
                <CheckCircle2Icon />
                Runner ready
              </Badge>
            </CardAction>
          ) : null}
        </CardHeader>
        <CardPanel>
          {experiments.length === 0 ? (
            <div className="rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">
              This server has no RL experiments in its catalog.
            </div>
          ) : (
            <form className="space-y-5" onSubmit={(event) => void handleSubmit(event)}>
              <div className="grid gap-4 @[32rem]/rl-lab:grid-cols-[minmax(0,1fr)_10rem]">
                <label className="space-y-1.5">
                  <span className="text-sm font-medium">Experiment</span>
                  <Select
                    items={experiments.map((entry) => ({
                      label: entry.displayName,
                      value: entry.experimentId,
                    }))}
                    value={selectedExperiment?.experimentId ?? null}
                    onValueChange={(value) => {
                      if (typeof value !== "string") return;
                      setExperimentOverride(value);
                      setSeedOverride(null);
                      setStartError(null);
                    }}
                  >
                    <SelectTrigger className="w-full" aria-label="RL experiment">
                      <SelectValue>
                        {selectedExperiment?.displayName ?? "Select an experiment"}
                      </SelectValue>
                    </SelectTrigger>
                    <SelectPopup>
                      {experiments.map((experiment) => (
                        <SelectItem key={experiment.experimentId} value={experiment.experimentId}>
                          {experiment.displayName}
                        </SelectItem>
                      ))}
                    </SelectPopup>
                  </Select>
                </label>
                <label className="space-y-1.5">
                  <span className="text-sm font-medium">Seed</span>
                  <Input
                    nativeInput
                    inputMode="numeric"
                    type="number"
                    value={seed}
                    onChange={(event) => {
                      if (selectedExperiment === null) return;
                      setSeedOverride({
                        experimentId: selectedExperiment.experimentId,
                        value: event.currentTarget.value,
                      });
                      setStartError(null);
                    }}
                  />
                </label>
              </div>

              {selectedExperiment !== null ? (
                <ExperimentDescription experiment={selectedExperiment} />
              ) : null}

              {startError !== null ? (
                <Alert variant="error">
                  <AlertCircleIcon />
                  <AlertTitle>Run not started</AlertTitle>
                  <AlertDescription>{startError}</AlertDescription>
                </Alert>
              ) : null}

              <div className="flex justify-end">
                <Button disabled={isStarting || selectedRunner?.available !== true} type="submit">
                  {isStarting ? <Spinner /> : <PlayIcon />}
                  {isStarting ? "Starting…" : "Start run"}
                </Button>
              </div>
            </form>
          )}
        </CardPanel>
      </Card>
    </div>
  );
}

function ExperimentDescription({ experiment }: { readonly experiment: RlExperimentSummary }) {
  return (
    <div className="rounded-xl border border-border/70 bg-muted/18 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-sm">{experiment.displayName}</span>
        <Badge variant="secondary">{experiment.runnerId}</Badge>
        <Badge variant="outline">{experiment.instrumentationLevel} telemetry</Badge>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{experiment.description}</p>
      <details className="mt-3 text-xs">
        <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
          Catalog configuration
        </summary>
        <pre className="mt-2 max-h-64 overflow-auto rounded-lg bg-background p-3 font-mono text-[11px] leading-relaxed">
          {JSON.stringify(experiment.config, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function RunDetail({
  environmentId,
  error,
  isPending,
  projection,
  runId,
  view,
  baselineActionPending,
  isBaseline,
  onRefresh,
  onRunsRefresh,
  onUseAsBaseline,
}: {
  readonly environmentId: EnvironmentId;
  readonly error: string | null;
  readonly isPending: boolean;
  readonly projection: RlRunProjection | null;
  readonly runId: RlRunId;
  readonly view: RunScopedRlLabView;
  readonly baselineActionPending: boolean;
  readonly isBaseline: boolean;
  readonly onRefresh: () => void;
  readonly onRunsRefresh: () => void;
  readonly onUseAsBaseline?: (runId: RlRunId) => void;
}) {
  const [isCancelling, setIsCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const cancelRun = useAtomCommand(rlEnvironment.cancel, { reportFailure: false });

  if (projection === null && isPending) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24 rounded-2xl" />
        <div className="grid gap-4 @[36rem]/rl-lab:grid-cols-2">
          <Skeleton className="h-52 rounded-2xl" />
          <Skeleton className="h-52 rounded-2xl" />
        </div>
      </div>
    );
  }

  if (projection === null) {
    return (
      <Alert variant="error">
        <AlertCircleIcon />
        <AlertTitle>Couldn’t load run {runId}</AlertTitle>
        <AlertDescription>
          {error ?? "The run stream ended before its snapshot arrived."}
        </AlertDescription>
        <Button size="xs" variant="outline" onClick={onRefresh}>
          <RotateCcwIcon />
          Try again
        </Button>
      </Alert>
    );
  }

  const { summary, manifest, artifacts, metrics } = projection;
  const canCancel = !isTerminalRlRunState(summary.state) && summary.state !== "cancelling";
  const handleCancel = async () => {
    if (!canCancel) return;
    setIsCancelling(true);
    setCancelError(null);
    const result = await cancelRun({ environmentId, input: { runId } });
    setIsCancelling(false);
    if (result._tag === "Success") {
      onRunsRefresh();
      return;
    }
    setCancelError(commandFailureMessage(result));
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="gap-y-2">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <CardTitle className="truncate">{summary.experimentId}</CardTitle>
            <Badge variant={rlStatusVariant(summary.state)}>{formatRlState(summary.state)}</Badge>
          </div>
          <CardDescription className="font-mono text-xs">{summary.runId}</CardDescription>
          <CardAction>
            <div className="flex flex-wrap justify-end gap-2">
              {onUseAsBaseline !== undefined ? (
                <Button
                  disabled={isBaseline || baselineActionPending}
                  size="sm"
                  variant="outline"
                  onClick={() => onUseAsBaseline(runId)}
                >
                  {isBaseline ? <CheckCircle2Icon /> : <FlaskConicalIcon />}
                  {baselineActionPending ? "Saving…" : isBaseline ? "Baseline" : "Use as baseline"}
                </Button>
              ) : null}
              {canCancel ? (
                <Button
                  disabled={isCancelling}
                  size="sm"
                  variant="destructive-outline"
                  onClick={() => void handleCancel()}
                >
                  {isCancelling ? <Spinner /> : <SquareIcon />}
                  {isCancelling ? "Cancelling…" : "Cancel"}
                </Button>
              ) : null}
            </div>
          </CardAction>
        </CardHeader>
        <CardPanel>
          <div className="grid gap-4 @[34rem]/rl-lab:grid-cols-2 @[64rem]/rl-lab:grid-cols-4">
            <RunFact icon={<Clock3Icon />} label="Elapsed">
              <RunElapsed summary={summary} />
            </RunFact>
            <RunFact icon={<PlayIcon />} label="Started">
              {formatRunTimestamp(summary.startedAt)}
            </RunFact>
            <RunFact icon={<GaugeIcon />} label="Last message">
              {formatRunTimestamp(summary.lastMessageAt)}
            </RunFact>
            <RunFact icon={<BoxIcon />} label="Outputs">
              {artifacts.length} artifacts · {metrics.length} batches
            </RunFact>
          </div>
        </CardPanel>
      </Card>

      {summary.errorMessage !== null ? (
        <Alert variant="error">
          <AlertCircleIcon />
          <AlertTitle>{summary.errorCode ?? "Run failed"}</AlertTitle>
          <AlertDescription>{summary.errorMessage}</AlertDescription>
        </Alert>
      ) : null}
      {cancelError !== null ? (
        <Alert variant="error">
          <AlertCircleIcon />
          <AlertTitle>Cancellation failed</AlertTitle>
          <AlertDescription>{cancelError}</AlertDescription>
        </Alert>
      ) : null}

      {view === "overview" ? (
        <>
          <section aria-labelledby="rl-metrics-title">
            <div className="mb-3 flex items-end justify-between gap-4">
              <div>
                <h2 className="font-semibold" id="rl-metrics-title">
                  Training metrics
                </h2>
                <p className="mt-1 text-xs text-muted-foreground">
                  Snapshot-backed charts continue from live worker events after reconnection.
                </p>
              </div>
              <Badge variant="secondary">{metrics.length.toLocaleString()} batches</Badge>
            </div>
            <div className="grid gap-3 @[38rem]/rl-lab:grid-cols-2 @[64rem]/rl-lab:grid-cols-4">
              {RL_METRIC_DEFINITIONS.map((definition) => (
                <RlMetricChart definition={definition} key={definition.key} metrics={metrics} />
              ))}
            </div>
          </section>

          <div className="grid gap-4 @[64rem]/rl-lab:grid-cols-2">
            <ManifestCard manifest={manifest} />
            <ArtifactsCard artifacts={artifacts} environmentId={environmentId} runId={runId} />
          </div>
        </>
      ) : view === "behavior" ? (
        <RunBehavior
          artifacts={artifacts}
          environmentId={environmentId}
          runId={runId}
          terminal={isTerminalRlRunState(summary.state)}
        />
      ) : (
        <RunDiagnostics projection={projection} />
      )}
    </div>
  );
}

function RunBehavior({
  artifacts,
  environmentId,
  runId,
  terminal,
}: {
  readonly artifacts: ReadonlyArray<RlArtifactMetadata>;
  readonly environmentId: EnvironmentId;
  readonly runId: RlRunId;
  readonly terminal: boolean;
}) {
  const replay = artifacts.find((artifact) => artifact.kind === "replay");
  if (replay !== undefined) {
    return <TrajectoryViewer artifact={replay} environmentId={environmentId} runId={runId} />;
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Trajectory replay</CardTitle>
        <CardDescription>
          Inspect observations, actions, rewards and episode endings.
        </CardDescription>
      </CardHeader>
      <CardPanel>
        <div className="rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">
          {terminal
            ? "This run did not produce a replay artifact."
            : "The replay appears after the worker completes its final evaluation."}
        </div>
      </CardPanel>
    </Card>
  );
}

function RunDiagnostics({ projection }: { readonly projection: RlRunProjection }) {
  const [nowMs, setNowMs] = useState(() => Date.now());
  const running = projection.summary.state === "running";
  useEffect(() => {
    if (!running) return;
    const timer = globalThis.setInterval(() => setNowMs(Date.now()), 5_000);
    return () => globalThis.clearInterval(timer);
  }, [running]);
  const report = analyzeRlDiagnostics({
    metrics: projection.metrics,
    runState: projection.summary.state,
    lastMessageAt: projection.summary.lastMessageAt,
    nowMs,
  });
  return <RlDiagnosticsPanel report={report} />;
}

function RunFact({
  children,
  icon,
  label,
}: {
  readonly children: ReactNode;
  readonly icon: ReactNode;
  readonly label: string;
}) {
  return (
    <div className="min-w-0 rounded-xl border border-border/60 bg-muted/12 p-3">
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground [&>svg]:size-3.5">
        {icon}
        {label}
      </div>
      <div className="mt-1.5 truncate text-sm font-medium tabular-nums">{children}</div>
    </div>
  );
}

function RunElapsed({ summary }: { readonly summary: RlRunSummary }) {
  const terminal = isTerminalRlRunState(summary.state);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (terminal) return;
    const timer = globalThis.setInterval(() => setNow(Date.now()), 1_000);
    return () => globalThis.clearInterval(timer);
  }, [terminal]);
  return formatRlDuration(summary.startedAt, summary.endedAt, now);
}

function ManifestCard({ manifest }: { readonly manifest: RlResolvedManifest | null }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Resolved manifest</CardTitle>
        <CardDescription>Immutable evidence of what actually ran.</CardDescription>
      </CardHeader>
      <CardPanel>
        {manifest === null ? (
          <div className="rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">
            The runner is still preparing its manifest.
          </div>
        ) : (
          <div className="space-y-4">
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
              <ManifestFact
                label="Runner"
                value={`${manifest.runnerId} ${manifest.runnerVersion}`}
              />
              <ManifestFact label="Python" value={manifest.pythonVersion} />
              <ManifestFact label="Seed" value={String(manifest.seed)} />
              <ManifestFact label="Instrumentation" value={manifest.instrumentationLevel} />
              <ManifestFact
                label="Source revision"
                value={manifest.sourceRevision ?? "Unavailable"}
              />
              <ManifestFact
                label="Source tree"
                value={
                  manifest.sourceDirty === null
                    ? "Unknown"
                    : manifest.sourceDirty
                      ? "Dirty"
                      : "Clean"
                }
              />
            </dl>
            <div>
              <div className="mb-1.5 text-xs font-medium text-muted-foreground">Hardware</div>
              <div className="text-sm">{manifest.hardwareSummary}</div>
            </div>
            <details className="text-xs">
              <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                Effective configuration and environment
              </summary>
              <pre className="mt-2 max-h-80 overflow-auto rounded-lg bg-muted/28 p-3 font-mono text-[11px] leading-relaxed">
                {JSON.stringify(
                  {
                    effectiveConfig: manifest.effectiveConfig,
                    pythonExecutable: manifest.pythonExecutable,
                    environmentFingerprint: manifest.environmentFingerprint,
                    protocolVersion: manifest.protocolVersion,
                  },
                  null,
                  2,
                )}
              </pre>
            </details>
          </div>
        )}
      </CardPanel>
    </Card>
  );
}

function ManifestFact({ label, value }: { readonly label: string; readonly value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 truncate font-medium">{value}</dd>
    </div>
  );
}

function ArtifactsCard({
  artifacts,
  environmentId,
  runId,
}: {
  readonly artifacts: ReadonlyArray<RlArtifactMetadata>;
  readonly environmentId: EnvironmentId;
  readonly runId: RlRunId;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Artifacts</CardTitle>
        <CardDescription>Signed, run-scoped outputs from the server.</CardDescription>
      </CardHeader>
      <CardPanel>
        {artifacts.length === 0 ? (
          <div className="rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">
            Artifacts appear here as the worker produces them.
          </div>
        ) : (
          <div className="divide-y divide-border/60 overflow-hidden rounded-xl border border-border/70">
            {artifacts.map((artifact) => (
              <ArtifactRow
                artifact={artifact}
                environmentId={environmentId}
                key={artifact.artifactId}
                runId={runId}
              />
            ))}
          </div>
        )}
      </CardPanel>
    </Card>
  );
}

function ArtifactRow({
  artifact,
  environmentId,
  runId,
}: {
  readonly artifact: RlArtifactMetadata;
  readonly environmentId: EnvironmentId;
  readonly runId: RlRunId;
}) {
  const urlState = useAssetUrlState(environmentId, {
    _tag: "rl-artifact",
    runId,
    artifactId: artifact.artifactId,
  });
  return (
    <div className="flex min-w-0 items-center gap-3 px-3 py-2.5">
      <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
        <BoxIcon className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate font-medium text-sm">{artifact.kind}</span>
          <Badge size="sm" variant="outline">
            {formatRlBytes(artifact.bytes)}
          </Badge>
        </div>
        <div className="truncate font-mono text-[10px] text-muted-foreground">
          {artifact.artifactId}
        </div>
      </div>
      {urlState._tag === "Success" ? (
        <a
          aria-label={`Open ${artifact.kind} artifact`}
          className={buttonVariants({ size: "icon-xs", variant: "ghost-muted" })}
          href={urlState.url}
          rel="noreferrer"
          target="_blank"
        >
          <ExternalLinkIcon />
          <span className="sr-only">Open {artifact.kind} artifact</span>
        </a>
      ) : (
        <Button
          aria-label={`${artifact.kind} artifact URL ${urlState._tag === "Failure" ? "unavailable" : "loading"}`}
          disabled
          size="icon-xs"
          variant="ghost-muted"
        >
          {urlState._tag === "Failure" ? <AlertCircleIcon /> : <Spinner />}
        </Button>
      )}
    </div>
  );
}
