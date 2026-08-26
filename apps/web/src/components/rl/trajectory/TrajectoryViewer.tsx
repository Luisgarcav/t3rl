import type { EnvironmentId, RlArtifactMetadata, RlRunId } from "@t3tools/contracts";
import {
  AlertCircleIcon,
  ChevronFirstIcon,
  ChevronLastIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  LoaderCircleIcon,
  PauseIcon,
  PlayIcon,
  RotateCcwIcon,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

import { useAssetUrlState } from "~/assets/assetUrls";
import { cn } from "~/lib/utils";

import { Alert, AlertDescription, AlertTitle } from "../../ui/alert";
import { Badge } from "../../ui/badge";
import { Button } from "../../ui/button";
import { Card, CardAction, CardDescription, CardHeader, CardPanel, CardTitle } from "../../ui/card";
import {
  clampTrajectoryIndex,
  DEFAULT_MAX_REPLAY_BYTES,
  DEFAULT_MAX_TRAJECTORY_STEPS,
  loadTrajectoryReplay,
  type TrajectoryJsonValue,
  type TrajectoryReplay,
  type TrajectoryStep,
  trajectoryTerminalKind,
} from "./trajectoryReplay";

const DEFAULT_PLAYBACK_INTERVAL_MS = 750;
const MIN_PLAYBACK_INTERVAL_MS = 100;
const MAX_PLAYBACK_INTERVAL_MS = 10_000;

interface TrajectoryViewerBaseProps {
  readonly environmentId: EnvironmentId;
  readonly runId: RlRunId;
  readonly className?: string | undefined;
  readonly maxBytes?: number | undefined;
  readonly maxSteps?: number | undefined;
  readonly playbackIntervalMs?: number | undefined;
}

export type TrajectoryViewerProps = TrajectoryViewerBaseProps &
  (
    | {
        readonly artifact: RlArtifactMetadata;
        readonly replayUrl?: never;
      }
    | {
        readonly replayUrl: string;
        readonly artifact?: RlArtifactMetadata | undefined;
      }
  );

interface ResolvedViewerProps extends TrajectoryViewerBaseProps {
  readonly replayUrl: string;
  readonly artifact?: RlArtifactMetadata | undefined;
}

type ReplayLoadState =
  | { readonly key: string; readonly status: "loading" }
  | { readonly key: string; readonly status: "ready"; readonly replay: TrajectoryReplay }
  | { readonly key: string; readonly status: "error"; readonly message: string };

function displayError(error: unknown): string {
  return error instanceof Error && error.message.trim().length > 0
    ? error.message
    : "The replay artifact could not be loaded.";
}

function normalizedPlaybackInterval(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) return DEFAULT_PLAYBACK_INTERVAL_MS;
  return Math.min(MAX_PLAYBACK_INTERVAL_MS, Math.max(MIN_PLAYBACK_INTERVAL_MS, Math.round(value)));
}

function artifactValidationMessage(
  artifact: RlArtifactMetadata | undefined,
  maxBytes: number,
): string | null {
  if (artifact === undefined) return null;
  if (artifact.kind !== "replay")
    return `Artifact ${artifact.artifactId} is not a replay artifact.`;
  if (artifact.bytes > maxBytes) {
    return `The replay artifact exceeds the ${maxBytes.toLocaleString()} byte limit.`;
  }
  const contentType = artifact.contentType.split(";", 1)[0]?.trim().toLowerCase();
  if (contentType !== "application/json" && contentType?.endsWith("+json") !== true) {
    return `The replay artifact has unsupported content type ${artifact.contentType}.`;
  }
  return null;
}

function ViewerCard({
  children,
  className,
  description,
  stepCount,
}: {
  readonly children: ReactNode;
  readonly className?: string | undefined;
  readonly description: string;
  readonly stepCount?: number | undefined;
}) {
  return (
    <Card className={cn("min-w-0 overflow-hidden", className)}>
      <CardHeader>
        <CardTitle className="text-base">Trajectory replay</CardTitle>
        <CardDescription>{description}</CardDescription>
        {stepCount !== undefined ? (
          <CardAction>
            <Badge variant="secondary">{stepCount.toLocaleString()} steps</Badge>
          </CardAction>
        ) : null}
      </CardHeader>
      <CardPanel>{children}</CardPanel>
    </Card>
  );
}

function LoadingViewer({ className }: { readonly className?: string | undefined }) {
  return (
    <ViewerCard className={className} description="Loading and validating replay.json…">
      <div
        aria-label="Loading trajectory replay"
        className="flex min-h-48 items-center justify-center gap-2 rounded-xl border border-dashed text-sm text-muted-foreground"
        role="status"
      >
        <LoaderCircleIcon className="size-4 animate-spin" />
        Loading trajectory
      </div>
    </ViewerCard>
  );
}

function ErrorViewer({
  className,
  message,
  onRetry,
}: {
  readonly className?: string | undefined;
  readonly message: string;
  readonly onRetry?: (() => void) | undefined;
}) {
  return (
    <ViewerCard className={className} description="The replay is unavailable.">
      <Alert variant="error">
        <AlertCircleIcon />
        <AlertTitle>Couldn’t open trajectory</AlertTitle>
        <AlertDescription>{message}</AlertDescription>
        {onRetry === undefined ? null : (
          <Button size="xs" variant="outline" onClick={onRetry}>
            <RotateCcwIcon />
            Try again
          </Button>
        )}
      </Alert>
    </ViewerCard>
  );
}

function ArtifactTrajectoryViewer(
  props: TrajectoryViewerBaseProps & { readonly artifact: RlArtifactMetadata },
) {
  const urlState = useAssetUrlState(props.environmentId, {
    _tag: "rl-artifact",
    runId: props.runId,
    artifactId: props.artifact.artifactId,
  });
  const maxBytes = props.maxBytes ?? DEFAULT_MAX_REPLAY_BYTES;
  const metadataError = artifactValidationMessage(props.artifact, maxBytes);
  if (metadataError !== null) {
    return <ErrorViewer className={props.className} message={metadataError} />;
  }
  if (urlState._tag === "Failure") {
    return (
      <ErrorViewer
        className={props.className}
        message="The server could not create a URL for this run-scoped artifact."
      />
    );
  }
  if (urlState._tag !== "Success") return <LoadingViewer className={props.className} />;
  return <ResolvedTrajectoryViewer {...props} replayUrl={urlState.url} />;
}

function ResolvedTrajectoryViewer({
  artifact,
  className,
  environmentId,
  maxBytes = DEFAULT_MAX_REPLAY_BYTES,
  maxSteps = DEFAULT_MAX_TRAJECTORY_STEPS,
  playbackIntervalMs,
  replayUrl,
  runId,
}: ResolvedViewerProps) {
  const metadataError = artifactValidationMessage(artifact, maxBytes);
  const [retry, setRetry] = useState(0);
  const loadKey = `${replayUrl}\u0000${maxBytes}\u0000${maxSteps}\u0000${retry}`;
  const [loadState, setLoadState] = useState<ReplayLoadState>({
    key: loadKey,
    status: "loading",
  });
  const currentState: ReplayLoadState =
    loadState.key === loadKey ? loadState : { key: loadKey, status: "loading" };

  useEffect(() => {
    if (metadataError !== null) return;
    const controller = new AbortController();
    void loadTrajectoryReplay(replayUrl, {
      maxBytes,
      maxSteps,
      signal: controller.signal,
    }).then(
      (replay) => setLoadState({ key: loadKey, status: "ready", replay }),
      (error: unknown) => {
        if (!controller.signal.aborted) {
          setLoadState({ key: loadKey, status: "error", message: displayError(error) });
        }
      },
    );
    return () => controller.abort();
  }, [loadKey, maxBytes, maxSteps, metadataError, replayUrl]);

  if (metadataError !== null) return <ErrorViewer className={className} message={metadataError} />;
  if (currentState.status === "loading") return <LoadingViewer className={className} />;
  if (currentState.status === "error") {
    return (
      <ErrorViewer
        className={className}
        message={currentState.message}
        onRetry={() => setRetry((value) => value + 1)}
      />
    );
  }
  return (
    <TrajectoryPlayback
      className={className}
      environmentId={environmentId}
      intervalMs={normalizedPlaybackInterval(playbackIntervalMs)}
      replay={currentState.replay}
      runId={runId}
    />
  );
}

function TrajectoryPlayback({
  className,
  environmentId,
  intervalMs,
  replay,
  runId,
}: {
  readonly className?: string | undefined;
  readonly environmentId: EnvironmentId;
  readonly intervalMs: number;
  readonly replay: TrajectoryReplay;
  readonly runId: RlRunId;
}) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const lastIndex = replay.trajectory.length - 1;
  const step = replay.trajectory[currentIndex] ?? replay.trajectory[0];

  useEffect(() => {
    if (!isPlaying || currentIndex >= lastIndex) return;
    const nextIndex = currentIndex + 1;
    const timer = globalThis.setTimeout(() => {
      setCurrentIndex(nextIndex);
      if (nextIndex >= lastIndex) setIsPlaying(false);
    }, intervalMs);
    return () => globalThis.clearTimeout(timer);
  }, [currentIndex, intervalMs, isPlaying, lastIndex]);

  if (step === undefined) {
    return <ErrorViewer className={className} message="The replay contains no trajectory steps." />;
  }

  const moveTo = (index: number) => {
    setIsPlaying(false);
    setCurrentIndex(clampTrajectoryIndex(index, replay.trajectory.length));
  };
  const togglePlayback = () => {
    if (isPlaying) {
      setIsPlaying(false);
      return;
    }
    if (currentIndex >= lastIndex) setCurrentIndex(0);
    setIsPlaying(true);
  };

  return (
    <ViewerCard
      className={className}
      description={`${replay.environment} · evaluation seed ${replay.evaluationSeed.toLocaleString()}`}
      stepCount={replay.trajectory.length}
    >
      <div className="space-y-4">
        <div className="flex min-w-0 flex-wrap items-center justify-between gap-2 rounded-xl border border-border/70 bg-muted/12 px-3 py-2">
          <div className="min-w-0">
            <div className="font-mono text-xs font-semibold tabular-nums">
              Step {step.step.toLocaleString()}
            </div>
            <div className="truncate font-mono text-[10px] text-muted-foreground">
              {runId} · {environmentId}
            </div>
          </div>
          <Badge variant={terminalBadgeVariant(step)}>{terminalLabel(step)}</Badge>
        </div>

        <div className="grid gap-3 md:grid-cols-[minmax(0,2fr)_minmax(12rem,1fr)]">
          <ReplayValue label="Observation" value={step.observation} />
          <div className="grid grid-cols-2 gap-3 md:grid-cols-1">
            <ReplayValue compact label="Action" value={step.action} />
            <div className="rounded-xl border border-border/70 bg-muted/12 p-3">
              <div className="text-xs font-medium text-muted-foreground">Reward</div>
              <div className="mt-2 font-mono text-lg font-semibold tabular-nums">
                {step.reward.toLocaleString(undefined, { maximumFractionDigits: 6 })}
              </div>
            </div>
          </div>
        </div>

        <dl className="grid grid-cols-2 gap-3 text-sm">
          <BooleanFact label="Terminated" value={step.terminated} />
          <BooleanFact label="Truncated" value={step.truncated} />
        </dl>

        <div className="space-y-3 rounded-xl border border-border/70 p-3">
          <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
            <span>Frame {currentIndex + 1}</span>
            <span>{replay.trajectory.length.toLocaleString()} total</span>
          </div>
          <input
            aria-label="Trajectory frame"
            className="h-2 w-full cursor-pointer accent-primary disabled:cursor-default"
            max={lastIndex}
            min={0}
            step={1}
            type="range"
            value={currentIndex}
            onChange={(event) => moveTo(Number(event.currentTarget.value))}
          />
          <div className="flex items-center justify-center gap-1.5">
            <Button
              aria-label="First trajectory step"
              disabled={currentIndex === 0}
              size="icon-sm"
              variant="outline"
              onClick={() => moveTo(0)}
            >
              <ChevronFirstIcon />
            </Button>
            <Button
              aria-label="Previous trajectory step"
              disabled={currentIndex === 0}
              size="icon-sm"
              variant="outline"
              onClick={() => moveTo(currentIndex - 1)}
            >
              <ChevronLeftIcon />
            </Button>
            <Button
              aria-label={isPlaying ? "Pause trajectory" : "Play trajectory"}
              aria-pressed={isPlaying}
              size="sm"
              onClick={togglePlayback}
            >
              {isPlaying ? <PauseIcon /> : <PlayIcon />}
              {isPlaying ? "Pause" : currentIndex >= lastIndex ? "Replay" : "Play"}
            </Button>
            <Button
              aria-label="Next trajectory step"
              disabled={currentIndex === lastIndex}
              size="icon-sm"
              variant="outline"
              onClick={() => moveTo(currentIndex + 1)}
            >
              <ChevronRightIcon />
            </Button>
            <Button
              aria-label="Last trajectory step"
              disabled={currentIndex === lastIndex}
              size="icon-sm"
              variant="outline"
              onClick={() => moveTo(lastIndex)}
            >
              <ChevronLastIcon />
            </Button>
          </div>
        </div>
      </div>
    </ViewerCard>
  );
}

function ReplayValue({
  compact = false,
  label,
  value,
}: {
  readonly compact?: boolean;
  readonly label: string;
  readonly value: TrajectoryJsonValue;
}) {
  return (
    <div className="min-w-0 rounded-xl border border-border/70 bg-muted/12 p-3">
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <pre
        className={cn(
          "mt-2 overflow-auto whitespace-pre-wrap break-all font-mono text-xs leading-relaxed",
          compact ? "max-h-24" : "max-h-56 min-h-24",
        )}
      >
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

function BooleanFact({ label, value }: { readonly label: string; readonly value: boolean }) {
  return (
    <div className="rounded-xl border border-border/70 px-3 py-2">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 font-mono font-medium">{String(value)}</dd>
    </div>
  );
}

function terminalLabel(step: TrajectoryStep): string {
  switch (trajectoryTerminalKind(step)) {
    case "terminated":
      return "Terminated";
    case "truncated":
      return "Truncated";
    case "continuing":
      return "Continuing";
  }
}

function terminalBadgeVariant(step: TrajectoryStep): "secondary" | "success" | "warning" {
  switch (trajectoryTerminalKind(step)) {
    case "terminated":
      return "success";
    case "truncated":
      return "warning";
    case "continuing":
      return "secondary";
  }
}

export function TrajectoryViewer(props: TrajectoryViewerProps) {
  if (props.replayUrl !== undefined) return <ResolvedTrajectoryViewer {...props} />;
  return <ArtifactTrajectoryViewer {...props} />;
}
