import type { RlMetricBatch, RlResolvedManifest } from "@t3tools/contracts";
import {
  ArrowRightIcon,
  ChevronFirstIcon,
  ChevronLastIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  GitBranchIcon,
  PauseIcon,
  PlayIcon,
  Repeat2Icon,
} from "lucide-react";
import { useEffect, useState } from "react";

import { cn } from "~/lib/utils";

import { Badge } from "../../ui/badge";
import { Button } from "../../ui/button";
import { Card, CardAction, CardDescription, CardHeader, CardPanel, CardTitle } from "../../ui/card";
import { formatRlMetricValue, latestRlMetricValue } from "../rlPresentation";
import {
  type VisualizationMode,
  VisualizationModeTabs,
} from "../visualization/VisualizationModeTabs";
import {
  clampRlAlgorithmStageIndex,
  type RlAlgorithmSpecSource,
  type RlAlgorithmStage,
  resolveRlAlgorithmVisualization,
} from "./algorithmVisualization";

const ALGORITHM_PLAYBACK_INTERVAL_MS = 1_100;

function sourceLabel(source: RlAlgorithmSpecSource): string {
  switch (source) {
    case "resolved-manifest":
      return "Resolved manifest";
    case "experiment-id":
      return "Inferred from experiment";
    case "generic":
      return "Generic flow";
  }
}

function AlgorithmFlow({
  currentIndex,
  stages,
  onSelect,
}: {
  readonly currentIndex: number;
  readonly stages: ReadonlyArray<RlAlgorithmStage>;
  readonly onSelect: (index: number) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-xl border border-border/70 bg-muted/12 p-4">
      <div className="flex min-w-max items-stretch">
        {stages.map((stage, index) => (
          <div className="flex items-center" key={stage.id}>
            <button
              aria-current={index === currentIndex ? "step" : undefined}
              className={cn(
                "flex w-48 shrink-0 flex-col rounded-xl border p-3 text-left outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring",
                index === currentIndex
                  ? "border-primary/60 bg-primary/10 text-foreground"
                  : "border-border/70 bg-card hover:bg-accent",
              )}
              type="button"
              onClick={() => onSelect(index)}
            >
              <span className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                Stage {index + 1}
              </span>
              <span className="mt-1 font-semibold text-sm">{stage.label}</span>
              <span className="mt-2 line-clamp-3 text-xs leading-relaxed text-muted-foreground">
                {stage.description}
              </span>
            </button>
            {index < stages.length - 1 ? (
              <ArrowRightIcon className="mx-2 size-4 shrink-0 text-muted-foreground" />
            ) : null}
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-center justify-end gap-1.5 text-[11px] text-muted-foreground">
        <Repeat2Icon className="size-3.5" />
        Evaluation feeds the next collection cycle
      </div>
    </div>
  );
}

function StageEvidence({
  metrics,
  stage,
}: {
  readonly metrics: ReadonlyArray<RlMetricBatch>;
  readonly stage: RlAlgorithmStage;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="rounded-xl border border-border/70 bg-muted/12 p-4">
        <div className="text-xs font-medium text-muted-foreground">Input</div>
        <p className="mt-2 text-sm leading-relaxed">{stage.input}</p>
      </div>
      <div className="rounded-xl border border-border/70 bg-muted/12 p-4">
        <div className="text-xs font-medium text-muted-foreground">Output</div>
        <p className="mt-2 text-sm leading-relaxed">{stage.output}</p>
      </div>
      <div className="rounded-xl border border-border/70 bg-muted/12 p-4 sm:col-span-2">
        <div className="text-xs font-medium text-muted-foreground">Evidence in this run</div>
        {stage.evidenceMetricKeys.length === 0 ? (
          <p className="mt-2 text-sm text-muted-foreground">
            This stage has no direct scalar in the current bounded metric contract.
          </p>
        ) : (
          <dl className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {stage.evidenceMetricKeys.map((metricKey) => (
              <div className="rounded-lg bg-background px-3 py-2" key={metricKey}>
                <dt className="truncate font-mono text-[10px] text-muted-foreground">
                  {metricKey}
                </dt>
                <dd className="mt-1 font-mono text-sm font-semibold tabular-nums">
                  {formatRlMetricValue(latestRlMetricValue(metrics, metricKey))}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </div>
    </div>
  );
}

export function RlAlgorithmVisualizer({
  experimentId,
  manifest,
  metrics,
}: {
  readonly experimentId: string;
  readonly manifest: RlResolvedManifest | null;
  readonly metrics: ReadonlyArray<RlMetricBatch>;
}) {
  const spec = resolveRlAlgorithmVisualization(manifest, experimentId);
  const [mode, setMode] = useState<VisualizationMode>("visual");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const lastIndex = spec.stages.length - 1;
  const activeIndex = clampRlAlgorithmStageIndex(currentIndex, spec.stages.length);
  const currentStage = spec.stages[activeIndex]!;

  useEffect(() => {
    if (!isPlaying || activeIndex >= lastIndex) return;
    const nextIndex = activeIndex + 1;
    const timer = globalThis.setTimeout(() => {
      setCurrentIndex(nextIndex);
      if (nextIndex >= lastIndex) setIsPlaying(false);
    }, ALGORITHM_PLAYBACK_INTERVAL_MS);
    return () => globalThis.clearTimeout(timer);
  }, [activeIndex, isPlaying, lastIndex]);

  const moveTo = (index: number) => {
    setIsPlaying(false);
    setCurrentIndex(clampRlAlgorithmStageIndex(index, spec.stages.length));
  };
  const togglePlayback = () => {
    if (isPlaying) {
      setIsPlaying(false);
      return;
    }
    if (activeIndex >= lastIndex) setCurrentIndex(0);
    setIsPlaying(true);
  };

  return (
    <Card className="min-w-0 overflow-hidden">
      <CardHeader>
        <div className="flex min-w-0 items-center gap-2">
          <GitBranchIcon className="size-4 shrink-0 text-muted-foreground" />
          <CardTitle className="truncate text-base">{spec.algorithm} algorithm flow</CardTitle>
        </div>
        <CardDescription>{spec.summary}</CardDescription>
        <CardAction>
          <Badge variant="secondary">{spec.family}</Badge>
        </CardAction>
      </CardHeader>
      <CardPanel>
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <Badge variant="outline">{sourceLabel(spec.source)}</Badge>
              <span>
                Conceptual structure; the highlight is an explanation step, not live execution.
              </span>
            </div>
            <VisualizationModeTabs
              label="Algorithm visualizer display"
              value={mode}
              onChange={setMode}
            />
          </div>

          <div aria-label={`${spec.algorithm} ${mode}`} role="tabpanel">
            {mode === "visual" ? (
              <AlgorithmFlow currentIndex={activeIndex} stages={spec.stages} onSelect={moveTo} />
            ) : mode === "data" ? (
              <StageEvidence metrics={metrics} stage={currentStage} />
            ) : (
              <pre className="max-h-[32rem] overflow-auto rounded-xl border border-border/70 bg-muted/18 p-4 font-mono text-[11px] leading-relaxed">
                {JSON.stringify(spec, null, 2)}
              </pre>
            )}
          </div>

          <div className="space-y-3 rounded-xl border border-border/70 p-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                  Stage {activeIndex + 1} of {spec.stages.length}
                </div>
                <div className="mt-1 font-semibold text-sm">{currentStage.label}</div>
                <p className="mt-1 max-w-3xl text-xs leading-relaxed text-muted-foreground">
                  {currentStage.description}
                </p>
              </div>
              <Badge variant={activeIndex === lastIndex ? "success" : "info"}>
                {activeIndex === lastIndex ? "Evaluation" : "Learning loop"}
              </Badge>
            </div>
            <input
              aria-label="Algorithm explanation stage"
              className="h-2 w-full cursor-pointer accent-primary"
              max={lastIndex}
              min={0}
              step={1}
              type="range"
              value={activeIndex}
              onChange={(event) => moveTo(Number(event.currentTarget.value))}
            />
            <div className="flex items-center justify-center gap-1.5">
              <Button
                aria-label="First algorithm stage"
                disabled={activeIndex === 0}
                size="icon-sm"
                variant="outline"
                onClick={() => moveTo(0)}
              >
                <ChevronFirstIcon />
              </Button>
              <Button
                aria-label="Previous algorithm stage"
                disabled={activeIndex === 0}
                size="icon-sm"
                variant="outline"
                onClick={() => moveTo(activeIndex - 1)}
              >
                <ChevronLeftIcon />
              </Button>
              <Button
                aria-label={
                  isPlaying ? "Pause algorithm explanation" : "Play algorithm explanation"
                }
                size="icon-sm"
                variant="secondary"
                onClick={togglePlayback}
              >
                {isPlaying ? <PauseIcon /> : <PlayIcon />}
              </Button>
              <Button
                aria-label="Next algorithm stage"
                disabled={activeIndex === lastIndex}
                size="icon-sm"
                variant="outline"
                onClick={() => moveTo(activeIndex + 1)}
              >
                <ChevronRightIcon />
              </Button>
              <Button
                aria-label="Last algorithm stage"
                disabled={activeIndex === lastIndex}
                size="icon-sm"
                variant="outline"
                onClick={() => moveTo(lastIndex)}
              >
                <ChevronLastIcon />
              </Button>
            </div>
          </div>
        </div>
      </CardPanel>
    </Card>
  );
}
