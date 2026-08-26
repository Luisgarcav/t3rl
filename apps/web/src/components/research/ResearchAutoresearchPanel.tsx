import {
  RESEARCH_MAX_RUN_BUDGET,
  RESEARCH_MAX_SELECTED_PROFILES,
  RESEARCH_MAX_WALL_CLOCK_MINUTES,
  RESEARCH_WORKSPACE_FILE,
  ResearchAlgorithmId,
  RlRunId,
  type EnvironmentId,
  type ProjectId,
  type ResearchStudyDraft,
  type ResearchWorkspaceDocument,
} from "@t3tools/contracts";
import { buildAutoresearchIterationPrompt } from "@t3tools/client-runtime/research";
import {
  RESEARCH_ALGORITHM_CATALOG,
  RESEARCH_ALGORITHM_FAMILIES,
  RESEARCH_REWARD_SOURCE_OPTIONS,
  RESEARCH_TASK_TYPE_OPTIONS,
  researchAlgorithmById,
  researchAlgorithmFamily,
  researchExecutionStatus,
} from "@t3tools/client-runtime/research-algorithms";
import {
  AlertCircleIcon,
  BrainCircuitIcon,
  CheckIcon,
  ExternalLinkIcon,
  FileCode2Icon,
  FlaskConicalIcon,
  SaveIcon,
  SendIcon,
  Settings2Icon,
  ShieldCheckIcon,
  WorkflowIcon,
} from "lucide-react";
import { useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "~/components/ui/alert";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Checkbox } from "~/components/ui/checkbox";
import { Input } from "~/components/ui/input";
import {
  Select,
  SelectItem,
  SelectPopup,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Skeleton } from "~/components/ui/skeleton";
import { Textarea } from "~/components/ui/textarea";
import { Tooltip, TooltipPopup, TooltipTrigger } from "~/components/ui/tooltip";
import { useEnvironmentQuery } from "~/state/query";
import { rlEnvironment } from "~/state/rl";

import { formatResearchRunEvidence } from "./researchEvidence";
import { parseResearchSeeds } from "./researchStudy";
import { useResearchWorkspaceFile } from "./researchWorkspaceFile";

const NO_BASELINE_VALUE = "__no_baseline__";

export function ResearchAutoresearchPanel(props: {
  readonly environmentId: EnvironmentId;
  readonly projectId: ProjectId;
  readonly cwd: string;
  readonly onPreparePrompt: (prompt: string) => void;
  readonly onOpenBaseline: (runId: RlRunId) => void;
  readonly onOpenSpecialists: () => void;
}) {
  const workspace = useResearchWorkspaceFile(props.environmentId, props.cwd);
  if (workspace.isPending) {
    return (
      <div className="space-y-3 p-4">
        <Skeleton className="h-20 rounded-xl" />
        <Skeleton className="h-80 rounded-xl" />
      </div>
    );
  }
  return (
    <ResearchAutoresearchEditor
      environmentId={props.environmentId}
      projectId={props.projectId}
      initialDocument={workspace.document}
      parseError={workspace.parseError}
      readNotice={workspace.readNotice}
      onPreparePrompt={props.onPreparePrompt}
      onOpenBaseline={props.onOpenBaseline}
      onOpenSpecialists={props.onOpenSpecialists}
      onSave={workspace.save}
    />
  );
}

function ResearchAutoresearchEditor(props: {
  readonly environmentId: EnvironmentId;
  readonly projectId: ProjectId;
  readonly initialDocument: ResearchWorkspaceDocument;
  readonly readNotice: string | null;
  readonly parseError: string | null;
  readonly onPreparePrompt: (prompt: string) => void;
  readonly onOpenBaseline: (runId: RlRunId) => void;
  readonly onOpenSpecialists: () => void;
  readonly onSave: (document: ResearchWorkspaceDocument) => Promise<string | null>;
}) {
  const [document, setDocument] = useState(props.initialDocument);
  const [seedText, setSeedText] = useState(document.studyDraft.seeds.join(", "));
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");
  const [error, setError] = useState<string | null>(props.parseError);
  const runs = useEnvironmentQuery(
    rlEnvironment.runs({
      environmentId: props.environmentId,
      input: { projectId: props.projectId, limit: 200 },
    }),
  );
  const baselineRunId = document.studyDraft.baselineRunId;
  const baseline = useEnvironmentQuery(
    baselineRunId === null
      ? null
      : rlEnvironment.run({
          environmentId: props.environmentId,
          input: { runId: baselineRunId },
        }),
  );
  const selectedAlgorithm = researchAlgorithmById(document.studyDraft.target.algorithmId);
  const selectedFamily = researchAlgorithmFamily(selectedAlgorithm.family);
  const familyAlgorithms = RESEARCH_ALGORITHM_CATALOG.filter(
    (entry) => entry.family === selectedAlgorithm.family,
  );
  const executionStatus = researchExecutionStatus(
    selectedAlgorithm,
    document.studyDraft.target.taskType,
  );

  const updateStudy = (patch: Partial<ResearchStudyDraft>) => {
    setSaveState("idle");
    setDocument((current) => ({
      ...current,
      studyDraft: { ...current.studyDraft, ...patch },
    }));
  };

  const updateTarget = (patch: Partial<ResearchStudyDraft["target"]>) => {
    updateStudy({ target: { ...document.studyDraft.target, ...patch } });
  };

  const resolvedDocument = (
    requireReady: boolean,
  ): {
    document: ResearchWorkspaceDocument | null;
    error: string | null;
  } => {
    if (requireReady && document.studyDraft.objective.trim().length === 0) {
      return { document: null, error: "Describe the research objective." };
    }
    if (requireReady && document.studyDraft.successMetric.trim().length === 0) {
      return { document: null, error: "Choose an explicit success metric." };
    }
    const parsedSeeds = parseResearchSeeds(seedText, document.studyDraft.maxRuns);
    if (parsedSeeds.error !== null) return { document: null, error: parsedSeeds.error };
    return {
      document: {
        ...document,
        studyDraft: { ...document.studyDraft, seeds: parsedSeeds.seeds },
      },
      error: null,
    };
  };

  const persist = async (requireReady: boolean): Promise<ResearchWorkspaceDocument | null> => {
    const resolved = resolvedDocument(requireReady);
    if (resolved.document === null) {
      setError(resolved.error);
      return null;
    }
    setSaveState("saving");
    setError(null);
    const failure = await props.onSave(resolved.document);
    if (failure !== null) {
      setSaveState("idle");
      setError(failure);
      return null;
    }
    setDocument(resolved.document);
    setSaveState("saved");
    return resolved.document;
  };

  const prepareIteration = async () => {
    if (baselineRunId !== null && baseline.data === null) {
      setError(
        baseline.error ??
          (baseline.isPending
            ? "The baseline evidence is still loading."
            : "The selected baseline run is unavailable."),
      );
      return;
    }
    const persisted = await persist(true);
    if (persisted === null) return;
    props.onPreparePrompt(
      buildAutoresearchIterationPrompt({
        study: persisted.studyDraft,
        profiles: persisted.profiles,
        baselineEvidence:
          baselineRunId === null || baseline.data === null
            ? null
            : {
                runId: baselineRunId,
                summary: formatResearchRunEvidence(baseline.data),
              },
      }),
    );
  };

  const openBaseline = async () => {
    if (baselineRunId === null) return;
    const persisted = await persist(false);
    if (persisted !== null) props.onOpenBaseline(baselineRunId);
  };

  const openSpecialists = async () => {
    const persisted = await persist(false);
    if (persisted !== null) props.onOpenSpecialists();
  };

  const toggleProfile = (profileId: (typeof document.profiles)[number]["id"], checked: boolean) => {
    const current = document.studyDraft.specialistProfileIds;
    if (checked && !current.includes(profileId)) {
      if (current.length >= RESEARCH_MAX_SELECTED_PROFILES) return;
      updateStudy({ specialistProfileIds: [...current, profileId] });
      return;
    }
    if (!checked) {
      updateStudy({ specialistProfileIds: current.filter((entry) => entry !== profileId) });
    }
  };

  const runOptions = runs.data?.runs ?? [];
  const baselineValue = baselineRunId ?? NO_BASELINE_VALUE;

  return (
    <div className="@container/autoresearch flex min-h-0 flex-1 flex-col bg-background">
      <header className="flex shrink-0 items-center gap-3 border-b border-border px-4 py-3">
        <WorkflowIcon className="size-4 text-muted-foreground" />
        <div className="min-w-0">
          <h2 className="text-sm font-semibold">Autoresearch</h2>
          <p className="truncate text-xs text-muted-foreground">
            Prepare one bounded iteration, then review it before any side effect.
          </p>
        </div>
        <Badge
          aria-label={`Stored in ${RESEARCH_WORKSPACE_FILE}`}
          className="ml-auto text-[10px]"
          title={RESEARCH_WORKSPACE_FILE}
          variant="secondary"
        >
          <FileCode2Icon />
          <span className="hidden @[30rem]/autoresearch:inline">Project config</span>
        </Badge>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-3 @[38rem]/autoresearch:p-4">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          <div className="order-0 flex items-start gap-2 rounded-lg border border-info/20 bg-info/8 px-3 py-2 text-xs leading-5 text-muted-foreground">
            <ShieldCheckIcon className="mt-0.5 size-3.5 shrink-0 text-info" />
            <span>
              <strong className="font-medium text-foreground">Review required.</strong> The turn
              prepares a falsifiable hypothesis, proposed diff, and run plan. Changes and training
              still require explicit approval.
            </span>
          </div>
          {props.readNotice !== null ? (
            <Alert className="order-0" variant="info">
              <FileCode2Icon />
              <AlertDescription>{props.readNotice}</AlertDescription>
            </Alert>
          ) : null}
          {error !== null ? (
            <Alert className="order-0" variant="error">
              <AlertCircleIcon />
              <AlertTitle>Research iteration not ready</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          <section className="space-y-4 rounded-xl border border-border bg-card p-4">
            <div>
              <h3 className="text-sm font-semibold">Scientific objective</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Define what should improve and the evidence that would count.
              </p>
            </div>
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">Objective</span>
              <Textarea
                maxLength={2_000}
                placeholder="Example: improve robust evaluation return without increasing KL instability."
                rows={4}
                value={document.studyDraft.objective}
                onChange={(event) => updateStudy({ objective: event.currentTarget.value })}
              />
            </label>
            <div className="grid gap-3 @[30rem]/autoresearch:grid-cols-[9rem_minmax(0,1fr)]">
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Direction</span>
                <Select
                  items={[
                    { label: "Increase", value: "increase" },
                    { label: "Decrease", value: "decrease" },
                  ]}
                  value={document.studyDraft.direction}
                  onValueChange={(value) => {
                    if (value === "increase" || value === "decrease")
                      updateStudy({ direction: value });
                  }}
                >
                  <SelectTrigger className="w-full" aria-label="Success direction">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectPopup>
                    <SelectItem value="increase">Increase</SelectItem>
                    <SelectItem value="decrease">Decrease</SelectItem>
                  </SelectPopup>
                </Select>
              </label>
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Success metric</span>
                <Input
                  maxLength={160}
                  nativeInput
                  placeholder="eval/mean_return"
                  value={document.studyDraft.successMetric}
                  onChange={(event) => updateStudy({ successMetric: event.currentTarget.value })}
                />
              </label>
            </div>
          </section>

          <section className="space-y-4 rounded-xl border border-border bg-card p-4">
            <div className="flex items-center gap-2">
              <FlaskConicalIcon className="size-4 text-muted-foreground" />
              <div>
                <h3 className="text-sm font-semibold">Baseline evidence</h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  Anchor the next hypothesis in an existing run when possible.
                </p>
              </div>
            </div>
            <div className="flex items-end gap-2">
              <label className="min-w-0 flex-1 space-y-1.5">
                <span className="text-sm font-medium">Baseline run</span>
                <Select
                  items={[
                    { label: "No baseline", value: NO_BASELINE_VALUE },
                    ...runOptions.map((run) => ({
                      label: `${run.experimentId} · ${run.runId}`,
                      value: run.runId,
                    })),
                  ]}
                  value={baselineValue}
                  onValueChange={(value) => {
                    if (typeof value !== "string") return;
                    updateStudy({
                      baselineRunId: value === NO_BASELINE_VALUE ? null : RlRunId.make(value),
                    });
                  }}
                >
                  <SelectTrigger className="w-full" aria-label="Baseline RL run">
                    <SelectValue>
                      {baselineRunId === null ? "No baseline" : baselineRunId}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectPopup>
                    <SelectItem value={NO_BASELINE_VALUE}>No baseline</SelectItem>
                    {runOptions.map((run) => (
                      <SelectItem key={run.runId} value={run.runId}>
                        {run.experimentId} · {run.runId}
                      </SelectItem>
                    ))}
                  </SelectPopup>
                </Select>
              </label>
              {baselineRunId !== null ? (
                <Button
                  disabled={baseline.isPending || saveState === "saving"}
                  size="sm"
                  variant="outline"
                  onClick={() => void openBaseline()}
                >
                  <ExternalLinkIcon />
                  <span className="hidden @[28rem]/autoresearch:inline">Open run</span>
                </Button>
              ) : null}
            </div>
            <p className="text-[11px] leading-4 text-muted-foreground">
              The composer receives a bounded manifest and metric summary, not screenshots.
            </p>
          </section>

          <section className="space-y-4 rounded-xl border border-border bg-card p-4">
            <div className="flex items-center gap-2">
              <BrainCircuitIcon className="size-4 text-muted-foreground" />
              <div>
                <h3 className="text-sm font-semibold">RL target</h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  Algorithm, task topology, and reward regime are modeled independently.
                </p>
              </div>
            </div>
            <div className="grid gap-3 @[34rem]/autoresearch:grid-cols-2">
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Algorithm type</span>
                <Select
                  items={RESEARCH_ALGORITHM_FAMILIES.map((family) => ({
                    label: family.label,
                    value: family.id,
                  }))}
                  value={selectedAlgorithm.family}
                  onValueChange={(value) => {
                    const first = RESEARCH_ALGORITHM_CATALOG.find(
                      (entry) => entry.family === value,
                    );
                    if (first !== undefined) updateTarget({ algorithmId: first.id });
                  }}
                >
                  <SelectTrigger className="w-full" aria-label="Research algorithm type">
                    <SelectValue>{selectedFamily.label}</SelectValue>
                  </SelectTrigger>
                  <SelectPopup>
                    {RESEARCH_ALGORITHM_FAMILIES.map((family) => (
                      <SelectItem key={family.id} value={family.id}>
                        {family.label}
                      </SelectItem>
                    ))}
                  </SelectPopup>
                </Select>
              </label>
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Algorithm or method</span>
                <Select
                  items={familyAlgorithms.map((entry) => ({
                    label: entry.name,
                    value: entry.id,
                  }))}
                  value={document.studyDraft.target.algorithmId}
                  onValueChange={(value) => {
                    if (typeof value === "string") {
                      updateTarget({ algorithmId: ResearchAlgorithmId.make(value) });
                    }
                  }}
                >
                  <SelectTrigger className="w-full" aria-label="Research algorithm">
                    <SelectValue>{selectedAlgorithm.name}</SelectValue>
                  </SelectTrigger>
                  <SelectPopup>
                    {familyAlgorithms.map((entry) => (
                      <SelectItem key={entry.id} value={entry.id}>
                        {entry.name}
                      </SelectItem>
                    ))}
                  </SelectPopup>
                </Select>
              </label>
            </div>
            <div className="grid gap-3 @[34rem]/autoresearch:grid-cols-2">
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Task type</span>
                <Select
                  items={RESEARCH_TASK_TYPE_OPTIONS}
                  value={document.studyDraft.target.taskType}
                  onValueChange={(value) => {
                    const option = RESEARCH_TASK_TYPE_OPTIONS.find(
                      (candidate) => candidate.value === value,
                    );
                    if (option !== undefined) updateTarget({ taskType: option.value });
                  }}
                >
                  <SelectTrigger className="w-full" aria-label="RL task type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectPopup>
                    {RESEARCH_TASK_TYPE_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectPopup>
                </Select>
              </label>
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Reward source</span>
                <Select
                  items={RESEARCH_REWARD_SOURCE_OPTIONS}
                  value={document.studyDraft.target.rewardSource}
                  onValueChange={(value) => {
                    const option = RESEARCH_REWARD_SOURCE_OPTIONS.find(
                      (candidate) => candidate.value === value,
                    );
                    if (option !== undefined) updateTarget({ rewardSource: option.value });
                  }}
                >
                  <SelectTrigger className="w-full" aria-label="RL reward source">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectPopup>
                    {RESEARCH_REWARD_SOURCE_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectPopup>
                </Select>
              </label>
            </div>
            <div className="space-y-2 rounded-lg border border-border bg-muted/15 p-3">
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge variant="secondary">{selectedFamily.label}</Badge>
                {selectedAlgorithm.modes.map((mode) => (
                  <Badge key={mode} variant="outline">
                    {mode}
                  </Badge>
                ))}
                {selectedAlgorithm.actionSpaces.map((space) => (
                  <Badge key={space} variant="outline">
                    action: {space}
                  </Badge>
                ))}
                {document.studyDraft.target.rewardSource === "verifiable" ? (
                  <Badge variant="info">RLVR</Badge>
                ) : null}
              </div>
              <p className="text-xs leading-5 text-muted-foreground">
                {selectedAlgorithm.description} Required evidence:{" "}
                {selectedFamily.evidence.join(", ")}.
              </p>
              <p
                className={`text-xs leading-5 ${
                  executionStatus.native ? "text-success-foreground" : "text-warning-foreground"
                }`}
              >
                {executionStatus.detail}
              </p>
            </div>
          </section>

          <section className="space-y-4 rounded-xl border border-border bg-card p-4">
            <div>
              <h3 className="text-sm font-semibold">Experiment budget</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Bound the number of trials, elapsed time, and random seeds.
              </p>
            </div>
            <div className="grid gap-3 @[34rem]/autoresearch:grid-cols-2">
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Maximum runs</span>
                <Input
                  max={RESEARCH_MAX_RUN_BUDGET}
                  min={1}
                  nativeInput
                  type="number"
                  value={document.studyDraft.maxRuns}
                  onChange={(event) => {
                    const value = Number(event.currentTarget.value);
                    if (!Number.isSafeInteger(value)) return;
                    updateStudy({ maxRuns: Math.min(RESEARCH_MAX_RUN_BUDGET, Math.max(1, value)) });
                  }}
                />
              </label>
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Wall-clock minutes</span>
                <Input
                  max={RESEARCH_MAX_WALL_CLOCK_MINUTES}
                  min={1}
                  nativeInput
                  type="number"
                  value={document.studyDraft.maxWallClockMinutes}
                  onChange={(event) => {
                    const value = Number(event.currentTarget.value);
                    if (!Number.isSafeInteger(value)) return;
                    updateStudy({
                      maxWallClockMinutes: Math.min(
                        RESEARCH_MAX_WALL_CLOCK_MINUTES,
                        Math.max(1, value),
                      ),
                    });
                  }}
                />
              </label>
            </div>
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">Seeds</span>
              <Input
                nativeInput
                placeholder="0, 1, 2"
                value={seedText}
                onChange={(event) => {
                  setSeedText(event.currentTarget.value);
                  setSaveState("idle");
                }}
              />
              <span className="block text-[11px] leading-4 text-muted-foreground">
                Unique integers; the count cannot exceed the run budget.
              </span>
            </label>
          </section>

          <section className="space-y-3 rounded-xl border border-border bg-card p-4">
            <div className="flex items-start gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold">Specialists</h3>
                  <Badge size="sm" variant="secondary">
                    {document.studyDraft.specialistProfileIds.length} selected
                  </Badge>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Select roles for this iteration; edit their instructions separately.
                </p>
              </div>
              <Button
                disabled={saveState === "saving"}
                size="xs"
                variant="ghost-muted"
                onClick={() => void openSpecialists()}
              >
                <Settings2Icon />
                Manage
              </Button>
            </div>
            <div className="grid gap-2 @[32rem]/autoresearch:grid-cols-2">
              {document.profiles.map((profile) => {
                const checked = document.studyDraft.specialistProfileIds.includes(profile.id);
                const disabled =
                  !checked &&
                  document.studyDraft.specialistProfileIds.length >= RESEARCH_MAX_SELECTED_PROFILES;
                return (
                  <Tooltip key={profile.id}>
                    <TooltipTrigger
                      render={
                        <label
                          className={`flex items-center gap-2 rounded-lg border border-border px-2.5 py-2 ${
                            disabled
                              ? "cursor-not-allowed opacity-50"
                              : "cursor-pointer hover:bg-accent/40"
                          }`}
                        />
                      }
                    >
                      <Checkbox
                        checked={checked}
                        disabled={disabled}
                        onCheckedChange={(value) => toggleProfile(profile.id, value === true)}
                      />
                      <span className="min-w-0 truncate text-sm font-medium">{profile.name}</span>
                    </TooltipTrigger>
                    <TooltipPopup side="top">{profile.summary}</TooltipPopup>
                  </Tooltip>
                );
              })}
            </div>
          </section>
        </div>
      </div>

      <footer className="shrink-0 border-t border-border bg-background px-3 py-2 @[38rem]/autoresearch:px-4">
        <div className="mx-auto flex max-w-3xl flex-wrap items-center gap-2">
          <span className="hidden min-w-0 flex-1 items-center gap-1.5 text-xs text-muted-foreground @[34rem]/autoresearch:flex">
            <ShieldCheckIcon className="size-3.5" />
            Review the prepared turn before sending.
          </span>
          <div className="ml-auto flex items-center gap-2">
            <Button
              disabled={saveState === "saving"}
              size="sm"
              variant="outline"
              onClick={() => void persist(false)}
            >
              {saveState === "saved" ? <CheckIcon /> : <SaveIcon />}
              {saveState === "saving" ? "Saving…" : saveState === "saved" ? "Saved" : "Save study"}
            </Button>
            <Button
              disabled={saveState === "saving" || (baselineRunId !== null && baseline.isPending)}
              size="sm"
              onClick={() => void prepareIteration()}
            >
              <SendIcon />
              Prepare iteration
            </Button>
          </div>
        </div>
      </footer>
    </div>
  );
}
