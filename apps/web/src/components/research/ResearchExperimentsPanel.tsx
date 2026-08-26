import type { EnvironmentId, ProjectId, RlRunId } from "@t3tools/contracts";
import { useState } from "react";

import { RlLabPage, type RlLabView } from "~/components/rl/RlLabPage";
import { stackedThreadToast, toastManager } from "~/components/ui/toast";

import { useResearchWorkspaceFile } from "./researchWorkspaceFile";

export function ResearchExperimentsPanel(props: {
  readonly environmentId: EnvironmentId;
  readonly projectId: ProjectId;
  readonly cwd: string;
  readonly initialSelectedRunId?: RlRunId | null;
  readonly onOpenAutoresearch: () => void;
}) {
  const workspace = useResearchWorkspaceFile(props.environmentId, props.cwd);
  const [selectedRunId, setSelectedRunId] = useState<RlRunId | null>(
    props.initialSelectedRunId ?? null,
  );
  const [activeView, setActiveView] = useState<RlLabView>("overview");
  const [savingBaselineRunId, setSavingBaselineRunId] = useState<RlRunId | null>(null);

  const useAsBaseline = async (runId: RlRunId) => {
    if (workspace.isPending || savingBaselineRunId !== null) return;
    setSavingBaselineRunId(runId);
    const failure = await workspace.save({
      ...workspace.document,
      studyDraft: { ...workspace.document.studyDraft, baselineRunId: runId },
    });
    setSavingBaselineRunId(null);
    if (failure !== null) {
      toastManager.add(
        stackedThreadToast({
          type: "error",
          title: "Could not set the baseline",
          description: failure,
        }),
      );
      return;
    }
    toastManager.add(
      stackedThreadToast({
        type: "success",
        title: "Baseline selected",
        description: "The run is ready to use in the next Autoresearch iteration.",
      }),
    );
    props.onOpenAutoresearch();
  };

  return (
    <div className="flex min-h-0 flex-1 overflow-hidden">
      <RlLabPage
        environmentId={props.environmentId}
        projectId={props.projectId}
        selectedRunId={selectedRunId}
        activeView={activeView}
        baselineActionPending={workspace.isPending || savingBaselineRunId !== null}
        baselineRunId={workspace.document.studyDraft.baselineRunId}
        chromeVariant="embedded"
        onSelectRun={(runId) => {
          setSelectedRunId(runId);
          if (runId === null && (activeView === "behavior" || activeView === "diagnostics")) {
            setActiveView("overview");
          }
        }}
        onUseRunAsBaseline={(runId) => void useAsBaseline(runId)}
        onViewChange={setActiveView}
      />
    </div>
  );
}
