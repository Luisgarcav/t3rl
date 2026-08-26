import { EnvironmentId, ProjectId, type RlRunId } from "@t3tools/contracts";
import { createFileRoute } from "@tanstack/react-router";

import { RlLabPage, type RlLabView } from "~/components/rl/RlLabPage";
import { SidebarInset } from "~/components/ui/sidebar";

export interface RlLabSearch {
  readonly runId?: RlRunId;
  readonly view?: RlLabView;
}

const RUN_ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;
const RL_LAB_VIEWS = new Set<RlLabView>(["overview", "behavior", "compare", "diagnostics"]);

export const Route = createFileRoute("/_chat/rl/$environmentId/$projectId")({
  validateSearch: (raw: Record<string, unknown>): RlLabSearch => {
    const runId =
      typeof raw.runId === "string" && RUN_ID_PATTERN.test(raw.runId)
        ? (raw.runId as RlRunId)
        : undefined;
    const view =
      typeof raw.view === "string" && RL_LAB_VIEWS.has(raw.view as RlLabView)
        ? (raw.view as RlLabView)
        : undefined;
    return {
      ...(runId === undefined ? {} : { runId }),
      ...(view === undefined || (runId === undefined && view !== "compare") ? {} : { view }),
    };
  },
  component: RlLabRoute,
});

function RlLabRoute() {
  const params = Route.useParams();
  const search = Route.useSearch();
  const navigate = Route.useNavigate();
  const environmentId = EnvironmentId.make(params.environmentId);
  const projectId = ProjectId.make(params.projectId);

  return (
    <SidebarInset className="h-dvh min-h-0 overflow-hidden overscroll-y-none bg-background text-foreground">
      <RlLabPage
        environmentId={environmentId}
        projectId={projectId}
        selectedRunId={search.runId ?? null}
        activeView={search.view ?? "overview"}
        onSelectRun={(runId) => {
          void navigate({ search: runId === null ? {} : { runId } });
        }}
        onViewChange={(view) => {
          void navigate({
            search: {
              ...(search.runId === undefined ? {} : { runId: search.runId }),
              ...(view === "overview" ? {} : { view }),
            },
          });
        }}
      />
    </SidebarInset>
  );
}
