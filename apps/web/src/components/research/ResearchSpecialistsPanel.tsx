import {
  RESEARCH_MAX_PROFILES,
  RESEARCH_WORKSPACE_FILE,
  ResearchAgentProfileId,
  type EnvironmentId,
  type ResearchAgentProfile,
  type ResearchWorkspaceDocument,
} from "@t3tools/contracts";
import {
  buildResearchSpecialistPrompt,
  createDefaultResearchWorkspaceDocument,
} from "@t3tools/client-runtime/research";
import {
  AlertCircleIcon,
  CheckIcon,
  FileCode2Icon,
  MicroscopeIcon,
  PlusIcon,
  RotateCcwIcon,
  SaveIcon,
  SendIcon,
  Trash2Icon,
  WorkflowIcon,
} from "lucide-react";
import { useState } from "react";

import { randomUUID } from "~/lib/utils";
import { Alert, AlertDescription, AlertTitle } from "~/components/ui/alert";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Skeleton } from "~/components/ui/skeleton";
import { Textarea } from "~/components/ui/textarea";

import { useResearchWorkspaceFile } from "./researchWorkspaceFile";

export function ResearchSpecialistsPanel(props: {
  readonly environmentId: EnvironmentId;
  readonly cwd: string;
  readonly onPreparePrompt: (prompt: string) => void;
  readonly onOpenAutoresearch: () => void;
}) {
  const workspace = useResearchWorkspaceFile(props.environmentId, props.cwd);
  if (workspace.isPending) {
    return (
      <div className="space-y-3 p-4">
        <Skeleton className="h-16 rounded-xl" />
        <Skeleton className="h-72 rounded-xl" />
      </div>
    );
  }
  return (
    <ResearchSpecialistsEditor
      initialDocument={workspace.document}
      parseError={workspace.parseError}
      readNotice={workspace.readNotice}
      onPreparePrompt={props.onPreparePrompt}
      onOpenAutoresearch={props.onOpenAutoresearch}
      onSave={workspace.save}
    />
  );
}

function ResearchSpecialistsEditor(props: {
  readonly initialDocument: ResearchWorkspaceDocument;
  readonly readNotice: string | null;
  readonly parseError: string | null;
  readonly onPreparePrompt: (prompt: string) => void;
  readonly onOpenAutoresearch: () => void;
  readonly onSave: (document: ResearchWorkspaceDocument) => Promise<string | null>;
}) {
  const [document, setDocument] = useState(props.initialDocument);
  const [selectedProfileId, setSelectedProfileId] = useState<ResearchAgentProfileId | null>(
    document.profiles[0]?.id ?? null,
  );
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");
  const [error, setError] = useState<string | null>(props.parseError);
  const selectedProfile =
    document.profiles.find((profile) => profile.id === selectedProfileId) ??
    document.profiles[0] ??
    null;

  const replaceProfile = (next: ResearchAgentProfile) => {
    setSaveState("idle");
    setDocument((current) => ({
      ...current,
      profiles: current.profiles.map((profile) => (profile.id === next.id ? next : profile)),
    }));
  };

  const save = async () => {
    setSaveState("saving");
    setError(null);
    const failure = await props.onSave(document);
    if (failure !== null) {
      setSaveState("idle");
      setError(failure);
      return false;
    }
    setSaveState("saved");
    return true;
  };

  const openAutoresearch = async () => {
    if (await save()) props.onOpenAutoresearch();
  };

  const addProfile = () => {
    if (document.profiles.length >= RESEARCH_MAX_PROFILES) return;
    const id = ResearchAgentProfileId.make(`specialist-${randomUUID()}`);
    setDocument((current) => ({
      ...current,
      profiles: [
        ...current.profiles,
        {
          id,
          name: "New specialist",
          summary: "Describe the scientific responsibility of this role.",
          instructions:
            "State the evidence this specialist should inspect and the claims it may make.",
        },
      ],
    }));
    setSelectedProfileId(id);
    setSaveState("idle");
  };

  const deleteProfile = () => {
    if (selectedProfile === null) return;
    const profiles = document.profiles.filter((profile) => profile.id !== selectedProfile.id);
    setDocument({
      ...document,
      profiles,
      studyDraft: {
        ...document.studyDraft,
        specialistProfileIds: document.studyDraft.specialistProfileIds.filter(
          (profileId) => profileId !== selectedProfile.id,
        ),
      },
    });
    setSelectedProfileId(profiles[0]?.id ?? null);
    setSaveState("idle");
  };

  const resetDefaults = () => {
    const defaults = createDefaultResearchWorkspaceDocument();
    setDocument((current) => ({
      ...current,
      profiles: defaults.profiles,
      studyDraft: {
        ...current.studyDraft,
        specialistProfileIds: defaults.studyDraft.specialistProfileIds,
      },
    }));
    setSelectedProfileId(defaults.profiles[0]?.id ?? null);
    setSaveState("idle");
    setError(null);
  };

  return (
    <div className="@container/specialists flex min-h-0 flex-1 flex-col bg-background">
      <header className="flex shrink-0 items-center gap-3 border-b border-border px-4 py-3">
        <MicroscopeIcon className="size-4 text-muted-foreground" />
        <div className="min-w-0">
          <h2 className="text-sm font-semibold">Research specialists</h2>
          <p className="truncate text-xs text-muted-foreground">
            Portable project instructions, using the current thread’s provider and model.
          </p>
        </div>
        <Badge
          aria-label={`Stored in ${RESEARCH_WORKSPACE_FILE}`}
          className="ml-auto text-[10px]"
          title={RESEARCH_WORKSPACE_FILE}
          variant="secondary"
        >
          <FileCode2Icon />
          <span className="hidden @[30rem]/specialists:inline">Project config</span>
        </Badge>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-3 @[38rem]/specialists:p-4">
        <div className="mx-auto max-w-3xl space-y-4">
          {props.readNotice !== null ? (
            <Alert variant="info">
              <FileCode2Icon />
              <AlertDescription>{props.readNotice}</AlertDescription>
            </Alert>
          ) : null}
          {error !== null ? (
            <Alert variant="error">
              <AlertCircleIcon />
              <AlertTitle>Research setup not saved</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          <div className="grid gap-4 @[42rem]/specialists:grid-cols-[13rem_minmax(0,1fr)]">
            <section className="rounded-xl border border-border bg-muted/8 p-2">
              <div className="mb-2 flex items-center gap-2 px-1">
                <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Profiles
                </span>
                <Button
                  aria-label="Add research specialist"
                  className="ml-auto"
                  disabled={document.profiles.length >= RESEARCH_MAX_PROFILES}
                  size="icon-xs"
                  variant="ghost-muted"
                  onClick={addProfile}
                >
                  <PlusIcon />
                </Button>
              </div>
              <div className="space-y-1">
                {document.profiles.map((profile) => (
                  <button
                    aria-current={profile.id === selectedProfile?.id ? "true" : undefined}
                    className={`w-full rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-accent ${
                      profile.id === selectedProfile?.id ? "bg-accent" : ""
                    }`}
                    key={profile.id}
                    type="button"
                    onClick={() => setSelectedProfileId(profile.id)}
                  >
                    <span className="block truncate text-sm font-medium">{profile.name}</span>
                    <span className="mt-0.5 line-clamp-2 text-[11px] leading-4 text-muted-foreground">
                      {profile.summary}
                    </span>
                  </button>
                ))}
                {document.profiles.length === 0 ? (
                  <p className="p-3 text-center text-xs text-muted-foreground">
                    Add a specialist to begin.
                  </p>
                ) : null}
              </div>
            </section>

            <section className="rounded-xl border border-border bg-card p-4">
              {selectedProfile === null ? (
                <div className="flex min-h-56 items-center justify-center text-sm text-muted-foreground">
                  Select or add a specialist.
                </div>
              ) : (
                <div className="space-y-4">
                  <label className="block space-y-1.5">
                    <span className="text-sm font-medium">Name</span>
                    <Input
                      maxLength={80}
                      nativeInput
                      value={selectedProfile.name}
                      onChange={(event) =>
                        replaceProfile({ ...selectedProfile, name: event.currentTarget.value })
                      }
                    />
                  </label>
                  <label className="block space-y-1.5">
                    <span className="text-sm font-medium">Responsibility</span>
                    <Input
                      maxLength={240}
                      nativeInput
                      value={selectedProfile.summary}
                      onChange={(event) =>
                        replaceProfile({ ...selectedProfile, summary: event.currentTarget.value })
                      }
                    />
                  </label>
                  <label className="block space-y-1.5">
                    <span className="text-sm font-medium">Instructions</span>
                    <Textarea
                      className="font-mono text-xs"
                      maxLength={12_000}
                      rows={12}
                      value={selectedProfile.instructions}
                      onChange={(event) =>
                        replaceProfile({
                          ...selectedProfile,
                          instructions: event.currentTarget.value,
                        })
                      }
                    />
                    <span className="block text-[11px] leading-4 text-muted-foreground">
                      T3 inserts these as visible project-authored instructions. They are not a
                      hidden provider-specific system prompt.
                    </span>
                  </label>
                  <div className="flex flex-wrap justify-between gap-2 border-t border-border pt-4">
                    <Button size="sm" variant="destructive-outline" onClick={deleteProfile}>
                      <Trash2Icon />
                      Delete
                    </Button>
                    <Button
                      disabled={
                        selectedProfile.name.trim().length === 0 ||
                        selectedProfile.instructions.trim().length === 0
                      }
                      size="sm"
                      onClick={() =>
                        props.onPreparePrompt(buildResearchSpecialistPrompt(selectedProfile))
                      }
                    >
                      <SendIcon />
                      Use in composer
                    </Button>
                  </div>
                </div>
              )}
            </section>
          </div>
        </div>
      </div>

      <footer className="shrink-0 border-t border-border bg-background px-3 py-2 @[38rem]/specialists:px-4">
        <div className="mx-auto flex max-w-3xl flex-wrap items-center gap-2">
          <Button
            disabled={saveState === "saving"}
            size="sm"
            variant="ghost-muted"
            onClick={() => void openAutoresearch()}
          >
            <WorkflowIcon />
            Autoresearch
          </Button>
          <div className="ml-auto flex items-center gap-2">
            <Button size="sm" variant="ghost-muted" onClick={resetDefaults}>
              <RotateCcwIcon />
              <span className="hidden @[32rem]/specialists:inline">Restore defaults</span>
            </Button>
            <Button disabled={saveState === "saving"} size="sm" onClick={() => void save()}>
              {saveState === "saved" ? <CheckIcon /> : <SaveIcon />}
              {saveState === "saving"
                ? "Saving…"
                : saveState === "saved"
                  ? "Saved"
                  : "Save profiles"}
            </Button>
          </div>
        </div>
      </footer>
    </div>
  );
}
