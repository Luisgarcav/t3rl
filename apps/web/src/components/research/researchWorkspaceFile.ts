import {
  RESEARCH_WORKSPACE_FILE,
  type EnvironmentId,
  type ResearchWorkspaceDocument,
} from "@t3tools/contracts";
import {
  createDefaultResearchWorkspaceDocument,
  parseResearchWorkspaceDocument,
  serializeResearchWorkspaceDocument,
} from "@t3tools/client-runtime/research";
import { squashAtomCommandFailure } from "@t3tools/client-runtime/state/runtime";
import { useCallback, useMemo } from "react";

import {
  setProjectFileQueryData,
  useProjectFileQuery,
} from "~/components/files/projectFilesQueryState";
import { projectEnvironment } from "~/state/projects";
import { useAtomCommand } from "~/state/use-atom-command";

export interface ResearchWorkspaceFileState {
  readonly document: ResearchWorkspaceDocument;
  readonly isPending: boolean;
  readonly readNotice: string | null;
  readonly parseError: string | null;
  readonly save: (document: ResearchWorkspaceDocument) => Promise<string | null>;
}

export function useResearchWorkspaceFile(
  environmentId: EnvironmentId,
  cwd: string,
): ResearchWorkspaceFileState {
  const query = useProjectFileQuery(environmentId, cwd, RESEARCH_WORKSPACE_FILE);
  const refresh = query.refresh;
  const writeFile = useAtomCommand(projectEnvironment.writeFile, { reportFailure: false });
  const parsed = useMemo(
    () =>
      query.data === null
        ? { document: createDefaultResearchWorkspaceDocument(), error: null }
        : parseResearchWorkspaceDocument(query.data.contents),
    [query.data],
  );
  const save = useCallback(
    async (document: ResearchWorkspaceDocument): Promise<string | null> => {
      let contents: string;
      try {
        contents = serializeResearchWorkspaceDocument(document);
      } catch (cause) {
        return cause instanceof Error ? cause.message : "The research setup is invalid.";
      }
      const result = await writeFile({
        environmentId,
        input: { cwd, relativePath: RESEARCH_WORKSPACE_FILE, contents },
      });
      if (result._tag === "Failure") {
        const failure = squashAtomCommandFailure(result);
        return failure instanceof Error ? failure.message : "Could not save the research setup.";
      }
      setProjectFileQueryData(environmentId, cwd, RESEARCH_WORKSPACE_FILE, contents);
      refresh();
      return null;
    },
    [cwd, environmentId, refresh, writeFile],
  );

  return {
    document: parsed.document,
    isPending: query.isPending && query.data === null,
    readNotice:
      query.data === null && query.error !== null
        ? `No readable ${RESEARCH_WORKSPACE_FILE} was found. Saving creates it in this workspace.`
        : null,
    parseError: parsed.error,
    save,
  };
}
