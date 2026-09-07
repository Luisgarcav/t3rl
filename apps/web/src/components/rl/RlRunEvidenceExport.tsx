import { useAtomRefresh } from "@effect/atom-react";
import { squashAtomCommandFailure } from "@t3tools/client-runtime/state/runtime";
import type { EnvironmentId, ProjectId, RlEvidenceExport, RlRunId } from "@t3tools/contracts";
import { DownloadIcon } from "lucide-react";
import { useState } from "react";

import { useAssetUrlState } from "~/assets/assetUrls";
import { assetEnvironment } from "~/state/assets";
import { rlEnvironment } from "~/state/rl";
import { useAtomCommand } from "~/state/use-atom-command";

import { Button, buttonVariants } from "../ui/button";
import {
  Dialog,
  DialogDescription,
  DialogHeader,
  DialogPanel,
  DialogPopup,
  DialogTitle,
} from "../ui/dialog";
import { formatRlBytes } from "./rlPresentation";

function EvidenceDownload({
  environmentId,
  exported,
}: {
  readonly environmentId: EnvironmentId;
  readonly exported: RlEvidenceExport;
}) {
  const resource = {
    _tag: "rl-evidence",
    projectId: exported.projectId,
    exportId: exported.exportId,
  } as const;
  const url = useAssetUrlState(environmentId, resource);
  const refreshUrl = useAtomRefresh(
    assetEnvironment.createUrl({ environmentId, input: { resource } }),
  );
  return (
    <div className="space-y-4">
      <p className="text-sm">
        {formatRlBytes(exported.bytes)} · {exported.fileCount} files · {exported.omittedCount}{" "}
        omissions
      </p>
      <p className="text-xs text-muted-foreground">
        The bundle includes an offline verifier and lists omitted evidence. Environment
        reconstruction, trainer resume, and repeatability require their own recorded evidence.
      </p>
      <dl className="space-y-3 text-xs">
        <div>
          <dt className="font-medium">Archive SHA-256</dt>
          <dd className="mt-1 break-all font-mono text-muted-foreground">{exported.sha256}</dd>
        </div>
        <div>
          <dt className="font-medium">Index SHA-256</dt>
          <dd className="mt-1 break-all font-mono text-muted-foreground">{exported.indexSha256}</dd>
        </div>
      </dl>
      {url._tag === "Success" ? (
        <a
          className={buttonVariants({ size: "sm", variant: "outline" })}
          download={`${exported.exportId}.tar`}
          href={url.url}
          rel="noreferrer"
          target="_blank"
        >
          <DownloadIcon />
          Download evidence (.tar)
        </a>
      ) : url._tag === "Failure" ? (
        <div className="space-y-2">
          <p role="alert" className="text-sm text-destructive">
            The download link is unavailable.
          </p>
          <Button size="sm" variant="outline" onClick={refreshUrl}>
            Retry download link
          </Button>
        </div>
      ) : (
        <p role="status" className="text-sm text-muted-foreground">
          Preparing download link…
        </p>
      )}
    </div>
  );
}

export function RlRunEvidenceExport({
  environmentId,
  projectId,
  runId,
  enabled,
}: {
  readonly environmentId: EnvironmentId;
  readonly projectId: ProjectId;
  readonly runId: RlRunId;
  readonly enabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [exported, setExported] = useState<RlEvidenceExport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const exportEvidence = useAtomCommand(rlEnvironment.exportEvidence, { reportFailure: false });

  const createExport = async () => {
    if (!enabled) return;
    setOpen(true);
    if (pending || exported !== null) return;
    setError(null);
    setPending(true);
    const result = await exportEvidence({
      environmentId,
      input: { projectId, runIds: [runId], study: null, recordIds: [], additionalArtifacts: [] },
    }).finally(() => setPending(false));
    if (result._tag === "Success") setExported(result.value);
    else {
      const failure = squashAtomCommandFailure(result);
      setError(
        failure instanceof Error && failure.message.length > 0
          ? failure.message
          : "The evidence export could not be created.",
      );
    }
  };

  return (
    <>
      <Button disabled={!enabled || pending} size="sm" variant="outline" onClick={createExport}>
        <DownloadIcon />
        {pending ? "Exporting…" : exported === null ? "Export evidence" : "Evidence export"}
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogPopup>
          <DialogHeader>
            <DialogTitle>Run evidence export</DialogTitle>
            <DialogDescription className="break-all">{runId}</DialogDescription>
          </DialogHeader>
          <DialogPanel>
            {pending ? (
              <p role="status" className="text-sm text-muted-foreground">
                Preparing evidence bundle…
              </p>
            ) : null}
            {error !== null ? (
              <div className="space-y-3">
                <p role="alert" className="text-sm text-destructive">
                  {error}
                </p>
                <Button size="sm" variant="outline" onClick={createExport}>
                  Try again
                </Button>
              </div>
            ) : null}
            {exported !== null ? (
              <EvidenceDownload environmentId={environmentId} exported={exported} />
            ) : null}
          </DialogPanel>
        </DialogPopup>
      </Dialog>
    </>
  );
}
