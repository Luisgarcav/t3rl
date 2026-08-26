import {
  ActivityIcon,
  AlertTriangleIcon,
  CheckCircle2Icon,
  CircleHelpIcon,
  InfoIcon,
  ShieldAlertIcon,
} from "lucide-react";

import { cn } from "~/lib/utils";

import { Badge } from "../../ui/badge";
import { Card, CardDescription, CardHeader, CardPanel } from "../../ui/card";
import type {
  RlDiagnosticFinding,
  RlDiagnosticReport,
  RlDiagnosticSeverity,
} from "./rlDiagnostics";

const SEVERITY_LABEL: Readonly<Record<RlDiagnosticSeverity, string>> = {
  critical: "Critical signal",
  warning: "Warning signal",
  info: "Context signal",
};

const SEVERITY_BADGE: Readonly<Record<RlDiagnosticSeverity, "error" | "warning" | "info">> = {
  critical: "error",
  warning: "warning",
  info: "info",
};

function FindingIcon({ severity }: { readonly severity: RlDiagnosticSeverity }) {
  switch (severity) {
    case "critical":
      return <ShieldAlertIcon aria-hidden="true" className="size-4 text-destructive" />;
    case "warning":
      return <AlertTriangleIcon aria-hidden="true" className="size-4 text-warning" />;
    case "info":
      return <InfoIcon aria-hidden="true" className="size-4 text-info" />;
  }
}

function DiagnosticFinding({ finding }: { readonly finding: RlDiagnosticFinding }) {
  const titleId = `rl-diagnostic-${finding.id}-title`;
  return (
    <li>
      <article
        aria-labelledby={titleId}
        className="rounded-xl border border-border/70 bg-muted/12 p-4"
      >
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-background">
            <FindingIcon severity={finding.severity} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="font-medium text-sm" id={titleId}>
                {finding.title}
              </h3>
              <Badge size="sm" variant={SEVERITY_BADGE[finding.severity]}>
                {SEVERITY_LABEL[finding.severity]}
              </Badge>
            </div>
            <p className="mt-1 text-muted-foreground text-sm leading-relaxed">
              {finding.explanation}
            </p>
          </div>
        </div>

        <dl className="mt-4 grid gap-2 sm:grid-cols-3">
          {finding.evidence.map((item) => (
            <div
              className="min-w-0 rounded-lg border border-border/60 bg-background/60 px-3 py-2"
              key={`${item.label}-${item.step ?? "run"}-${item.value}`}
            >
              <dt className="truncate text-[11px] text-muted-foreground">{item.label}</dt>
              <dd className="mt-0.5 font-mono text-xs tabular-nums">
                {item.value}
                {item.step === undefined ? null : (
                  <span className="ml-1 text-muted-foreground">at step {item.step}</span>
                )}
              </dd>
            </div>
          ))}
        </dl>

        <div className="mt-4 grid gap-4 text-xs md:grid-cols-2">
          <div>
            <h4 className="font-medium">What to inspect next</h4>
            <ul className="mt-1.5 list-disc space-y-1 pl-4 text-muted-foreground">
              {finding.suggestedChecks.map((suggestion) => (
                <li key={suggestion}>{suggestion}</li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="font-medium">Limits of this signal</h4>
            <ul className="mt-1.5 list-disc space-y-1 pl-4 text-muted-foreground">
              {finding.limitations.map((limitation) => (
                <li key={limitation}>{limitation}</li>
              ))}
            </ul>
          </div>
        </div>
      </article>
    </li>
  );
}

export interface RlDiagnosticsPanelProps {
  readonly report: RlDiagnosticReport;
  readonly className?: string;
  readonly headingId?: string;
}

export function RlDiagnosticsPanel({
  report,
  className,
  headingId = "rl-diagnostics-title",
}: RlDiagnosticsPanelProps) {
  const clearCount = report.checks.filter((check) => check.status === "clear").length;
  const insufficient = report.checks.filter((check) => check.status === "insufficient");

  return (
    <section aria-labelledby={headingId} className={className}>
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <ActivityIcon aria-hidden="true" className="size-4 text-muted-foreground" />
            <h2 className="font-semibold text-base leading-none" id={headingId}>
              Automatic diagnostics
            </h2>
          </div>
          <CardDescription>
            Explainable screening signals from retained metrics. They guide inspection; they are not
            causal conclusions or universal RL thresholds.
          </CardDescription>
        </CardHeader>
        <CardPanel aria-live="polite" aria-atomic="false">
          <div className="mb-4 flex flex-wrap items-center gap-2 text-xs">
            <Badge variant={report.findings.length === 0 ? "success" : "secondary"}>
              {report.findings.length} signal{report.findings.length === 1 ? "" : "s"}
            </Badge>
            <Badge variant="outline">{clearCount} checks clear</Badge>
            <Badge variant="outline">{report.analyzedBatchCount} batches analyzed</Badge>
          </div>

          {report.findings.length === 0 ? (
            <div className="flex items-start gap-3 rounded-xl border border-success/24 bg-success/4 p-4">
              <CheckCircle2Icon
                aria-hidden="true"
                className="mt-0.5 size-4 shrink-0 text-success"
              />
              <div>
                <h3 className="font-medium text-sm">No diagnostic signal fired</h3>
                <p className="mt-1 text-muted-foreground text-sm">
                  The available evidence did not cross the configured screening references. This
                  does not establish that the policy or experiment is correct.
                </p>
              </div>
            </div>
          ) : (
            <ol className="space-y-3">
              {report.findings.map((item) => (
                <DiagnosticFinding finding={item} key={item.id} />
              ))}
            </ol>
          )}

          <details className="mt-4 rounded-xl border border-border/60 px-3 py-2.5 text-xs">
            <summary className="flex cursor-pointer list-none items-center gap-2 font-medium marker:hidden">
              <CircleHelpIcon aria-hidden="true" className="size-3.5 text-muted-foreground" />
              Evidence coverage
              <span className="ml-auto text-muted-foreground">
                {insufficient.length} check{insufficient.length === 1 ? "" : "s"} need more evidence
              </span>
            </summary>
            <ul
              className={cn("mt-3 space-y-2", insufficient.length === 0 && "text-muted-foreground")}
            >
              {insufficient.length === 0 ? (
                <li>Every automatic check had enough evidence to evaluate.</li>
              ) : (
                insufficient.map((check) => (
                  <li className="grid gap-0.5 sm:grid-cols-[12rem_1fr]" key={check.kind}>
                    <span className="font-medium">{check.label}</span>
                    <span className="text-muted-foreground">{check.reason}</span>
                  </li>
                ))
              )}
            </ul>
          </details>
        </CardPanel>
      </Card>
    </section>
  );
}
