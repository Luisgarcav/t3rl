import type {
  EnvironmentId,
  RlStudyComparison as StudyComparison,
  RlStudyEstimator,
} from "@t3tools/contracts";
import { useState } from "react";

import { rlEnvironment } from "~/state/rl";
import { useAtomCommand } from "~/state/use-atom-command";

import { Button } from "../../ui/button";
import { Input } from "../../ui/input";
import { Select, SelectItem, SelectPopup, SelectTrigger, SelectValue } from "../../ui/select";
import { RlStudyComparisonResult } from "./RlStudyComparisonResult";

const STATISTICAL_UNITS = [
  { label: "Paired samples within training seeds", value: "paired-sample-within-run-seed" },
  { label: "Training seeds", value: "run-seed" },
] as const;

export function RlStudyComparison({ environmentId }: { readonly environmentId: EnvironmentId }) {
  const [request, setRequest] = useState({
    studyId: "",
    baselineLabel: "baseline",
    candidateLabel: "candidate",
    metricKey: "eval_after/loss",
    statisticalUnit: "paired-sample-within-run-seed" as RlStudyEstimator["statisticalUnit"],
    resampleCount: "10000",
    missingPairPolicy: "exclude" as RlStudyEstimator["missingPairPolicy"],
  });
  const [result, setResult] = useState<StudyComparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const compareStudy = useAtomCommand(rlEnvironment.compareStudy, { reportFailure: false });
  const resampleCount = Number(request.resampleCount);
  const validRequest =
    request.studyId.trim().length > 0 &&
    request.baselineLabel.trim().length > 0 &&
    request.candidateLabel.trim().length > 0 &&
    request.baselineLabel.trim() !== request.candidateLabel.trim() &&
    request.metricKey.trim().length > 0 &&
    Number.isInteger(resampleCount) &&
    resampleCount >= 100 &&
    resampleCount <= 100_000;

  const runComparison = async () => {
    if (pending || !validRequest) return;
    setError(null);
    setResult(null);
    setPending(true);
    const response = await compareStudy({
      environmentId,
      input: {
        studyId: request.studyId.trim(),
        baselineLabel: request.baselineLabel.trim(),
        candidateLabel: request.candidateLabel.trim(),
        metricKey: request.metricKey.trim(),
        estimator: {
          version: 2,
          statistic: "paired-mean-delta",
          statisticalUnit: request.statisticalUnit,
          confidenceLevel: 0.95,
          resamplingSeed: 17,
          resampleCount,
          missingPairPolicy: request.missingPairPolicy,
        },
      },
    }).finally(() => setPending(false));
    if (response._tag === "Success") setResult(response.value);
    else setError("The study could not be compared in this environment.");
  };

  return (
    <section className="space-y-3 rounded-xl border p-4" aria-label="Authoritative paired study">
      <div>
        <div className="text-sm font-medium">Authoritative paired study</div>
        <div className="text-xs text-muted-foreground">
          Server-computed comparison from verified evaluation evidence, with explicit seed pairs and
          exclusions.
        </div>
      </div>
      <fieldset disabled={pending} className="space-y-3">
        <div className="grid gap-2 sm:grid-cols-3">
          <Input
            aria-label="Study ID"
            placeholder="study_…"
            value={request.studyId}
            onChange={(event) => setRequest({ ...request, studyId: event.currentTarget.value })}
          />
          <Input
            aria-label="Baseline label"
            value={request.baselineLabel}
            onChange={(event) =>
              setRequest({ ...request, baselineLabel: event.currentTarget.value })
            }
          />
          <Input
            aria-label="Candidate label"
            value={request.candidateLabel}
            onChange={(event) =>
              setRequest({ ...request, candidateLabel: event.currentTarget.value })
            }
          />
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1.5 text-xs">
            <span>Evaluation metric</span>
            <Input
              value={request.metricKey}
              onChange={(event) => setRequest({ ...request, metricKey: event.currentTarget.value })}
            />
          </label>
          <label className="space-y-1.5 text-xs">
            <span>Statistical unit</span>
            <Select
              disabled={pending}
              items={STATISTICAL_UNITS}
              value={request.statisticalUnit}
              onValueChange={(value) => {
                if (value === "run-seed" || value === "paired-sample-within-run-seed")
                  setRequest({ ...request, statisticalUnit: value });
              }}
            >
              <SelectTrigger aria-label="Study statistical unit" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectPopup>
                {STATISTICAL_UNITS.map((unit) => (
                  <SelectItem key={unit.value} value={unit.value}>
                    {unit.label}
                  </SelectItem>
                ))}
              </SelectPopup>
            </Select>
          </label>
          <label className="space-y-1.5 text-xs">
            <span>Resamples</span>
            <Input
              type="number"
              min={100}
              max={100_000}
              step={100}
              value={request.resampleCount}
              onChange={(event) =>
                setRequest({ ...request, resampleCount: event.currentTarget.value })
              }
            />
          </label>
          <label className="space-y-1.5 text-xs">
            <span>Missing-pair policy</span>
            <Select
              disabled={pending}
              items={[
                { label: "Exclude missing pairs", value: "exclude" },
                { label: "Require every pair", value: "fail" },
              ]}
              value={request.missingPairPolicy}
              onValueChange={(value) => {
                if (value === "exclude" || value === "fail")
                  setRequest({ ...request, missingPairPolicy: value });
              }}
            >
              <SelectTrigger aria-label="Study missing-pair policy" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectPopup>
                <SelectItem value="exclude">Exclude missing pairs</SelectItem>
                <SelectItem value="fail">Require every pair</SelectItem>
              </SelectPopup>
            </Select>
          </label>
        </div>
        <Button disabled={!validRequest || pending} size="sm" onClick={runComparison}>
          {pending ? "Comparing…" : "Compare study"}
        </Button>
      </fieldset>
      {error !== null ? (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      ) : null}
      {result !== null ? <RlStudyComparisonResult comparison={result} /> : null}
    </section>
  );
}
