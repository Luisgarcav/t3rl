import type { RlMetricBatch } from "@t3tools/contracts";
import { describe, expect, it } from "vite-plus/test";

import {
  analyzeRlDiagnostics,
  RL_DIAGNOSTIC_KINDS,
  type AnalyzeRlDiagnosticsInput,
  type RlDiagnosticKind,
} from "./rlDiagnostics";

const NOW = Date.parse("2026-08-25T12:00:00.000Z");

function batch(step: number, values: RlMetricBatch["values"]): RlMetricBatch {
  return { step, wallClockMs: step * 500, values };
}

function series(key: string, values: ReadonlyArray<number>): ReadonlyArray<RlMetricBatch> {
  return values.map((value, index) => batch((index + 1) * 100, { [key]: value }));
}

function input(
  metrics: ReadonlyArray<RlMetricBatch>,
  overrides: Partial<AnalyzeRlDiagnosticsInput> = {},
): AnalyzeRlDiagnosticsInput {
  return {
    metrics,
    runState: "completed",
    lastMessageAt: "2026-08-25T11:59:59.000Z",
    nowMs: NOW,
    ...overrides,
  };
}

function findingKinds(metrics: ReadonlyArray<RlMetricBatch>): ReadonlyArray<RlDiagnosticKind> {
  return analyzeRlDiagnostics(input(metrics)).findings.map((finding) => finding.kind);
}

describe("analyzeRlDiagnostics", () => {
  it("reports every explicit non-finite marker as direct critical evidence", () => {
    const report = analyzeRlDiagnostics(
      input([
        batch(10, { "train/value_loss": "nan" }),
        batch(20, { "train/policy_loss": "+inf" }),
        batch(30, { "train/return": "-inf", "train/entropy": null }),
      ]),
    );

    expect(report.highestSeverity).toBe("critical");
    expect(report.findings[0]).toMatchObject({
      kind: "non-finite-metrics",
      severity: "critical",
    });
    expect(report.findings[0]?.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ metricKey: "train/value_loss", step: 10, value: "NAN" }),
        expect.objectContaining({ metricKey: "train/policy_loss", step: 20, value: "+INF" }),
        expect.objectContaining({ metricKey: "train/return", step: 30, value: "-INF" }),
      ]),
    );
    expect(report.findings[0]?.limitations.join(" ")).toContain("symptom");
  });

  it("does not confuse null, which means not captured, with numerical failure", () => {
    const report = analyzeRlDiagnostics(input([batch(1, { "train/value_loss": null })]));
    expect(report.checks.find((check) => check.kind === "non-finite-metrics")?.status).toBe(
      "clear",
    );
    expect(findingKinds([batch(1, { "train/value_loss": null })])).not.toContain(
      "non-finite-metrics",
    );
  });

  it("detects a sustained return collapse relative to the run's own peak window", () => {
    const metrics = series("train/return", [20, 40, 70, 100, 105, 102, 104, 103, 30, 28, 25, 24]);
    const report = analyzeRlDiagnostics(input(metrics));
    const collapse = report.findings.find((item) => item.kind === "return-collapse");

    expect(collapse).toMatchObject({ severity: "warning" });
    expect(collapse?.evidence).toEqual(
      expect.arrayContaining([expect.objectContaining({ label: "Scale-adjusted drop" })]),
    );
    expect(collapse?.limitations.join(" ")).toContain("relative to this run");
  });

  it("does not call a noisy but stable return trace collapsed", () => {
    const metrics = series("train/return", [20, 22, 19, 21, 23, 20, 22, 21, 20, 19, 22, 21]);
    expect(findingKinds(metrics)).not.toContain("return-collapse");
  });

  it("detects a sustained KL breach and names the value a configurable screening reference", () => {
    const metrics = series("train/approx_kl", [0.005, 0.008, 0.01, 0.08, 0.09, 0.1]);
    const report = analyzeRlDiagnostics(input(metrics));
    const kl = report.findings.find((item) => item.kind === "excessive-kl");

    expect(kl).toMatchObject({ severity: "warning" });
    expect(kl?.evidence).toContainEqual({
      label: "Configured screening reference",
      value: "0.05",
    });
    expect(kl?.limitations.join(" ")).toContain("not a universal");
  });

  it("allows an experiment to replace or disable the KL reference", () => {
    const metrics = series("train/approx_kl", [0.06, 0.07, 0.08, 0.09, 0.1]);
    const relaxed = analyzeRlDiagnostics(
      input(metrics, { policy: { approximateKlUpperReference: 0.2 } }),
    );
    const disabled = analyzeRlDiagnostics(
      input(metrics, { policy: { approximateKlUpperReference: null } }),
    );

    expect(relaxed.findings.map((item) => item.kind)).not.toContain("excessive-kl");
    expect(disabled.checks.find((check) => check.kind === "excessive-kl")?.status).toBe(
      "insufficient",
    );
  });

  it("detects entropy that retains little of its own early baseline", () => {
    const metrics = series(
      "train/entropy",
      [0.68, 0.67, 0.65, 0.66, 0.6, 0.5, 0.4, 0.3, 0.12, 0.1, 0.08, 0.07],
    );
    const entropy = analyzeRlDiagnostics(input(metrics)).findings.find(
      (item) => item.kind === "low-entropy",
    );

    expect(entropy).toMatchObject({ severity: "warning" });
    expect(entropy?.limitations.join(" ")).toContain("can be expected");
    expect(entropy?.limitations.join(" ")).toContain("relative collapse only");
  });

  it("requires a positive early entropy baseline rather than inventing an absolute floor", () => {
    const metrics = series(
      "train/entropy",
      Array.from({ length: 12 }, () => 0),
    );
    const report = analyzeRlDiagnostics(input(metrics));

    expect(report.findings.map((item) => item.kind)).not.toContain("low-entropy");
    expect(report.checks.find((check) => check.kind === "low-entropy")).toMatchObject({
      status: "insufficient",
      reason: expect.stringContaining("no positive early entropy baseline"),
    });
  });

  it("detects sustained value-loss growth without assigning meaning to its absolute scale", () => {
    const metrics = series("train/value_loss", [1, 1.1, 0.9, 1, 2, 3, 4, 6, 10, 12, 14, 16]);
    const valueLoss = analyzeRlDiagnostics(input(metrics)).findings.find(
      (item) => item.kind === "divergent-value-loss",
    );

    expect(valueLoss).toMatchObject({ severity: "warning" });
    expect(valueLoss?.evidence).toEqual(
      expect.arrayContaining([expect.objectContaining({ label: "Growth relative to baseline" })]),
    );
    expect(valueLoss?.limitations.join(" ")).toContain("reward scale");
  });

  it("does not diagnose value-loss divergence from a single spike", () => {
    const metrics = series("train/value_loss", [1, 1, 1, 1, 1, 1, 1, 1, 100, 1, 1, 1]);
    expect(findingKinds(metrics)).not.toContain("divergent-value-loss");
  });

  it("detects stream silence only from authoritative time evidence while running", () => {
    const stale = analyzeRlDiagnostics(
      input([], {
        runState: "running",
        lastMessageAt: "2026-08-25T11:59:30.000Z",
      }),
    );
    const fresh = analyzeRlDiagnostics(
      input([], {
        runState: "running",
        lastMessageAt: "2026-08-25T11:59:50.000Z",
      }),
    );
    const terminal = analyzeRlDiagnostics(
      input([], {
        runState: "completed",
        lastMessageAt: "2026-08-25T11:00:00.000Z",
      }),
    );

    expect(stale.findings.find((item) => item.kind === "stalled-stream")).toMatchObject({
      severity: "warning",
    });
    expect(
      stale.findings.find((item) => item.kind === "stalled-stream")?.limitations.join(" "),
    ).toContain("does not prove");
    expect(fresh.findings.map((item) => item.kind)).not.toContain("stalled-stream");
    expect(terminal.checks.find((check) => check.kind === "stalled-stream")?.status).toBe(
      "insufficient",
    );
  });

  it("requires a valid server timestamp before diagnosing stream silence", () => {
    for (const lastMessageAt of [null, "not-a-time", "2026-08-25T12:01:00.000Z"] as const) {
      const report = analyzeRlDiagnostics(input([], { runState: "running", lastMessageAt }));
      expect(report.findings.map((item) => item.kind)).not.toContain("stalled-stream");
      expect(report.checks.find((check) => check.kind === "stalled-stream")?.status).toBe(
        "insufficient",
      );
    }
  });

  it("reports a train/eval gap only after enough training evidence exists", () => {
    const enough = [
      ...series("train/return", [90, 95, 100, 98, 102, 101, 99, 100, 103, 101]),
      batch(2_000, { "eval/return": 30 }),
    ];
    const tooShort = [
      ...series("train/return", [100, 100, 100]),
      batch(500, { "eval/return": 20 }),
    ];
    const report = analyzeRlDiagnostics(input(enough));
    const gap = report.findings.find((item) => item.kind === "train-eval-gap");

    expect(gap).toMatchObject({
      severity: "warning",
      title: "Evaluation return trails training return",
    });
    expect(gap?.limitations.join(" ")).toContain("single evaluation aggregate");
    expect(
      analyzeRlDiagnostics(input(tooShort)).checks.find((check) => check.kind === "train-eval-gap"),
    ).toMatchObject({ status: "insufficient" });
  });

  it("treats an unusually stronger evaluation as context rather than a warning", () => {
    const metrics = [
      ...series("train/return", [20, 21, 19, 20, 22, 20, 21, 20]),
      batch(2_000, { "eval/return": 100 }),
    ];
    expect(
      analyzeRlDiagnostics(input(metrics)).findings.find((item) => item.kind === "train-eval-gap"),
    ).toMatchObject({
      severity: "info",
      title: "Evaluation return exceeds recent training return",
    });
  });

  it("marks checks insufficient instead of manufacturing findings from sparse evidence", () => {
    const report = analyzeRlDiagnostics(input([]));

    expect(report.findings).toEqual([]);
    expect(report.highestSeverity).toBeNull();
    expect(report.checks.map((check) => check.kind)).toEqual(RL_DIAGNOSTIC_KINDS);
    expect(report.checks.every((check) => check.status === "insufficient")).toBe(true);
  });

  it("is deterministic for unsorted batches and orders findings by severity then stable kind", () => {
    const metrics = [
      ...series("train/return", [20, 40, 70, 100, 105, 102, 104, 103, 30, 28, 25, 24]),
      batch(9_999, { "train/value_loss": "nan" }),
      ...series("train/approx_kl", [0.005, 0.008, 0.01, 0.08, 0.09, 0.1]),
    ].toReversed();
    const first = analyzeRlDiagnostics(input(metrics));
    const second = analyzeRlDiagnostics(input([...metrics]));

    expect(first).toEqual(second);
    expect(first.findings.map((item) => item.kind).slice(0, 3)).toEqual([
      "non-finite-metrics",
      "return-collapse",
      "excessive-kl",
    ]);
  });
});
