import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vite-plus/test";

import { RlDiagnosticsPanel } from "./RlDiagnosticsPanel";
import { analyzeRlDiagnostics } from "./rlDiagnostics";

describe("RlDiagnosticsPanel", () => {
  it("renders findings as accessible, severity-labelled, explainable articles", () => {
    const report = analyzeRlDiagnostics({
      metrics: [{ step: 10, wallClockMs: 100, values: { "train/value_loss": "nan" } }],
      runState: "running",
      lastMessageAt: "2026-08-25T11:59:30.000Z",
      nowMs: Date.parse("2026-08-25T12:00:00.000Z"),
    });
    const html = renderToStaticMarkup(
      <RlDiagnosticsPanel headingId="run-debugger-title" report={report} />,
    );

    expect(html).toContain('<section aria-labelledby="run-debugger-title"');
    expect(html).toContain("<h2");
    expect(html).toContain('id="run-debugger-title"');
    expect(html).toContain('aria-live="polite"');
    expect(html).toContain("Non-finite metrics were emitted");
    expect(html).toContain("Critical signal");
    expect(html).toContain("The metric stream may be stalled");
    expect(html).toContain("What to inspect next");
    expect(html).toContain("Limits of this signal");
    expect(html).toContain("screening signals");
  });

  it("renders a cautious empty state and explains missing evidence", () => {
    const report = analyzeRlDiagnostics({
      metrics: [],
      runState: "completed",
      lastMessageAt: null,
      nowMs: 0,
    });
    const html = renderToStaticMarkup(<RlDiagnosticsPanel report={report} />);

    expect(html).toContain("No diagnostic signal fired");
    expect(html).toContain("does not establish that the policy or experiment is correct");
    expect(html).toContain("Evidence coverage");
    expect(html).toContain("7 checks need more evidence");
    expect(html).toContain("Training/evaluation gap");
  });
});
