import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vite-plus/test";

import { RlMetricChart } from "./RlMetricChart";

describe("RlMetricChart", () => {
  it("renders an explicit empty state instead of fabricating a zero", () => {
    const html = renderToStaticMarkup(
      <RlMetricChart
        definition={{
          key: "train/reward",
          label: "Training reward",
          description: "Mean reward",
          color: "var(--color-info)",
        }}
        metrics={[]}
      />,
    );

    expect(html).toContain("Waiting for metric data");
    expect(html).toContain(">—</span>");
    expect(html).not.toContain('data-slot="rl-metric-chart-line"');
    expect(html).not.toContain('data-slot="rl-metric-chart-latest-point"');
  });

  it("centers a visible marker when the run has only one metric point", () => {
    const html = renderToStaticMarkup(
      <RlMetricChart
        definition={{
          key: "train/reward",
          label: "Training reward",
          description: "Mean reward",
          color: "var(--color-info)",
        }}
        metrics={[{ step: 2, wallClockMs: 100, values: { "train/reward": 1 } }]}
      />,
    );

    expect(html).toContain('data-slot="rl-metric-chart-latest-point"');
    expect(html).not.toContain('data-slot="rl-metric-chart-line"');
    expect(html).toContain("left:50%");
    expect(html).toContain("top:50%");
  });

  it("renders a line and latest-point marker for multiple observations", () => {
    const html = renderToStaticMarkup(
      <RlMetricChart
        definition={{
          key: "train/reward",
          label: "Training reward",
          description: "Mean reward",
          color: "var(--color-info)",
        }}
        metrics={[
          { step: 2, wallClockMs: 100, values: { "train/reward": 1 } },
          { step: 3, wallClockMs: 200, values: { "train/reward": 2 } },
        ]}
      />,
    );

    expect(html).toContain('data-slot="rl-metric-chart-line"');
    expect(html).toContain('d="M8.00,96.00 L312.00,8.00"');
    expect(html).toContain('data-slot="rl-metric-chart-latest-point"');
  });
});
