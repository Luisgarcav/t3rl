import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vite-plus/test";

import { RlMetricChart } from "./RlMetricChart";

describe("RlMetricChart", () => {
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
    expect(html).toContain("left:50%");
    expect(html).toContain("top:50%");
  });
});
