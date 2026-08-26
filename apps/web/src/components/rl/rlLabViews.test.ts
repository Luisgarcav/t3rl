import { describe, expect, it } from "vite-plus/test";

import { isRlLabView, rlLabViewRequiresRun } from "./rlLabViews";

describe("RL Lab views", () => {
  it("recognizes every routable visualization view", () => {
    expect(isRlLabView("data")).toBe(true);
    expect(isRlLabView("algorithm")).toBe(true);
    expect(isRlLabView("unknown")).toBe(false);
  });

  it("keeps run-scoped views unavailable without a selection", () => {
    expect(rlLabViewRequiresRun("overview")).toBe(false);
    expect(rlLabViewRequiresRun("compare")).toBe(false);
    expect(rlLabViewRequiresRun("data")).toBe(true);
    expect(rlLabViewRequiresRun("algorithm")).toBe(true);
  });
});
