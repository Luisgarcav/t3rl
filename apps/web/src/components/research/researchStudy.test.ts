import { describe, expect, it } from "vite-plus/test";

import { parseResearchSeeds } from "./researchStudy";

describe("parseResearchSeeds", () => {
  it("accepts comma and whitespace separated safe integers", () => {
    expect(parseResearchSeeds("0, 7  19", 3)).toEqual({ seeds: [0, 7, 19], error: null });
  });

  it("rejects duplicates and budgets smaller than the seed set", () => {
    expect(parseResearchSeeds("1, 1", 2).error).toBe("Seeds must be unique.");
    expect(parseResearchSeeds("1, 2, 3", 2).error).toContain("exceeds");
  });
});
