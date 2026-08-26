export type ResearchSeedParseResult =
  | { readonly seeds: ReadonlyArray<number>; readonly error: null }
  | { readonly seeds: ReadonlyArray<number>; readonly error: string };

export function parseResearchSeeds(value: string, maxRuns: number): ResearchSeedParseResult {
  const tokens = value
    .split(/[\s,]+/)
    .map((token) => token.trim())
    .filter(Boolean);
  if (tokens.length === 0) return { seeds: [], error: "Enter at least one integer seed." };
  const seeds = tokens.map(Number);
  if (seeds.some((seed) => !Number.isSafeInteger(seed))) {
    return { seeds: [], error: "Seeds must be safe integers separated by commas." };
  }
  if (new Set(seeds).size !== seeds.length) {
    return { seeds: [], error: "Seeds must be unique." };
  }
  if (seeds.length > maxRuns) {
    return { seeds: [], error: `The seed list exceeds the ${maxRuns}-run budget.` };
  }
  return { seeds, error: null };
}
