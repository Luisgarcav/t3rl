export const RL_LAB_VIEW_IDS = [
  "overview",
  "data",
  "algorithm",
  "behavior",
  "compare",
  "diagnostics",
] as const;

export type RlLabView = (typeof RL_LAB_VIEW_IDS)[number];

export function isRlLabView(value: unknown): value is RlLabView {
  return typeof value === "string" && (RL_LAB_VIEW_IDS as ReadonlyArray<string>).includes(value);
}

export function rlLabViewRequiresRun(view: RlLabView): boolean {
  return view !== "overview" && view !== "compare";
}
