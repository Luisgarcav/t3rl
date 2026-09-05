import { expect, it } from "@effect/vitest";
import * as Context from "effect/Context";
import { Tool } from "effect/unstable/ai";

import { RlToolkit } from "./tools.ts";

const schemaHasDescription = (schema: unknown): boolean => {
  if (!schema || typeof schema !== "object") return false;
  const record = schema as Record<string, unknown>;
  if (typeof record.description === "string" && record.description.length > 0) return true;
  return [record.anyOf, record.oneOf, record.allOf]
    .filter(Array.isArray)
    .some((members) => members.some(schemaHasDescription));
};

it("exports every RL Lab utility with closed-world annotations", () => {
  expect(Object.keys(RlToolkit.tools).sort()).toEqual([
    "rl_cancel_run",
    "rl_capabilities",
    "rl_compare_runs",
    "rl_compare_study",
    "rl_create_study",
    "rl_get_run",
    "rl_get_study",
    "rl_list_artifacts",
    "rl_list_runs",
    "rl_query_metrics",
    "rl_read_artifact",
    "rl_resume_run",
    "rl_start_run",
    "rl_validate_experiment",
    "rl_warm_start_run",
  ]);

  for (const tool of Object.values(RlToolkit.tools)) {
    expect(tool.description?.length ?? 0).toBeGreaterThan(60);
    const schema = Tool.getJsonSchema(tool) as {
      readonly properties?: Readonly<Record<string, unknown>>;
    };
    expect(schema, tool.name).toMatchObject({ type: "object" });
    expect(schema, tool.name).not.toHaveProperty("oneOf");
    expect(schema, tool.name).not.toHaveProperty("anyOf");
    for (const [field, fieldSchema] of Object.entries(schema.properties ?? {})) {
      expect(schemaHasDescription(fieldSchema), `${tool.name}.${field}`).toBe(true);
    }
  }

  for (const name of [
    "rl_start_run",
    "rl_cancel_run",
    "rl_resume_run",
    "rl_warm_start_run",
    "rl_create_study",
  ] as const) {
    const tool = RlToolkit.tools[name];
    expect(Context.get(tool.annotations, Tool.Readonly)).toBe(false);
    expect(Context.get(tool.annotations, Tool.Destructive)).toBe(true);
    expect(Context.get(tool.annotations, Tool.Idempotent)).toBe(true);
    expect(Context.get(tool.annotations, Tool.OpenWorld)).toBe(false);
  }

  for (const name of [
    "rl_capabilities",
    "rl_list_runs",
    "rl_get_run",
    "rl_get_study",
    "rl_list_artifacts",
    "rl_query_metrics",
    "rl_compare_runs",
    "rl_compare_study",
    "rl_read_artifact",
    "rl_validate_experiment",
  ] as const) {
    const tool = RlToolkit.tools[name];
    expect(Context.get(tool.annotations, Tool.Readonly)).toBe(true);
    expect(Context.get(tool.annotations, Tool.Destructive)).toBe(false);
    expect(Context.get(tool.annotations, Tool.Idempotent)).toBe(true);
    expect(Context.get(tool.annotations, Tool.OpenWorld)).toBe(false);
  }
});
