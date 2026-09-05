import * as NodeServices from "@effect/platform-node/NodeServices";
import { expect, it } from "@effect/vitest";
import {
  EnvironmentId,
  ProjectId,
  ProviderInstanceId,
  RlRunNotFoundError,
  ThreadId,
  type RlRunSummary,
} from "@t3tools/contracts";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Option from "effect/Option";
import * as Path from "effect/Path";
import { McpSchema, McpServer } from "effect/unstable/ai";

import * as ServerConfig from "../../../config.ts";
import * as ProjectionSnapshotQuery from "../../../orchestration/Services/ProjectionSnapshotQuery.ts";
import * as RlManager from "../../../rl/Manager.ts";
import * as RunStore from "../../../rl/RunStore.ts";
import * as McpHttpServer from "../../McpHttpServer.ts";
import * as McpInvocationContext from "../../McpInvocationContext.ts";

const projectId = ProjectId.make("project-rl-agent");
const foreignProjectId = ProjectId.make("project-foreign");
const threadId = ThreadId.make("thread-rl-agent");
const invocation: McpInvocationContext.McpInvocationScope = {
  environmentId: EnvironmentId.make("environment-rl-agent"),
  threadId,
  providerSessionId: "provider-session-rl-agent",
  providerInstanceId: ProviderInstanceId.make("codex"),
  capabilities: new Set(["rl"]),
  issuedAt: 1,
};
const client = McpSchema.McpServerClient.of({
  clientId: 1,
  protocolVersion: "2025-06-18",
  initializePayload: {
    protocolVersion: "2025-06-18",
    capabilities: {},
    clientInfo: { name: "rl-agent-test", version: "1.0.0" },
  },
  getClient: Effect.die("unused"),
});

const summary = (runId: string, scopedProjectId = projectId): RlRunSummary => ({
  runId,
  projectId: scopedProjectId,
  experimentId: "cartpole-ppo",
  state: "completed",
  requestedAt: "2026-01-01T00:00:00.000Z",
  startedAt: "2026-01-01T00:00:01.000Z",
  endedAt: "2026-01-01T00:01:00.000Z",
  lastMessageAt: "2026-01-01T00:01:00.000Z",
  errorCode: null,
  errorMessage: null,
});

it.effect("scopes RL tools to the current thread project and starts runs in that project", () => {
  const starts: Array<{ readonly projectId: string; readonly requestId?: string }> = [];
  const continuations: Array<{
    readonly projectId: string;
    readonly parentRunId: string;
    readonly sourceArtifactId: string;
    readonly requestId: string;
    readonly relation: "resume" | "warm-start";
  }> = [];
  const manager = RlManager.RlManager.of({
    capabilities: () => Effect.succeed({ runners: [], experiments: [] }),
    list: ({ projectId: requestedProjectId }) =>
      Effect.succeed({ runs: [summary("run-owned", ProjectId.make(requestedProjectId))] }),
    get: ({ runId }) =>
      runId === "run-foreign"
        ? Effect.succeed({
            summary: summary(runId, foreignProjectId),
            manifest: null,
            lineage: { edges: [], truncated: false },
            artifacts: [],
            metrics: [],
          })
        : runId === "run-owned"
          ? Effect.succeed({
              summary: summary(runId),
              manifest: null,
              lineage: { edges: [], truncated: false },
              artifacts: [],
              metrics: [
                { step: 1, wallClockMs: 10, values: { "train/return": 2 } },
                { step: 2, wallClockMs: 20, values: { "train/return": 4 } },
              ],
            })
          : new RlRunNotFoundError({ runId }),
    listArtifacts: ({ runId }) =>
      Effect.succeed({
        artifacts: [
          {
            artifactId: "artifact-page-1",
            kind: "summary",
            bytes: 2,
            contentType: "application/json",
            producedAt: "2026-01-01T00:01:00.000Z",
          },
        ],
        nextCursor: runId === "run-owned" ? "artifact-page-1" : null,
      }),
    start: (input) =>
      Effect.sync(() => {
        starts.push({
          projectId: input.projectId,
          ...(input.requestId === undefined ? {} : { requestId: input.requestId }),
        });
        return { runId: "run-started" };
      }),
    resume: (input) =>
      Effect.sync(() => {
        continuations.push({ ...input, relation: "resume" });
        return { runId: "run-resumed" };
      }),
    warmStart: (input) =>
      Effect.sync(() => {
        continuations.push({ ...input, relation: "warm-start" });
        return { runId: "run-warmed" };
      }),
    cancel: () => Effect.succeed({ state: "cancelling" }),
    subscribe: () => Effect.die("unused"),
    sweepInterruptedRuns: () => Effect.die("unused"),
    createStudy: () => Effect.die("unused"),
    getStudy: () => Effect.die("unused"),
    compareStudy: () => Effect.die("unused"),
    validateExperiment: () => Effect.die("unused"),
  });
  const projection = ProjectionSnapshotQuery.ProjectionSnapshotQuery.of({
    getThreadShellById: () => Effect.succeed(Option.some({ projectId } as never)),
  } as never);
  const runStore = RunStore.RunStore.of({ findArtifact: () => Effect.succeed(null) } as never);
  const testLayer = McpHttpServer.RlToolkitRegistrationLive.pipe(
    Layer.provideMerge(McpServer.McpServer.layer),
    Layer.provide(Layer.succeed(RlManager.RlManager, manager)),
    Layer.provide(Layer.succeed(ProjectionSnapshotQuery.ProjectionSnapshotQuery, projection)),
    Layer.provide(Layer.succeed(RunStore.RunStore, runStore)),
    Layer.provide(ServerConfig.layer({ rlRunsDir: "/tmp/t3rl-agent-tests" } as never)),
    Layer.provideMerge(NodeServices.layer),
  );

  return Effect.gen(function* () {
    const server = yield* McpServer.McpServer;
    const call = (name: string, args: Record<string, unknown>) =>
      server
        .callTool({ name, arguments: args })
        .pipe(
          Effect.provideService(McpInvocationContext.McpInvocationContext, invocation),
          Effect.provideService(McpSchema.McpServerClient, client),
        );

    const denied = yield* server.callTool({ name: "rl_list_runs", arguments: {} }).pipe(
      Effect.provideService(McpInvocationContext.McpInvocationContext, {
        ...invocation,
        capabilities: new Set(["preview"] as const),
      }),
      Effect.provideService(McpSchema.McpServerClient, client),
    );
    expect(denied.isError).toBe(true);
    expect(denied.content).toEqual([
      expect.objectContaining({ text: expect.stringContaining("does not grant RL Lab access") }),
    ]);

    const inspected = yield* call("rl_get_run", { runId: "run-owned" });
    expect(inspected.isError).toBe(false);
    expect(inspected.structuredContent).toMatchObject({
      projectId,
      metricBatchCount: 2,
      availableMetricKeys: ["train/return"],
    });

    const artifacts = yield* call("rl_list_artifacts", { runId: "run-owned", limit: 1 });
    expect(artifacts.isError).toBe(false);
    expect(artifacts.structuredContent).toMatchObject({
      projectId,
      runId: "run-owned",
      page: { nextCursor: "artifact-page-1" },
    });

    const foreign = yield* call("rl_get_run", { runId: "run-foreign" });
    expect(foreign.isError).toBe(true);
    expect(foreign.content).toEqual([
      expect.objectContaining({
        type: "text",
        text: expect.stringContaining("current project"),
      }),
    ]);

    const started = yield* call("rl_start_run", {
      experimentId: "cartpole-ppo",
      seed: 7,
      requestId: "agent-retry-1",
    });
    expect(started.isError).toBe(false);
    expect(started.structuredContent).toMatchObject({ projectId, runId: "run-started" });
    expect(starts).toEqual([{ projectId, requestId: "agent-retry-1" }]);

    const resumed = yield* call("rl_resume_run", {
      parentRunId: "run-owned",
      sourceArtifactId: "checkpoint-4",
      requestId: "agent-resume-1",
    });
    expect(resumed.isError).toBe(false);
    expect(resumed.structuredContent).toMatchObject({
      projectId,
      parentRunId: "run-owned",
      runId: "run-resumed",
      relation: "resume",
    });

    const warmed = yield* call("rl_warm_start_run", {
      parentRunId: "run-owned",
      sourceArtifactId: "adapter-8",
      requestId: "agent-warm-1",
    });
    expect(warmed.isError).toBe(false);
    expect(warmed.structuredContent).toMatchObject({
      projectId,
      parentRunId: "run-owned",
      runId: "run-warmed",
      relation: "warm-start",
    });
    expect(continuations).toEqual([
      {
        projectId,
        parentRunId: "run-owned",
        sourceArtifactId: "checkpoint-4",
        requestId: "agent-resume-1",
        relation: "resume",
      },
      {
        projectId,
        parentRunId: "run-owned",
        sourceArtifactId: "adapter-8",
        requestId: "agent-warm-1",
        relation: "warm-start",
      },
    ]);
  }).pipe(Effect.provide(testLayer));
});

it.effect("reads bounded textual artifacts and rejects canonical path escapes", () => {
  const runSummary = summary("run-artifacts");
  const manager = RlManager.RlManager.of({
    capabilities: () => Effect.die("unused"),
    list: () => Effect.die("unused"),
    get: () =>
      Effect.succeed({
        summary: runSummary,
        manifest: null,
        lineage: { edges: [], truncated: false },
        artifacts: [
          {
            artifactId: "artifact-summary",
            kind: "summary",
            bytes: 11,
            contentType: "application/json",
            producedAt: "2026-01-01T00:01:00.000Z",
          },
          {
            artifactId: "artifact-escape",
            kind: "log",
            bytes: 6,
            contentType: "text/plain",
            producedAt: "2026-01-01T00:01:00.000Z",
          },
        ],
        metrics: [],
      }),
    listArtifacts: () => Effect.succeed({ artifacts: [], nextCursor: null }),
    start: () => Effect.die("unused"),
    resume: () => Effect.die("unused"),
    warmStart: () => Effect.die("unused"),
    cancel: () => Effect.die("unused"),
    subscribe: () => Effect.die("unused"),
    sweepInterruptedRuns: () => Effect.die("unused"),
    createStudy: () => Effect.die("unused"),
    getStudy: () => Effect.die("unused"),
    compareStudy: () => Effect.die("unused"),
    validateExperiment: () => Effect.die("unused"),
  });
  const projection = ProjectionSnapshotQuery.ProjectionSnapshotQuery.of({
    getThreadShellById: () => Effect.succeed(Option.some({ projectId } as never)),
  } as never);
  const runStore = RunStore.RunStore.of({
    findArtifact: ({ artifactId }: { readonly artifactId: string }) =>
      Effect.succeed(
        artifactId === "artifact-summary"
          ? {
              metadata: {
                artifactId,
                kind: "summary",
                bytes: 11,
                contentType: "application/json",
                producedAt: "2026-01-01T00:01:00.000Z",
              },
              relativePath: "summary.json",
            }
          : artifactId === "artifact-escape"
            ? {
                metadata: {
                  artifactId,
                  kind: "log",
                  bytes: 6,
                  contentType: "text/plain",
                  producedAt: "2026-01-01T00:01:00.000Z",
                },
                relativePath: "escape.log",
              }
            : null,
      ),
  } as never);
  const configLayer = ServerConfig.layerTest(process.cwd(), {
    prefix: "t3rl-agent-artifact-test-",
  }).pipe(Layer.provide(NodeServices.layer));
  const testLayer = McpHttpServer.RlToolkitRegistrationLive.pipe(
    Layer.provideMerge(McpServer.McpServer.layer),
    Layer.provide(Layer.succeed(RlManager.RlManager, manager)),
    Layer.provide(Layer.succeed(ProjectionSnapshotQuery.ProjectionSnapshotQuery, projection)),
    Layer.provide(Layer.succeed(RunStore.RunStore, runStore)),
    Layer.provideMerge(configLayer),
    Layer.provideMerge(NodeServices.layer),
  );

  return Effect.scoped(
    Effect.gen(function* () {
      const config = yield* ServerConfig.ServerConfig;
      const fs = yield* FileSystem.FileSystem;
      const path = yield* Path.Path;
      const runRoot = path.join(config.rlRunsDir, runSummary.runId);
      const outside = path.join(config.rlRunsDir, "outside-secret.txt");
      yield* fs.makeDirectory(runRoot, { recursive: true });
      yield* fs.writeFileString(path.join(runRoot, "summary.json"), '{"ok":true}');
      yield* fs.writeFileString(outside, "secret");
      yield* fs.symlink(outside, path.join(runRoot, "escape.log"));

      const server = yield* McpServer.McpServer;
      const call = (artifactId: string) =>
        server
          .callTool({
            name: "rl_read_artifact",
            arguments: { runId: runSummary.runId, artifactId },
          })
          .pipe(
            Effect.provideService(McpInvocationContext.McpInvocationContext, invocation),
            Effect.provideService(McpSchema.McpServerClient, client),
          );

      const readable = yield* call("artifact-summary");
      expect(readable.isError).toBe(false);
      expect(readable.structuredContent).toMatchObject({
        projectId,
        runId: runSummary.runId,
        content: '{"ok":true}',
      });

      const escaped = yield* call("artifact-escape");
      expect(escaped.isError).toBe(true);
      expect(escaped.content).toEqual([
        expect.objectContaining({ text: expect.stringContaining("path is unavailable") }),
      ]);
    }).pipe(Effect.provide(testLayer)),
  );
});
