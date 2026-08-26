import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Option from "effect/Option";
import * as Path from "effect/Path";
import * as Schema from "effect/Schema";

import * as ServerConfig from "../../../config.ts";
import * as ProjectionSnapshotQuery from "../../../orchestration/Services/ProjectionSnapshotQuery.ts";
import * as Artifacts from "../../../rl/Artifacts.ts";
import * as RlManager from "../../../rl/Manager.ts";
import * as RunStore from "../../../rl/RunStore.ts";
import * as McpInvocationContext from "../../McpInvocationContext.ts";
import { queryMetrics, summarizeMetrics } from "./analysis.ts";
import { RL_AGENT_ARTIFACT_MAX_BYTES, RlAgentToolError, RlToolkit } from "./tools.ts";

const fail = (code: RlAgentToolError["code"], detail: string) =>
  Effect.fail(new RlAgentToolError({ code, detail: detail.slice(0, 2048) }));
const isRlAgentToolError = Schema.is(RlAgentToolError);

const projectContext = Effect.fn("RlToolkit.projectContext")(function* () {
  const invocation = yield* McpInvocationContext.McpInvocationContext;
  if (!invocation.capabilities.has("rl")) {
    return yield* fail(
      "capability-unavailable",
      "The provider-scoped MCP credential does not grant RL Lab access.",
    );
  }
  const snapshotQuery = yield* ProjectionSnapshotQuery.ProjectionSnapshotQuery;
  const thread = yield* snapshotQuery.getThreadShellById(invocation.threadId).pipe(
    Effect.mapError(
      () =>
        new RlAgentToolError({
          code: "thread-unavailable",
          detail: "The current thread's project context could not be resolved.",
        }),
    ),
  );
  if (Option.isNone(thread)) {
    return yield* fail(
      "thread-unavailable",
      "The current thread no longer exists or is not available to this credential.",
    );
  }
  return {
    projectId: thread.value.projectId,
    threadId: invocation.threadId,
  };
});

const scopedRun = Effect.fn("RlToolkit.scopedRun")(function* (runId: string) {
  const context = yield* projectContext();
  const manager = yield* RlManager.RlManager;
  const detail = yield* manager.get({ runId }).pipe(
    Effect.mapError(
      () =>
        new RlAgentToolError({
          code: "run-not-found",
          detail: `RL run not found in the current project: ${runId}`,
        }),
    ),
  );
  if (detail.summary.projectId !== context.projectId) {
    return yield* fail("run-not-found", `RL run not found in the current project: ${runId}`);
  }
  return { context, detail };
});

const inspectRun = Effect.fn("RlToolkit.inspectRun")(function* (runId: string) {
  const { context, detail } = yield* scopedRun(runId);
  return {
    ...context,
    summary: detail.summary,
    manifest: detail.manifest,
    artifacts: detail.artifacts,
    metricBatchCount: detail.metrics.length,
    availableMetricKeys: [
      ...new Set(detail.metrics.flatMap((batch) => Object.keys(batch.values))),
    ].sort((left, right) => left.localeCompare(right)),
    metricSummaries: summarizeMetrics(detail.metrics),
  };
});

const textualArtifact = (kind: string, contentType: string): boolean =>
  kind !== "model" &&
  (contentType.startsWith("text/") ||
    contentType === "application/json" ||
    contentType.endsWith("+json") ||
    contentType === "application/yaml" ||
    contentType === "application/x-yaml" ||
    contentType === "application/xml" ||
    contentType.endsWith("+xml"));

const readArtifact = Effect.fn("RlToolkit.readArtifact")(function* (input: {
  readonly runId: string;
  readonly artifactId: string;
  readonly maxBytes: number;
}) {
  const { context } = yield* scopedRun(input.runId);
  const store = yield* RunStore.RunStore;
  const artifact = yield* store.findArtifact(input).pipe(
    Effect.mapError(
      () =>
        new RlAgentToolError({
          code: "operation-failed",
          detail: "RL artifact metadata could not be read.",
        }),
    ),
  );
  if (artifact === null) {
    return yield* fail(
      "artifact-not-found",
      `RL artifact not found in run ${input.runId}: ${input.artifactId}`,
    );
  }
  if (!textualArtifact(artifact.metadata.kind, artifact.metadata.contentType)) {
    return yield* fail(
      "artifact-unsupported",
      `Artifact ${input.artifactId} is not a supported textual artifact (${artifact.metadata.contentType}).`,
    );
  }
  if (artifact.metadata.bytes > input.maxBytes) {
    return yield* fail(
      "artifact-too-large",
      `Artifact ${input.artifactId} is ${artifact.metadata.bytes} bytes; the requested limit is ${input.maxBytes}.`,
    );
  }

  const config = yield* ServerConfig.ServerConfig;
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const runRoot = Artifacts.runDirectory({ rlRunsDir: config.rlRunsDir, runId: input.runId });
  const artifactPath = Artifacts.resolveArtifactPath({
    rlRunsDir: config.rlRunsDir,
    runId: input.runId,
    relativePath: artifact.relativePath,
  });
  if (runRoot === null || artifactPath === null) {
    return yield* fail(
      "artifact-not-found",
      `RL artifact path is unavailable: ${input.artifactId}`,
    );
  }

  const canonical = yield* Effect.all([fs.realPath(runRoot), fs.realPath(artifactPath)]).pipe(
    Effect.mapError(
      () =>
        new RlAgentToolError({
          code: "artifact-not-found",
          detail: `RL artifact file is unavailable: ${input.artifactId}`,
        }),
    ),
  );
  const [canonicalRoot, canonicalFile] = canonical;
  const relative = path.relative(canonicalRoot, canonicalFile);
  if (relative === "" || relative.startsWith("..") || path.isAbsolute(relative)) {
    return yield* fail(
      "artifact-not-found",
      `RL artifact path is unavailable: ${input.artifactId}`,
    );
  }

  const bytes = yield* Effect.scoped(
    Effect.gen(function* () {
      const file = yield* fs.open(canonicalFile, { flag: "r" });
      const info = yield* file.stat;
      if (info.type !== "File") {
        return yield* fail(
          "artifact-not-found",
          `RL artifact file is unavailable: ${input.artifactId}`,
        );
      }
      if (Number(info.size) > input.maxBytes) {
        return yield* fail(
          "artifact-too-large",
          `Artifact ${input.artifactId} exceeds the ${input.maxBytes} byte limit.`,
        );
      }
      return Option.getOrElse(yield* file.readAlloc(input.maxBytes + 1), () => new Uint8Array());
    }),
  ).pipe(
    Effect.mapError((error) =>
      isRlAgentToolError(error)
        ? error
        : new RlAgentToolError({
            code: "artifact-not-found",
            detail: `RL artifact file is unavailable: ${input.artifactId}`,
          }),
    ),
  );
  if (bytes.byteLength > input.maxBytes || bytes.byteLength > RL_AGENT_ARTIFACT_MAX_BYTES) {
    return yield* fail(
      "artifact-too-large",
      `Artifact ${input.artifactId} grew beyond the ${input.maxBytes} byte limit while being read.`,
    );
  }
  const content = yield* Effect.try({
    try: () => new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    catch: () =>
      new RlAgentToolError({
        code: "artifact-unsupported",
        detail: `Artifact ${input.artifactId} is not valid UTF-8 text.`,
      }),
  });
  return { ...context, runId: input.runId, artifact: artifact.metadata, content };
});

const handlers = {
  rl_capabilities: () =>
    Effect.gen(function* () {
      const context = yield* projectContext();
      const manager = yield* RlManager.RlManager;
      const report = yield* manager.capabilities();
      return { ...context, report };
    }),
  rl_list_runs: ({ limit }) =>
    Effect.gen(function* () {
      const context = yield* projectContext();
      const manager = yield* RlManager.RlManager;
      const result = yield* manager.list({ projectId: context.projectId, limit: limit ?? 50 });
      return { ...context, runs: result.runs };
    }),
  rl_get_run: ({ runId }) => inspectRun(runId),
  rl_query_metrics: ({ runId, metricKeys, stepFrom, stepTo, limit }) =>
    Effect.gen(function* () {
      if (stepFrom !== undefined && stepTo !== undefined && stepFrom > stepTo) {
        return yield* fail("invalid-range", "stepFrom must be less than or equal to stepTo.");
      }
      const { context, detail } = yield* scopedRun(runId);
      const result = queryMetrics({
        metrics: detail.metrics,
        metricKeys,
        ...(stepFrom === undefined ? {} : { stepFrom }),
        ...(stepTo === undefined ? {} : { stepTo }),
        limit: limit ?? 200,
      });
      return { ...context, runId, ...result };
    }),
  rl_compare_runs: ({ runIds, metricKeys }) =>
    Effect.gen(function* () {
      const uniqueRunIds = [...new Set(runIds)];
      if (uniqueRunIds.length !== runIds.length) {
        return yield* fail("invalid-range", "runIds must not contain duplicates.");
      }
      const inspected = yield* Effect.all(
        uniqueRunIds.map((runId) => scopedRun(runId)),
        { concurrency: 4 },
      );
      const context = inspected[0]!.context;
      return {
        ...context,
        metricKeys,
        runs: inspected.map(({ detail }) => ({
          summary: detail.summary,
          manifest: detail.manifest,
          metricSummaries: summarizeMetrics(detail.metrics, metricKeys),
        })),
      };
    }),
  rl_read_artifact: ({ runId, artifactId, maxBytes }) =>
    readArtifact({ runId, artifactId, maxBytes: maxBytes ?? 256 * 1024 }),
  rl_start_run: ({ experimentId, seed, requestId }) =>
    Effect.gen(function* () {
      const context = yield* projectContext();
      const manager = yield* RlManager.RlManager;
      const result = yield* manager
        .start({ projectId: context.projectId, experimentId, seed, requestId })
        .pipe(
          Effect.mapError(
            (error) =>
              new RlAgentToolError({
                code: "operation-failed",
                detail: error.message,
              }),
          ),
        );
      return { ...context, runId: result.runId };
    }),
  rl_cancel_run: ({ runId }) =>
    Effect.gen(function* () {
      const { context } = yield* scopedRun(runId);
      const manager = yield* RlManager.RlManager;
      const result = yield* manager.cancel({ runId }).pipe(
        Effect.mapError(
          () =>
            new RlAgentToolError({
              code: "run-not-found",
              detail: `RL run not found in the current project: ${runId}`,
            }),
        ),
      );
      return { ...context, runId, state: result.state };
    }),
} satisfies Parameters<typeof RlToolkit.toLayer>[0];

export const RlToolkitHandlersLive = RlToolkit.toLayer(handlers);
