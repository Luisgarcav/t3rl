import {
  RL_SUPPORTED_WORKER_PROTOCOL_VERSIONS,
  RlArtifactEvidence,
  RlArtifactKind,
  RlCheckpointPolicy,
  RlErrorCode,
  RlMetricBatch,
  RlModelIdentity,
} from "@t3tools/contracts";
import * as Schema from "effect/Schema";

/** A single worker line may not exceed this. Enforced before parsing. */
export const MAX_WORKER_LINE_BYTES = 64 * 1024;

export type RlWorkerMessage =
  | {
      readonly _tag: "Hello";
      readonly protocol: number;
      readonly runner: string;
      readonly runnerVersion: string;
    }
  | {
      readonly _tag: "Manifest";
      readonly values: Record<string, unknown>;
      readonly model?: RlModelIdentity | undefined;
      readonly checkpointPolicy?: RlCheckpointPolicy | undefined;
    }
  | { readonly _tag: "Metrics"; readonly batch: RlMetricBatch }
  | {
      readonly _tag: "Artifact";
      readonly kind: RlArtifactKind;
      readonly path: string;
      readonly evidence?: RlArtifactEvidence | undefined;
    }
  | { readonly _tag: "Heartbeat"; readonly step: number; readonly wallClockMs: number }
  | { readonly _tag: "Resource"; readonly batch: RlMetricBatch }
  | { readonly _tag: "Error"; readonly code: RlErrorCode; readonly detail: string }
  | { readonly _tag: "Done"; readonly success: boolean };

export type RlWorkerDecodeResult =
  | { readonly _tag: "Message"; readonly message: RlWorkerMessage }
  | { readonly _tag: "Ignored" }
  | { readonly _tag: "Failure"; readonly code: RlErrorCode; readonly detail: string };

const failure = (code: RlErrorCode, detail: string): RlWorkerDecodeResult => ({
  _tag: "Failure",
  code,
  detail,
});

const message = (value: RlWorkerMessage): RlWorkerDecodeResult => ({
  _tag: "Message",
  message: value,
});

const HelloSchema = Schema.Struct({
  type: Schema.Literals(["hello"]),
  protocol: Schema.Int,
  runner: Schema.String.check(Schema.isMaxLength(64)),
  runnerVersion: Schema.String.check(Schema.isMaxLength(64)),
});

const ManifestSchema = Schema.Struct({
  type: Schema.Literals(["manifest"]),
  values: Schema.Record(Schema.String, Schema.Unknown).check(Schema.isMaxProperties(128)),
  model: Schema.optionalKey(RlModelIdentity),
  checkpointPolicy: Schema.optionalKey(RlCheckpointPolicy),
});

const MetricsSchema = Schema.Struct({
  type: Schema.Literals(["metrics"]),
  ...RlMetricBatch.fields,
});

const ArtifactSchema = Schema.Struct({
  type: Schema.Literals(["artifact"]),
  kind: RlArtifactKind,
  path: Schema.String.check(Schema.isNonEmpty()).check(Schema.isMaxLength(512)),
  evidence: Schema.optionalKey(RlArtifactEvidence),
});

const HeartbeatSchema = Schema.Struct({
  type: Schema.Literals(["heartbeat"]),
  step: RlMetricBatch.fields.step,
  wallClockMs: RlMetricBatch.fields.wallClockMs,
});

const ResourceSchema = Schema.Struct({
  type: Schema.Literals(["resource"]),
  ...RlMetricBatch.fields,
});

const ErrorSchema = Schema.Struct({
  type: Schema.Literals(["error"]),
  code: RlErrorCode,
  detail: Schema.String.check(Schema.isMaxLength(2048)),
});

const DoneSchema = Schema.Struct({
  type: Schema.Literals(["done"]),
  status: Schema.Literals(["completed", "failed"]),
});

const decodeOrNull = <S extends Schema.Top>(
  schema: S,
  input: unknown,
): Schema.Schema.Type<S> | null => {
  try {
    return Schema.decodeUnknownSync(schema as never)(input) as Schema.Schema.Type<S>;
  } catch {
    return null;
  }
};

/**
 * Rejects any path that could leave the run directory. The manager re-checks
 * this against the resolved filesystem root; this is the cheap first gate.
 */
const isSafeRelativePath = (value: string): boolean =>
  !value.includes("\0") &&
  !value.startsWith("/") &&
  !/^[A-Za-z]:/.test(value) &&
  !value.split(/[/\\]/).includes("..");

export const decodeWorkerLine = (line: string): RlWorkerDecodeResult => {
  if (line.trim().length === 0) {
    return { _tag: "Ignored" };
  }
  if (Buffer.byteLength(line, "utf8") > MAX_WORKER_LINE_BYTES) {
    return failure("WorkerMessageTooLarge", "worker line exceeded the size limit");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch {
    return failure("MalformedWorkerMessage", "worker line was not valid JSON");
  }

  if (typeof parsed !== "object" || parsed === null || !("type" in parsed)) {
    return failure("MalformedWorkerMessage", "worker line had no message type");
  }

  switch ((parsed as { type: unknown }).type) {
    case "hello": {
      const hello = decodeOrNull(HelloSchema, parsed);
      if (hello === null) return failure("MalformedWorkerMessage", "invalid hello");
      if (
        !(RL_SUPPORTED_WORKER_PROTOCOL_VERSIONS as ReadonlyArray<number>).includes(hello.protocol)
      ) {
        return failure(
          "ProtocolIncompatible",
          `worker speaks protocol ${hello.protocol}; server supports ${RL_SUPPORTED_WORKER_PROTOCOL_VERSIONS.join(", ")}`,
        );
      }
      return message({
        _tag: "Hello",
        protocol: hello.protocol,
        runner: hello.runner,
        runnerVersion: hello.runnerVersion,
      });
    }
    case "manifest": {
      const manifest = decodeOrNull(ManifestSchema, parsed);
      return manifest === null
        ? failure("MalformedWorkerMessage", "invalid manifest")
        : message({
            _tag: "Manifest",
            values: manifest.values,
            ...(manifest.model === undefined ? {} : { model: manifest.model }),
            ...(manifest.checkpointPolicy === undefined
              ? {}
              : { checkpointPolicy: manifest.checkpointPolicy }),
          });
    }
    case "metrics": {
      const metrics = decodeOrNull(MetricsSchema, parsed);
      return metrics === null
        ? failure("MalformedWorkerMessage", "invalid metrics batch")
        : message({
            _tag: "Metrics",
            batch: {
              step: metrics.step,
              wallClockMs: metrics.wallClockMs,
              values: metrics.values,
            },
          });
    }
    case "artifact": {
      const artifact = decodeOrNull(ArtifactSchema, parsed);
      if (artifact === null) return failure("MalformedWorkerMessage", "invalid artifact");
      if (!isSafeRelativePath(artifact.path)) {
        return failure("MalformedWorkerMessage", "artifact path left the run directory");
      }
      return message({
        _tag: "Artifact",
        kind: artifact.kind,
        path: artifact.path,
        ...(artifact.evidence === undefined ? {} : { evidence: artifact.evidence }),
      });
    }
    case "heartbeat": {
      const heartbeat = decodeOrNull(HeartbeatSchema, parsed);
      return heartbeat === null
        ? failure("MalformedWorkerMessage", "invalid heartbeat")
        : message({
            _tag: "Heartbeat",
            step: heartbeat.step,
            wallClockMs: heartbeat.wallClockMs,
          });
    }
    case "resource": {
      const resource = decodeOrNull(ResourceSchema, parsed);
      if (resource === null) return failure("MalformedWorkerMessage", "invalid resource sample");
      if (Object.keys(resource.values).some((key) => !key.startsWith("system/"))) {
        return failure("MalformedWorkerMessage", "resource keys must use the system namespace");
      }
      return message({
        _tag: "Resource",
        batch: {
          step: resource.step,
          wallClockMs: resource.wallClockMs,
          values: resource.values,
        },
      });
    }
    case "error": {
      const error = decodeOrNull(ErrorSchema, parsed);
      return error === null
        ? failure("MalformedWorkerMessage", "invalid error message")
        : message({
            _tag: "Error",
            code: error.code,
            detail: error.detail,
          });
    }
    case "done": {
      const done = decodeOrNull(DoneSchema, parsed);
      return done === null
        ? failure("MalformedWorkerMessage", "invalid done message")
        : message({ _tag: "Done", success: done.status === "completed" });
    }
    default:
      return failure("MalformedWorkerMessage", "unknown worker message type");
  }
};
