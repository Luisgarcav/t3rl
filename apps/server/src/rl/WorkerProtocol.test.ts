import { describe, expect, it } from "vite-plus/test";

import * as WorkerProtocol from "./WorkerProtocol.ts";

const decode = (value: unknown): WorkerProtocol.RlWorkerDecodeResult =>
  WorkerProtocol.decodeWorkerLine(typeof value === "string" ? value : JSON.stringify(value));

const model = {
  baseModelId: "test/tiny",
  baseModelRevision: "a".repeat(40),
  tokenizerRevision: "b".repeat(40),
  peftConfig: {
    peftType: "LORA",
    taskType: "CAUSAL_LM",
    rank: 2,
    alpha: 4,
    dropout: 0,
    bias: "none",
    targetModules: ["linear"],
    modulesToSave: [],
    useRslora: false,
  },
  peftConfigSha256: "c".repeat(64),
  quantization: "none",
  precision: "fp32",
  trainableModules: ["linear"],
};

const compatibility = {
  model,
  framework: { id: "fake", version: "0.1.0" },
  environmentFingerprint: "fixture-environment",
  environmentLockSha256: null,
};

describe("decodeWorkerLine", () => {
  it("decodes a hello at the supported protocol version", () => {
    const result = decode({ type: "hello", protocol: 1, runner: "fake", runnerVersion: "0.1.0" });
    expect(result._tag).toBe("Message");
    if (result._tag !== "Message") return;
    expect(result.message._tag).toBe("Hello");
  });

  it("decodes protocol-v2 model identity, checkpoint policy, and resume evidence", () => {
    expect(
      decode({ type: "hello", protocol: 2, runner: "fake", runnerVersion: "0.1.0" })._tag,
    ).toBe("Message");
    const manifest = decode({
      type: "manifest",
      values: { maxSteps: 8 },
      model,
      checkpointPolicy: {
        cadenceSteps: 2,
        maxIntermediateCheckpoints: 2,
        keepBest: false,
        keepFinal: true,
        gracefulDeadlineSeconds: 5,
      },
    });
    expect(manifest._tag).toBe("Message");
    const artifact = decode({
      type: "artifact",
      kind: "checkpoint",
      path: "checkpoints/checkpoint-4-intermediate",
      evidence: {
        _tag: "Checkpoint",
        checkpointClass: "intermediate",
        compatibility,
        globalStep: 4,
        tokensSeen: 64,
        datasetCursor: { epoch: 0.5, batchInEpoch: 4, sampleOffset: 4 },
        resumeState: {
          trainerState: true,
          optimizerState: true,
          schedulerState: true,
          rngState: true,
          datasetCursorState: true,
          gradientScalerState: "not-applicable",
          stateFiles: [
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "t3rl-dataset-cursor.json",
          ],
        },
      },
    });
    expect(artifact._tag).toBe("Message");
    if (artifact._tag !== "Message" || artifact.message._tag !== "Artifact") return;
    expect(artifact.message.evidence?._tag).toBe("Checkpoint");
  });

  it("rejects a mutable model revision in protocol-v2 evidence", () => {
    expect(
      decode({
        type: "manifest",
        values: {},
        model: { ...model, baseModelRevision: "main" },
        checkpointPolicy: {
          cadenceSteps: 2,
          maxIntermediateCheckpoints: 2,
          keepBest: false,
          keepFinal: true,
          gracefulDeadlineSeconds: 5,
        },
      })._tag,
    ).toBe("Failure");
  });

  it("rejects a hello at an unsupported protocol version with ProtocolIncompatible", () => {
    const result = decode({ type: "hello", protocol: 99, runner: "fake", runnerVersion: "0.1.0" });
    expect(result._tag).toBe("Failure");
    if (result._tag !== "Failure") return;
    expect(result.code).toBe("ProtocolIncompatible");
  });

  it("decodes a metrics batch", () => {
    const result = decode({
      type: "metrics",
      step: 4,
      wallClockMs: 900,
      values: { "train/return": 12.5, "train/kl": null },
    });
    expect(result._tag).toBe("Message");
  });

  it("keeps protocol-v2 liveness and resource samples explicit", () => {
    expect(decode({ type: "heartbeat", step: 4, wallClockMs: 900 })).toEqual({
      _tag: "Message",
      message: { _tag: "Heartbeat", step: 4, wallClockMs: 900 },
    });
    expect(
      decode({
        type: "resource",
        step: 4,
        wallClockMs: 900,
        values: { "system/gpu_count": 2, "system/gpu_memory_allocated_gb": 3.5 },
      })._tag,
    ).toBe("Message");
    expect(
      decode({ type: "resource", step: 4, wallClockMs: 900, values: { "train/loss": 1 } })._tag,
    ).toBe("Failure");
  });

  it("rejects a raw non-finite number instead of silently reading it as null", () => {
    // JSON.stringify turns NaN into null, so a worker must send the marker
    // string. A literal `null` here would mean "not captured", which is a
    // different claim than "diverged".
    const result = WorkerProtocol.decodeWorkerLine(
      '{"type":"metrics","step":1,"wallClockMs":1,"values":{"train/loss":NaN}}',
    );
    expect(result._tag).toBe("Failure");
    if (result._tag !== "Failure") return;
    expect(result.code).toBe("MalformedWorkerMessage");
  });

  it("decodes an artifact announcement", () => {
    const result = decode({ type: "artifact", kind: "log", path: "worker.log" });
    expect(result._tag).toBe("Message");
  });

  it("rejects an artifact path that escapes the run directory", () => {
    const result = decode({ type: "artifact", kind: "log", path: "../../escape.log" });
    expect(result._tag).toBe("Failure");
    if (result._tag !== "Failure") return;
    expect(result.code).toBe("MalformedWorkerMessage");
  });

  it("decodes done", () => {
    expect(decode({ type: "done", status: "completed" })._tag).toBe("Message");
    expect(decode({ type: "done", status: "failed" })._tag).toBe("Message");
  });

  it("preserves a stable worker exception code", () => {
    const result = decode({ type: "error", code: "RunnerException", detail: "training diverged" });
    expect(result._tag).toBe("Message");
    if (result._tag !== "Message" || result.message._tag !== "Error") return;
    expect(result.message.code).toBe("RunnerException");
    expect(result.message.detail).toBe("training diverged");
  });

  it("rejects an unknown worker error code", () => {
    expect(decode({ type: "error", code: "MadeUp", detail: "nope" })._tag).toBe("Failure");
  });

  it("rejects malformed JSON", () => {
    const result = WorkerProtocol.decodeWorkerLine("{not json");
    expect(result._tag).toBe("Failure");
    if (result._tag !== "Failure") return;
    expect(result.code).toBe("MalformedWorkerMessage");
  });

  it("rejects an unknown message type", () => {
    expect(decode({ type: "telepathy" })._tag).toBe("Failure");
  });

  it("rejects an oversized line without parsing it", () => {
    const line = `{"type":"metrics","pad":"${"x".repeat(WorkerProtocol.MAX_WORKER_LINE_BYTES)}"}`;
    const result = WorkerProtocol.decodeWorkerLine(line);
    expect(result._tag).toBe("Failure");
    if (result._tag !== "Failure") return;
    expect(result.code).toBe("WorkerMessageTooLarge");
  });

  it("ignores a blank line", () => {
    expect(WorkerProtocol.decodeWorkerLine("   ")._tag).toBe("Ignored");
  });
});
