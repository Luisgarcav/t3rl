import { describe, expect, it } from "vite-plus/test";

import * as WorkerProtocol from "./WorkerProtocol.ts";

const decode = (value: unknown): WorkerProtocol.RlWorkerDecodeResult =>
  WorkerProtocol.decodeWorkerLine(typeof value === "string" ? value : JSON.stringify(value));

describe("decodeWorkerLine", () => {
  it("decodes a hello at the supported protocol version", () => {
    const result = decode({ type: "hello", protocol: 1, runner: "fake", runnerVersion: "0.1.0" });
    expect(result._tag).toBe("Message");
    if (result._tag !== "Message") return;
    expect(result.message._tag).toBe("Hello");
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
    expect(result.code).toBe("MalformedWorkerMessage");
  });

  it("ignores a blank line", () => {
    expect(WorkerProtocol.decodeWorkerLine("   ")._tag).toBe("Ignored");
  });
});
