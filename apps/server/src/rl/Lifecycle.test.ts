import type { RlRunState } from "@t3tools/contracts";
import { describe, expect, it } from "vite-plus/test";

import * as Lifecycle from "./Lifecycle.ts";

const accept = (state: RlRunState, event: Lifecycle.RlLifecycleEvent): RlRunState => {
  const result = Lifecycle.transition(state, event);
  if (result._tag !== "Accepted") {
    throw new Error(`expected Accepted from ${state} on ${event._tag}, got ${result.reason}`);
  }
  return result.state;
};

describe("transition", () => {
  it("walks the happy path to completed", () => {
    let state: RlRunState = "requested";
    state = accept(state, { _tag: "PreparationStarted" });
    expect(state).toBe("preparing");
    state = accept(state, { _tag: "WorkerReady" });
    expect(state).toBe("running");
    state = accept(state, { _tag: "WorkerDone", success: true });
    expect(state).toBe("completed");
  });

  it("routes a worker-reported failure to failed from every active state", () => {
    for (const state of ["requested", "preparing", "running", "cancelling"] as const) {
      expect(accept(state, { _tag: "WorkerFailed", code: "WorkerExited", message: "boom" })).toBe(
        "failed",
      );
    }
  });

  it("routes a server restart to interrupted, never to failed", () => {
    for (const state of ["requested", "preparing", "running", "cancelling"] as const) {
      expect(accept(state, { _tag: "ServerRestarted" })).toBe("interrupted");
    }
  });

  it("reaches cancelled only after the process actually stops", () => {
    let state: RlRunState = accept("running", { _tag: "CancellationRequested" });
    expect(state).toBe("cancelling");
    state = accept(state, { _tag: "WorkerStopped" });
    expect(state).toBe("cancelled");
  });

  it("lets the first terminal fact win when the worker finishes during cancellation", () => {
    expect(accept("cancelling", { _tag: "WorkerDone", success: true })).toBe("completed");
    expect(accept("cancelling", { _tag: "WorkerDone", success: false })).toBe("failed");
  });

  it("rejects every event once terminal", () => {
    for (const state of ["completed", "failed", "cancelled", "interrupted"] as const) {
      for (const event of [
        { _tag: "PreparationStarted" },
        { _tag: "WorkerReady" },
        { _tag: "CancellationRequested" },
        { _tag: "WorkerDone", success: true },
        { _tag: "ServerRestarted" },
      ] satisfies ReadonlyArray<Lifecycle.RlLifecycleEvent>) {
        expect(Lifecycle.transition(state, event)._tag).toBe("Rejected");
      }
    }
  });

  it("rejects out-of-order activation", () => {
    expect(Lifecycle.transition("requested", { _tag: "WorkerReady" })._tag).toBe("Rejected");
    expect(Lifecycle.transition("running", { _tag: "PreparationStarted" })._tag).toBe("Rejected");
    expect(Lifecycle.transition("requested", { _tag: "WorkerDone", success: true })._tag).toBe(
      "Rejected",
    );
    expect(Lifecycle.transition("preparing", { _tag: "WorkerDone", success: true })._tag).toBe(
      "Rejected",
    );
  });

  it("treats a repeated cancellation request as accepted and idempotent", () => {
    expect(accept("cancelling", { _tag: "CancellationRequested" })).toBe("cancelling");
  });
});
