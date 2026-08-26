import { isTerminalRlRunState, type RlErrorCode, type RlRunState } from "@t3tools/contracts";

/**
 * Facts observed about a run. The manager translates process and protocol
 * observations into these; the machine itself never observes anything.
 */
export type RlLifecycleEvent =
  | { readonly _tag: "PreparationStarted" }
  | { readonly _tag: "WorkerReady" }
  | { readonly _tag: "CancellationRequested" }
  | { readonly _tag: "WorkerDone"; readonly success: boolean }
  | { readonly _tag: "WorkerFailed"; readonly code: RlErrorCode; readonly message: string }
  | { readonly _tag: "WorkerStopped" }
  | { readonly _tag: "ServerRestarted" };

export type RlTransitionResult =
  | { readonly _tag: "Accepted"; readonly state: RlRunState }
  | { readonly _tag: "Rejected"; readonly reason: string };

const accepted = (state: RlRunState): RlTransitionResult => ({ _tag: "Accepted", state });

const rejected = (state: RlRunState, event: RlLifecycleEvent): RlTransitionResult => ({
  _tag: "Rejected",
  reason: `${event._tag} is not valid while ${state}`,
});

/**
 * Pure transition. Time is never read here — callers stamp their own timestamps
 * so that tests need neither a clock nor a delay.
 */
export const transition = (state: RlRunState, event: RlLifecycleEvent): RlTransitionResult => {
  if (isTerminalRlRunState(state)) {
    return rejected(state, event);
  }

  switch (event._tag) {
    // A restart never claims to know how the run was going, so it is
    // `interrupted` rather than `failed`.
    case "ServerRestarted":
      return accepted("interrupted");

    case "WorkerFailed":
      return accepted("failed");

    // The first terminal fact wins: a worker that finished on its own before a
    // pending cancellation reached it produced a real result, and recording
    // `cancelled` would discard it.
    case "WorkerDone":
      return state === "running" || state === "cancelling"
        ? accepted(event.success ? "completed" : "failed")
        : rejected(state, event);

    case "PreparationStarted":
      return state === "requested" ? accepted("preparing") : rejected(state, event);

    case "WorkerReady":
      return state === "preparing" ? accepted("running") : rejected(state, event);

    case "CancellationRequested":
      return accepted("cancelling");

    case "WorkerStopped":
      return state === "cancelling" ? accepted("cancelled") : rejected(state, event);
  }
};
