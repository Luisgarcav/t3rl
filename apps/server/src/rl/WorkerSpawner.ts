/**
 * WorkerSpawner - injectable host boundary for T3RL worker processes.
 *
 * Exists for the same reason as `terminal/PtyAdapter.ts`: the manager must be
 * testable without spawning anything. The live implementation wraps
 * `ChildProcessSpawner`; tests provide a fake that emits lines on demand.
 *
 * @module WorkerSpawner
 */
import * as Context from "effect/Context";
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import * as Schema from "effect/Schema";
import * as Scope from "effect/Scope";
import * as Stream from "effect/Stream";
import { ChildProcess, ChildProcessSpawner } from "effect/unstable/process";

export class WorkerSpawnError extends Schema.TaggedErrorClass<WorkerSpawnError>()(
  "RlWorkerSpawnError",
  {
    command: Schema.String,
    cause: Schema.optional(Schema.Defect()),
  },
) {
  override get message() {
    return `Failed to spawn RL worker: ${this.command}`;
  }
}

export class WorkerStreamError extends Schema.TaggedErrorClass<WorkerStreamError>()(
  "RlWorkerStreamError",
  {
    stream: Schema.Literals(["stdout", "stderr", "exitCode"]),
    cause: Schema.optional(Schema.Defect()),
  },
) {
  override get message() {
    return `RL worker ${this.stream} failed`;
  }
}

export type WorkerSignal = "SIGTERM" | "SIGKILL";

export interface WorkerProcess {
  /** Captured at spawn. Cancellation targets this and only this. */
  readonly pid: number;
  readonly stdoutLines: Stream.Stream<string, WorkerStreamError>;
  readonly stderrLines: Stream.Stream<string, WorkerStreamError>;
  readonly exitCode: Effect.Effect<number | null, WorkerStreamError>;
  readonly kill: (signal: WorkerSignal) => Effect.Effect<void>;
}

export interface WorkerSpawnInput {
  readonly command: string;
  readonly args: ReadonlyArray<string>;
  readonly cwd: string;
  /** Explicit allowlist. The server's own environment is never copied wholesale. */
  readonly env: Record<string, string>;
}

export class WorkerSpawner extends Context.Service<
  WorkerSpawner,
  {
    readonly spawn: (
      input: WorkerSpawnInput,
    ) => Effect.Effect<WorkerProcess, WorkerSpawnError, Scope.Scope>;
  }
>()("t3/rl/WorkerSpawner") {}

const toLines = (
  stream: Stream.Stream<Uint8Array, unknown>,
  which: "stdout" | "stderr",
): Stream.Stream<string, WorkerStreamError> =>
  stream.pipe(
    Stream.mapError((cause) => new WorkerStreamError({ stream: which, cause })),
    Stream.decodeText(),
    Stream.splitLines,
  );

const makeWorkerSpawner = Effect.gen(function* () {
  const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;

  return WorkerSpawner.of({
    spawn: (input) =>
      Effect.gen(function* () {
        const handle = yield* spawner
          .spawn(
            ChildProcess.make(input.command, input.args, {
              cwd: input.cwd,
              env: input.env,
              // The worker gets exactly the variables the manager listed. It
              // must not inherit provider credentials or server secrets.
              extendEnv: false,
            }),
          )
          .pipe(
            Effect.mapError((cause) => new WorkerSpawnError({ command: input.command, cause })),
          );

        return {
          pid: handle.pid,
          stdoutLines: toLines(handle.stdout, "stdout"),
          stderrLines: toLines(handle.stderr, "stderr"),
          exitCode: handle.exitCode.pipe(
            Effect.mapError((cause) => new WorkerStreamError({ stream: "exitCode", cause })),
          ),
          // Signals go to the handle captured above, never to a process found
          // by name or argv: the worker carries this worktree's path in its
          // command line exactly as the developer's own agent does.
          kill: (signal: WorkerSignal) =>
            handle.kill({ killSignal: signal }).pipe(Effect.orElseSucceed(() => undefined)),
        } satisfies WorkerProcess;
      }),
  });
});

export const WorkerSpawnerLive = Layer.effect(WorkerSpawner, makeWorkerSpawner);
