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
    Stream.mapAccum(
      () => ({ pending: Buffer.alloc(0), discarding: false }),
      (state, chunk) => {
        let data = Buffer.from(chunk);
        const lines: string[] = [];
        if (state.discarding) {
          const newline = data.indexOf(0x0a);
          if (newline < 0) return [state, lines] as const;
          data = data.subarray(newline + 1);
          state = { pending: Buffer.alloc(0), discarding: false };
        }
        if (state.pending.byteLength > 0) data = Buffer.concat([state.pending, data]);

        while (true) {
          const newline = data.indexOf(0x0a);
          if (newline < 0) break;
          const line = data.subarray(
            0,
            newline > 0 && data[newline - 1] === 0x0d ? newline - 1 : newline,
          );
          lines.push(
            line.byteLength > 64 * 1024
              ? "x".repeat(64 * 1024 + 1)
              : new TextDecoder().decode(line),
          );
          data = data.subarray(newline + 1);
        }

        if (data.byteLength > 64 * 1024) {
          lines.push("x".repeat(64 * 1024 + 1));
          return [{ pending: Buffer.alloc(0), discarding: true }, lines] as const;
        }
        return [{ pending: Buffer.from(data), discarding: false }, lines] as const;
      },
      {
        onHalt: (state) =>
          state.pending.byteLength === 0 ? [] : [new TextDecoder().decode(state.pending)],
      },
    ),
  );

const makeWorkerSpawner = Effect.gen(function* () {
  const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;

  return WorkerSpawner.of({
    spawn: (input) =>
      Effect.gen(function* () {
        const spawnScope = yield* Effect.scope;
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
          //
          // Deliberately does not await the platform's kill, which resolves
          // only once the child has actually exited. Awaiting it would hang on
          // exactly the case escalation exists for: a worker that ignores
          // SIGTERM would block the caller before it could ever send SIGKILL.
          kill: (signal: WorkerSignal) =>
            Effect.forkIn(
              handle.kill({ killSignal: signal }).pipe(Effect.orElseSucceed(() => undefined)),
              spawnScope,
            ).pipe(Effect.asVoid),
        } satisfies WorkerProcess;
      }),
  });
});

export const WorkerSpawnerLive = Layer.effect(WorkerSpawner, makeWorkerSpawner);
