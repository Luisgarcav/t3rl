/**
 * FakeWorkerSpawner - a worker process the test drives by hand.
 *
 * Modeled on FakePtyProcess in terminal/Manager.test.ts. Nothing here spawns,
 * and nothing advances on a wall clock: the test decides when a line appears
 * and when the process exits.
 *
 * @module FakeWorkerSpawner
 */
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import * as Queue from "effect/Queue";
import * as Stream from "effect/Stream";

import type { WorkerProcess, WorkerSignal, WorkerStreamError } from "../WorkerSpawner.ts";
import { WorkerSpawner, type WorkerSpawnInput } from "../WorkerSpawner.ts";

type LineListener = (line: string) => void;
type ExitListener = (code: number | null) => void;

class Emitter {
  private readonly listeners = new Set<LineListener>();
  private readonly buffered: string[] = [];
  private ended = false;

  emit(line: string): void {
    if (this.ended) return;
    if (this.listeners.size === 0) {
      this.buffered.push(line);
      return;
    }
    for (const listener of this.listeners) listener(line);
  }

  end(): void {
    this.ended = true;
    this.listeners.clear();
  }

  subscribe(listener: LineListener): () => void {
    // Lines emitted before the pump attached still have to arrive; dropping
    // them would make tests depend on fiber scheduling order.
    while (this.buffered.length > 0) {
      const line = this.buffered.shift();
      if (line !== undefined) listener(line);
    }
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  get isEnded(): boolean {
    return this.ended;
  }
}

export class FakeWorkerProcess implements WorkerProcess {
  readonly pid = 4242;
  readonly killSignals: WorkerSignal[] = [];
  private readonly killWaiters = new Set<{ count: number; resolve: () => void }>();
  readonly spawnInputs: WorkerSpawnInput[] = [];

  private readonly stdout = new Emitter();
  private readonly stderr = new Emitter();
  private readonly exitListeners = new Set<ExitListener>();
  private exitedWith: number | null | undefined;

  emitStdout(line: string): void {
    this.stdout.emit(line);
  }

  emitStderr(line: string): void {
    this.stderr.emit(line);
  }

  /** Ends both streams and resolves `exitCode`. */
  exit(code: number | null): void {
    if (this.exitedWith !== undefined) return;
    this.exitedWith = code;
    this.stdout.end();
    this.stderr.end();
    for (const listener of this.exitListeners) listener(code);
    this.exitListeners.clear();
  }

  private lineStream(emitter: Emitter): Stream.Stream<string, WorkerStreamError> {
    return Stream.callback<string, WorkerStreamError>((queue) =>
      Effect.sync(() => {
        const unsubscribe = emitter.subscribe((line) => {
          Queue.offerUnsafe(queue, line);
        });
        if (emitter.isEnded) {
          Queue.endUnsafe(queue);
        } else {
          this.onExit(() => Queue.endUnsafe(queue));
        }
        return unsubscribe;
      }).pipe(Effect.flatMap((unsubscribe) => Effect.addFinalizer(() => Effect.sync(unsubscribe)))),
    );
  }

  private onExit(listener: ExitListener): void {
    if (this.exitedWith !== undefined) {
      listener(this.exitedWith);
      return;
    }
    this.exitListeners.add(listener);
  }

  get stdoutLines(): Stream.Stream<string, WorkerStreamError> {
    return this.lineStream(this.stdout);
  }

  get stderrLines(): Stream.Stream<string, WorkerStreamError> {
    return this.lineStream(this.stderr);
  }

  get exitCode(): Effect.Effect<number | null, WorkerStreamError> {
    return Effect.callback<number | null, WorkerStreamError>((resume) => {
      this.onExit((code) => resume(Effect.succeed(code)));
    });
  }

  /**
   * Records the signal and deliberately does NOT exit. Tests that need an
   * unresponsive worker depend on that: a fake that dies on SIGTERM could
   * never prove the SIGKILL escalation happens.
   */
  kill(signal: WorkerSignal): Effect.Effect<void> {
    return Effect.sync(() => {
      this.killSignals.push(signal);
      const satisfied = this.killSignals.length;
      for (const waiter of this.killWaiters) {
        if (satisfied >= waiter.count) {
          this.killWaiters.delete(waiter);
          waiter.resolve();
        }
      }
    });
  }

  /**
   * Resolves once `count` signals have been delivered. Gives tests a real
   * synchronisation point for an escalation that produces no state change,
   * instead of guessing how many fiber yields it takes.
   */
  awaitKills(count: number): Effect.Effect<void> {
    return Effect.callback<void>((resume) => {
      if (this.killSignals.length >= count) {
        resume(Effect.void);
        return;
      }
      this.killWaiters.add({ count, resolve: () => resume(Effect.void) });
    });
  }
}

export const fakeWorkerSpawnerLayer = (worker: FakeWorkerProcess) =>
  Layer.succeed(
    WorkerSpawner,
    WorkerSpawner.of({
      spawn: (input) =>
        Effect.sync(() => {
          worker.spawnInputs.push(input);
          return worker;
        }),
    }),
  );
