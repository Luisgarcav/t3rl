import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Path from "effect/Path";
import { ChildProcessSpawner } from "effect/unstable/process";
import { describe } from "vite-plus/test";

import * as ProcessRunner from "../processRunner.ts";
import * as Capabilities from "./Capabilities.ts";

const fakeRunner = (outputs: Record<string, { stdout: string; code: number }>) =>
  Layer.succeed(
    ProcessRunner.ProcessRunner,
    ProcessRunner.ProcessRunner.of({
      run: (input) => {
        const result = outputs[input.command];
        if (result === undefined) {
          return Effect.fail(
            new ProcessRunner.ProcessSpawnError({
              command: input.command,
              argumentCount: input.args.length,
              resolvedCommand: input.command,
              resolvedArgumentCount: input.args.length,
              shell: false,
              cause: "not found",
            }),
          );
        }
        return Effect.succeed({
          stdout: result.stdout,
          stderr: "",
          code: ChildProcessSpawner.ExitCode(result.code),
          timedOut: false,
          stdoutTruncated: false,
          stderrTruncated: false,
          stdoutInvalidUtf8: false,
          stderrInvalidUtf8: false,
        });
      },
    }),
  );

const withRunner = (outputs: Record<string, { stdout: string; code: number }>) =>
  Effect.provide(
    Capabilities.CapabilitiesLive.pipe(
      Layer.provide(fakeRunner(outputs)),
      Layer.provide(NodeServices.layer),
    ),
  );

const trlRunner = (cudaAvailable: boolean, trlVersion = "1.10.0") =>
  Layer.succeed(
    ProcessRunner.ProcessRunner,
    ProcessRunner.ProcessRunner.of({
      run: (input) => {
        const isVersion = input.args[0] === "--version";
        const isTrlProbe = input.args[0] === "-c" && input.args[1]?.includes("import accelerate");
        const stdout = isVersion
          ? "Python 3.12.4\n"
          : isTrlProbe
            ? JSON.stringify({
                executable: "/opt/t3rl/bin/python",
                python: "3.12.4",
                platform: "test-platform",
                trl: trlVersion,
                transformers: "5.0.0",
                datasets: "4.0.0",
                accelerate: "1.0.0",
                torch: "2.9.0",
                cudaAvailable,
                cudaRuntime: cudaAvailable ? "12.8" : null,
                cudaDeviceCount: cudaAvailable ? 1 : 0,
                driverVersion: cudaAvailable ? "12080" : null,
              })
            : "";
        return Effect.succeed({
          stdout,
          stderr: "",
          code: ChildProcessSpawner.ExitCode(isVersion || isTrlProbe ? 0 : 1),
          timedOut: false,
          stdoutTruncated: false,
          stderrTruncated: false,
          stdoutInvalidUtf8: false,
          stderrInvalidUtf8: false,
        });
      },
    }),
  );

const withTrlRunner = (cudaAvailable: boolean, trlVersion?: string) =>
  Effect.provide(
    Capabilities.CapabilitiesLive.pipe(
      Layer.provide(trlRunner(cudaAvailable, trlVersion)),
      Layer.provide(NodeServices.layer),
    ),
  );

/** Restores every touched variable, so interpreter tests cannot leak into each other. */
const withEnvironment = <A, E, R>(
  variables: Record<string, string | undefined>,
  effect: Effect.Effect<A, E, R>,
) =>
  Effect.acquireUseRelease(
    Effect.sync(() => {
      const previous: Record<string, string | undefined> = {};
      for (const [name, value] of Object.entries(variables)) {
        previous[name] = process.env[name];
        if (value === undefined) delete process.env[name];
        else process.env[name] = value;
      }
      return previous;
    }),
    () => effect,
    (previous) =>
      Effect.sync(() => {
        for (const [name, value] of Object.entries(previous)) {
          if (value === undefined) delete process.env[name];
          else process.env[name] = value;
        }
      }),
  );

interface FakeEnvironment {
  readonly python: string;
  readonly trl?: string;
  readonly axolotl?: string;
  readonly cudaAvailable?: boolean;
}

/**
 * Answers only for the interpreters it is given, so a test proves which
 * executable the resolver actually reached for.
 */
const interpreters = (environments: Record<string, FakeEnvironment>) =>
  Layer.succeed(
    ProcessRunner.ProcessRunner,
    ProcessRunner.ProcessRunner.of({
      run: (input) => {
        const environment = environments[input.command];
        if (environment === undefined) {
          return Effect.fail(
            new ProcessRunner.ProcessSpawnError({
              command: input.command,
              argumentCount: input.args.length,
              resolvedCommand: input.command,
              resolvedArgumentCount: input.args.length,
              shell: false,
              cause: "not found",
            }),
          );
        }
        const script = input.args[1] ?? "";
        const isVersion = input.args[0] === "--version";
        const isAxolotlProbe = input.args[0] === "-c" && script.includes("import axolotl");
        const isTrlProbe = input.args[0] === "-c" && script.includes("import accelerate");
        const cudaAvailable = environment.cudaAvailable ?? true;
        const shared = {
          executable: input.command,
          python: environment.python,
          platform: "test-platform",
          transformers: "5.0.0",
          torch: "2.9.0",
          accelerate: "1.0.0",
          cudaAvailable,
          cudaRuntime: cudaAvailable ? "12.8" : null,
          cudaDeviceCount: cudaAvailable ? 1 : 0,
          driverVersion: cudaAvailable ? "12080" : null,
        };
        const available =
          isVersion ||
          (isAxolotlProbe && environment.axolotl !== undefined) ||
          (isTrlProbe && environment.trl !== undefined);
        const stdout = isVersion
          ? `Python ${environment.python}\n`
          : isAxolotlProbe && environment.axolotl !== undefined
            ? JSON.stringify({ ...shared, axolotl: environment.axolotl, trl: "1.8.0" })
            : isTrlProbe && environment.trl !== undefined
              ? JSON.stringify({ ...shared, trl: environment.trl, datasets: "4.0.0" })
              : "";
        return Effect.succeed({
          stdout,
          stderr: "",
          code: ChildProcessSpawner.ExitCode(available ? 0 : 1),
          timedOut: false,
          stdoutTruncated: false,
          stderrTruncated: false,
          stdoutInvalidUtf8: false,
          stderrInvalidUtf8: false,
        });
      },
    }),
  );

const withInterpreters = (environments: Record<string, FakeEnvironment>) =>
  Effect.provide(
    Capabilities.CapabilitiesLive.pipe(
      Layer.provide(interpreters(environments)),
      Layer.provide(NodeServices.layer),
    ),
  );

const lockedSb3Runner = (
  python: string,
  lockMatches: boolean,
  calls: Array<{ readonly command: string; readonly args: ReadonlyArray<string> }>,
) =>
  Layer.succeed(
    ProcessRunner.ProcessRunner,
    ProcessRunner.ProcessRunner.of({
      run: (input) => {
        calls.push({ command: input.command, args: input.args });
        const isVersion = input.command === python && input.args[0] === "--version";
        const isProbe = input.command === python && input.args[0] === "-c";
        const isLockCheck = input.command === "uv" && input.args[0] === "lock";
        const stdout = isVersion
          ? "Python 3.12.4\n"
          : isProbe
            ? JSON.stringify({
                executable: python,
                python: "3.12.4",
                platform: "test-platform",
                stableBaselines3: "2.9.0",
                gymnasium: "1.3.0",
                numpy: "2.3.0",
                torch: "2.9.0",
                cudaAvailable: false,
                cudaRuntime: null,
                cudaDeviceCount: 0,
                driverVersion: null,
              })
            : "";
        return Effect.succeed({
          stdout,
          stderr: isLockCheck && !lockMatches ? "lockfile needs to be updated" : "",
          code: ChildProcessSpawner.ExitCode(
            isVersion || isProbe || (isLockCheck && lockMatches) ? 0 : 1,
          ),
          timedOut: false,
          stdoutTruncated: false,
          stderrTruncated: false,
          stdoutInvalidUtf8: false,
          stderrInvalidUtf8: false,
        });
      },
    }),
  );

describe("Capabilities", () => {
  it.effect("records checked uv lock evidence for the selected interpreter", () =>
    Effect.scoped(
      Effect.gen(function* () {
        const fs = yield* FileSystem.FileSystem;
        const path = yield* Path.Path;
        const project = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-uv-lock-" });
        const python = path.join(project, ".venv", "bin", "python");
        yield* fs.makeDirectory(path.dirname(python), { recursive: true });
        yield* fs.writeFileString(python, "");
        yield* fs.writeFileString(path.join(project, "pyproject.toml"), "[project]\nname='test'\n");
        yield* fs.writeFileString(path.join(project, "uv.lock"), "version = 1\n");
        const calls: Array<{ command: string; args: ReadonlyArray<string> }> = [];

        const resolved = yield* withEnvironment(
          { T3RL_PYTHON_STABLE_BASELINES3: python, T3RL_PYTHON: undefined },
          Effect.gen(function* () {
            const capabilities = yield* Capabilities.Capabilities;
            return yield* capabilities.resolveRunner({ runnerId: "stable-baselines3" });
          }).pipe(
            Effect.provide(
              Capabilities.CapabilitiesLive.pipe(
                Layer.provide(lockedSb3Runner(python, true, calls)),
                Layer.provide(NodeServices.layer),
              ),
            ),
          ),
        );

        assert.strictEqual(resolved.environmentLock?.projectPath, project);
        assert.strictEqual(resolved.environmentLock?.lockfilePath, path.join(project, "uv.lock"));
        assert.match(resolved.environmentLock?.lockfileSha256 ?? "", /^[0-9a-f]{64}$/);
        assert.deepStrictEqual(resolved.environmentLock?.framework, {
          id: "stable-baselines3",
          version: "2.9.0",
        });
        assert.isTrue(
          calls.some(
            (call) =>
              call.command === "uv" && call.args.join(" ") === `lock --check --project ${project}`,
          ),
        );
      }),
    ).pipe(Effect.provide(NodeServices.layer)),
  );

  it.effect("rejects a stale uv lock without mutating the environment", () =>
    Effect.scoped(
      Effect.gen(function* () {
        const fs = yield* FileSystem.FileSystem;
        const path = yield* Path.Path;
        const project = yield* fs.makeTempDirectoryScoped({ prefix: "t3rl-stale-lock-" });
        const python = path.join(project, ".venv", "bin", "python");
        yield* fs.makeDirectory(path.dirname(python), { recursive: true });
        yield* fs.writeFileString(python, "");
        yield* fs.writeFileString(path.join(project, "pyproject.toml"), "[project]\nname='test'\n");
        yield* fs.writeFileString(path.join(project, "uv.lock"), "version = 1\n");
        const calls: Array<{ command: string; args: ReadonlyArray<string> }> = [];

        const result = yield* withEnvironment(
          { T3RL_PYTHON_STABLE_BASELINES3: python, T3RL_PYTHON: undefined },
          Effect.gen(function* () {
            const capabilities = yield* Capabilities.Capabilities;
            const report = yield* capabilities.report();
            const error = yield* Effect.flip(
              capabilities.resolveRunner({ runnerId: "stable-baselines3" }),
            );
            return { error, report };
          }).pipe(
            Effect.provide(
              Capabilities.CapabilitiesLive.pipe(
                Layer.provide(lockedSb3Runner(python, false, calls)),
                Layer.provide(NodeServices.layer),
              ),
            ),
          ),
        );

        const reported = result.report.runners.find(
          (runner) => runner.runnerId === "stable-baselines3",
        );
        assert.strictEqual(reported?.available, false);
        assert.include(reported?.remedy ?? "", `uv sync --project ${project} --locked`);
        assert.include(result.error.detail, `uv sync --project ${project} --locked`);
        assert.isFalse(calls.some((call) => call.args[0] === "sync"));
      }),
    ).pipe(Effect.provide(NodeServices.layer)),
  );

  it.effect("resolves the axolotl runner through its dedicated interpreter", () =>
    withEnvironment(
      { T3RL_PYTHON_AXOLOTL: "/opt/axolotl/bin/python", T3RL_PYTHON: undefined },
      Effect.gen(function* () {
        const capabilities = yield* Capabilities.Capabilities;
        const resolved = yield* capabilities.resolveRunner({ runnerId: "axolotl" });
        assert.strictEqual(resolved.executable, "/opt/axolotl/bin/python");
        assert.strictEqual(resolved.runnerVersion, "0.18.0");
      }).pipe(
        withInterpreters({
          "/opt/axolotl/bin/python": { python: "3.12.4", axolotl: "0.18.0" },
        }),
      ),
    ),
  );

  it.effect("keeps the axolotl interpreter out of every other runner", () =>
    withEnvironment(
      { T3RL_PYTHON_AXOLOTL: "/opt/axolotl/bin/python", T3RL_PYTHON: "/opt/shared/bin/python" },
      Effect.gen(function* () {
        const capabilities = yield* Capabilities.Capabilities;
        const resolved = yield* capabilities.resolveRunner({ runnerId: "trl" });
        assert.strictEqual(resolved.executable, "/opt/shared/bin/python");
        assert.strictEqual(resolved.runnerVersion, "1.10.0");
      }).pipe(
        withInterpreters({
          "/opt/axolotl/bin/python": { python: "3.12.4", axolotl: "0.18.0" },
          "/opt/shared/bin/python": { python: "3.12.4", trl: "1.10.0" },
        }),
      ),
    ),
  );

  it.effect("falls back to the shared interpreter when a runner has no dedicated one", () =>
    withEnvironment(
      { T3RL_PYTHON_AXOLOTL: undefined, T3RL_PYTHON: "/opt/shared/bin/python" },
      Effect.gen(function* () {
        const capabilities = yield* Capabilities.Capabilities;
        const resolved = yield* capabilities.resolveRunner({ runnerId: "axolotl" });
        assert.strictEqual(resolved.executable, "/opt/shared/bin/python");
      }).pipe(
        withInterpreters({
          "/opt/shared/bin/python": { python: "3.12.4", axolotl: "0.18.0" },
        }),
      ),
    ),
  );

  it.effect("keeps Axolotl offline methods available without claiming GRPO CUDA support", () =>
    withEnvironment(
      { T3RL_PYTHON_AXOLOTL: undefined, T3RL_PYTHON: undefined },
      Effect.gen(function* () {
        const capabilities = yield* Capabilities.Capabilities;
        const report = yield* capabilities.report();
        const axolotl = report.runners.find((runner) => runner.runnerId === "axolotl");
        assert.strictEqual(axolotl?.available, true);
        assert.strictEqual(axolotl?.failureCode, null);
        assert.strictEqual(
          axolotl?.methodCapabilities?.find((entry) => entry.method === "sft")?.available,
          true,
        );
        assert.strictEqual(
          axolotl?.methodCapabilities?.find((entry) => entry.method === "grpo")?.available,
          false,
        );

        const exit = yield* Effect.exit(capabilities.resolveRunner({ runnerId: "axolotl" }));
        assert.isTrue(exit._tag === "Failure");
      }).pipe(
        withInterpreters({
          python3: { python: "3.12.4", axolotl: "0.18.0", cudaAvailable: false },
        }),
      ),
    ),
  );

  it.effect("rejects an Axolotl version the bundled adapter does not target", () =>
    withEnvironment(
      { T3RL_PYTHON_AXOLOTL: undefined, T3RL_PYTHON: undefined },
      Effect.gen(function* () {
        const capabilities = yield* Capabilities.Capabilities;
        const report = yield* capabilities.report();
        const axolotl = report.runners.find((runner) => runner.runnerId === "axolotl");
        assert.strictEqual(axolotl?.available, false);
        assert.include(axolotl?.remedy ?? "", "T3RL_PYTHON_AXOLOTL");
      }).pipe(
        withInterpreters({
          python3: { python: "3.12.4", axolotl: "0.17.0" },
        }),
      ),
    ),
  );
  it.effect("reports the fake runner as available when python3 answers", () =>
    Effect.gen(function* () {
      const capabilities = yield* Capabilities.Capabilities;
      const report = yield* capabilities.report();
      const fake = report.runners.find((runner) => runner.runnerId === "fake");
      assert.strictEqual(fake?.available, true);
      assert.strictEqual(fake?.failureCode, null);
      assert.strictEqual(fake?.version, "0.1.0");
    }).pipe(withRunner({ python3: { stdout: "Python 3.12.4\n", code: 0 } })),
  );

  it.effect("reports PythonNotFound with a remedy instead of installing anything", () =>
    Effect.gen(function* () {
      const capabilities = yield* Capabilities.Capabilities;
      const report = yield* capabilities.report();
      const fake = report.runners.find((runner) => runner.runnerId === "fake");
      assert.strictEqual(fake?.available, false);
      assert.strictEqual(fake?.failureCode, "PythonNotFound");
      assert.isTrue((fake?.remedy ?? "").length > 0);
    }).pipe(withRunner({})),
  );

  it.effect("falls back to python when python3 is absent", () =>
    Effect.gen(function* () {
      const capabilities = yield* Capabilities.Capabilities;
      const resolved = yield* capabilities.resolvePython();
      assert.strictEqual(resolved.executable, "python");
      assert.strictEqual(resolved.version, "3.11.9");
    }).pipe(withRunner({ python: { stdout: "Python 3.11.9\n", code: 0 } })),
  );

  it.effect("rejects a Python older than the supported floor", () =>
    Effect.gen(function* () {
      const capabilities = yield* Capabilities.Capabilities;
      const exit = yield* Effect.exit(capabilities.resolvePython());
      assert.isTrue(exit._tag === "Failure");
    }).pipe(withRunner({ python3: { stdout: "Python 3.7.1\n", code: 0 } })),
  );

  it.effect("treats unparseable version output as unavailable rather than guessing", () =>
    Effect.gen(function* () {
      const capabilities = yield* Capabilities.Capabilities;
      const exit = yield* Effect.exit(capabilities.resolvePython());
      assert.isTrue(exit._tag === "Failure");
    }).pipe(withRunner({ python3: { stdout: "something else entirely\n", code: 0 } })),
  );

  it.effect("resolves TRL with CUDA while preserving the probed Python launcher", () =>
    Effect.gen(function* () {
      const capabilities = yield* Capabilities.Capabilities;
      const report = yield* capabilities.report();
      const trl = report.runners.find((runner) => runner.runnerId === "trl");
      assert.strictEqual(trl?.available, true);
      assert.strictEqual(trl?.version, "1.10.0");

      const resolved = yield* capabilities.resolveRunner({ runnerId: "trl" });
      assert.strictEqual(resolved.executable, "python3");
      assert.strictEqual(resolved.runnerVersion, "1.10.0");
      assert.match(resolved.environmentFingerprint, /^[0-9a-f]{64}$/);
    }).pipe(withTrlRunner(true)),
  );

  it.effect("keeps TRL offline methods available without claiming GRPO CUDA support", () =>
    Effect.gen(function* () {
      const capabilities = yield* Capabilities.Capabilities;
      const report = yield* capabilities.report();
      const trl = report.runners.find((runner) => runner.runnerId === "trl");
      assert.strictEqual(trl?.available, true);
      assert.strictEqual(trl?.failureCode, null);
      assert.strictEqual(
        trl?.methodCapabilities?.find((entry) => entry.method === "dpo")?.available,
        true,
      );
      assert.strictEqual(
        trl?.methodCapabilities?.find((entry) => entry.method === "grpo")?.available,
        false,
      );

      const exit = yield* Effect.exit(capabilities.resolveRunner({ runnerId: "trl" }));
      assert.isTrue(exit._tag === "Failure");
    }).pipe(withTrlRunner(false)),
  );

  it.effect("rejects a TRL version the bundled adapter does not target", () =>
    Effect.gen(function* () {
      const capabilities = yield* Capabilities.Capabilities;
      const report = yield* capabilities.report();
      const trl = report.runners.find((runner) => runner.runnerId === "trl");
      assert.strictEqual(trl?.available, false);
      assert.strictEqual(trl?.version, "0.24.0");
      assert.include(trl?.remedy ?? "", "python/environments/trl");
    }).pipe(withTrlRunner(true, "0.24.0")),
  );
});
