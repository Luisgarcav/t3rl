import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, it } from "@effect/vitest";
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
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
                cudaDeviceCount: cudaAvailable ? 1 : 0,
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

describe("Capabilities", () => {
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

  it.effect("keeps an installed TRL runtime unavailable without CUDA", () =>
    Effect.gen(function* () {
      const capabilities = yield* Capabilities.Capabilities;
      const report = yield* capabilities.report();
      const trl = report.runners.find((runner) => runner.runnerId === "trl");
      assert.strictEqual(trl?.available, false);
      assert.strictEqual(trl?.failureCode, "RunnerUnavailable");
      assert.include(trl?.remedy ?? "", "CUDA");

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
      assert.include(trl?.remedy ?? "", "python[llm]");
    }).pipe(withTrlRunner(true, "0.24.0")),
  );
});
