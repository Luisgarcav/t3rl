/**
 * Capabilities - what the host can actually run.
 *
 * Discovery only. The lab never installs Python, native packages, or framework
 * wheels; an unavailable runner is reported with actionable remedy text so the
 * researcher decides what to install.
 *
 * @module Capabilities
 */
import { RlCapabilityReport, RlRunStartError, type RlEnvironmentLock } from "@t3tools/contracts";
import * as Context from "effect/Context";
import * as Crypto from "effect/Crypto";
import * as Effect from "effect/Effect";
import * as Encoding from "effect/Encoding";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Option from "effect/Option";
import * as Path from "effect/Path";
import * as Schema from "effect/Schema";

import * as ProcessRunner from "../processRunner.ts";

/** Gymnasium and Stable-Baselines3 both require 3.10 or newer. */
const MINIMUM_PYTHON = { major: 3, minor: 10 } as const;
const SUPPORTED_TRL_VERSION = "1.10.0";
const SUPPORTED_AXOLOTL_VERSION = "0.18.0";

const PYTHON_VERSION_PATTERN = /^Python (\d+)\.(\d+)(?:\.(\d+))?/m;

const REMEDY = `Install Python ${MINIMUM_PYTHON.major}.${MINIMUM_PYTHON.minor} or newer, or set T3RL_PYTHON to an interpreter path.`;
const SB3_REMEDY =
  "Run `uv sync --project python/environments/sb3 --locked`, then set T3RL_PYTHON_STABLE_BASELINES3 to that environment's interpreter.";
const TRL_REMEDY =
  "Run `uv sync --project python/environments/trl --locked`, then set T3RL_PYTHON_TRL to that environment's interpreter.";
const TRL_CUDA_REMEDY =
  "The bundled GRPO/RLVR experiment requires a CUDA GPU visible to the configured PyTorch runtime.";
// Axolotl pins exact dependency versions and no release accepts the TRL version
// the native adapter targets, so it gets its own environment rather than a
// downgrade of a runner that already works.
const AXOLOTL_REMEDY =
  "Run `uv sync --project python/environments/axolotl --locked`, then set T3RL_PYTHON_AXOLOTL to that environment's interpreter.";
const AXOLOTL_CUDA_REMEDY =
  "The bundled Axolotl GRPO experiment requires a CUDA GPU visible to the configured PyTorch runtime.";

const SB3_PROBE = `
import json, os, platform, sys
import gymnasium, stable_baselines3, numpy, torch
def driver_version():
    try:
        return str(torch._C._cuda_getDriverVersion())
    except Exception:
        return None
env = gymnasium.make("CartPole-v1")
try:
    env.reset(seed=0)
    env.action_space.seed(0)
    env.step(env.action_space.sample())
finally:
    env.close()
print(json.dumps({"executable": os.path.realpath(sys.executable), "python": platform.python_version(), "platform": platform.platform(), "stableBaselines3": stable_baselines3.__version__, "gymnasium": gymnasium.__version__, "numpy": numpy.__version__, "torch": torch.__version__, "cudaAvailable": torch.cuda.is_available(), "cudaRuntime": torch.version.cuda, "cudaDeviceCount": torch.cuda.device_count(), "driverVersion": driver_version()}, sort_keys=True, separators=(",", ":")))
`;

const TRL_PROBE = `
import importlib.metadata as metadata, json, os, platform, sys
import accelerate, datasets, torch, transformers, trl
def driver_version():
    try:
        return str(torch._C._cuda_getDriverVersion())
    except Exception:
        return None
print(json.dumps({"executable": os.path.realpath(sys.executable), "python": platform.python_version(), "platform": platform.platform(), "trl": metadata.version("trl"), "transformers": metadata.version("transformers"), "datasets": metadata.version("datasets"), "accelerate": metadata.version("accelerate"), "torch": metadata.version("torch"), "cudaAvailable": torch.cuda.is_available(), "cudaRuntime": torch.version.cuda, "cudaDeviceCount": torch.cuda.device_count(), "driverVersion": driver_version()}, sort_keys=True, separators=(",", ":")))
`;

const AXOLOTL_PROBE = `
import importlib.metadata as metadata, json, os, platform, sys
import axolotl, torch, transformers, trl
def driver_version():
    try:
        return str(torch._C._cuda_getDriverVersion())
    except Exception:
        return None
print(json.dumps({"executable": os.path.realpath(sys.executable), "python": platform.python_version(), "platform": platform.platform(), "axolotl": metadata.version("axolotl"), "trl": metadata.version("trl"), "transformers": metadata.version("transformers"), "accelerate": metadata.version("accelerate"), "torch": metadata.version("torch"), "cudaAvailable": torch.cuda.is_available(), "cudaRuntime": torch.version.cuda, "cudaDeviceCount": torch.cuda.device_count(), "driverVersion": driver_version()}, sort_keys=True, separators=(",", ":")))
`;

const CudaProbeFields = {
  cudaAvailable: Schema.Boolean,
  cudaRuntime: Schema.NullOr(Schema.String.check(Schema.isMaxLength(64))),
  cudaDeviceCount: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  driverVersion: Schema.NullOr(Schema.String.check(Schema.isMaxLength(64))),
};

const Sb3ProbeResult = Schema.Struct({
  executable: Schema.String.check(Schema.isMaxLength(1024)),
  python: Schema.String.check(Schema.isMaxLength(64)),
  platform: Schema.String.check(Schema.isMaxLength(256)),
  stableBaselines3: Schema.String.check(Schema.isMaxLength(64)),
  gymnasium: Schema.String.check(Schema.isMaxLength(64)),
  numpy: Schema.String.check(Schema.isMaxLength(64)),
  torch: Schema.String.check(Schema.isMaxLength(64)),
  ...CudaProbeFields,
});
const decodeSb3Probe = Schema.decodeUnknownOption(Schema.fromJsonString(Sb3ProbeResult));

const TrlProbeResult = Schema.Struct({
  executable: Schema.String.check(Schema.isMaxLength(1024)),
  python: Schema.String.check(Schema.isMaxLength(64)),
  platform: Schema.String.check(Schema.isMaxLength(256)),
  trl: Schema.String.check(Schema.isMaxLength(64)),
  transformers: Schema.String.check(Schema.isMaxLength(64)),
  datasets: Schema.String.check(Schema.isMaxLength(64)),
  accelerate: Schema.String.check(Schema.isMaxLength(64)),
  torch: Schema.String.check(Schema.isMaxLength(64)),
  ...CudaProbeFields,
});
const decodeTrlProbe = Schema.decodeUnknownOption(Schema.fromJsonString(TrlProbeResult));

const AxolotlProbeResult = Schema.Struct({
  executable: Schema.String.check(Schema.isMaxLength(1024)),
  python: Schema.String.check(Schema.isMaxLength(64)),
  platform: Schema.String.check(Schema.isMaxLength(256)),
  axolotl: Schema.String.check(Schema.isMaxLength(64)),
  trl: Schema.String.check(Schema.isMaxLength(64)),
  transformers: Schema.String.check(Schema.isMaxLength(64)),
  accelerate: Schema.String.check(Schema.isMaxLength(64)),
  torch: Schema.String.check(Schema.isMaxLength(64)),
  ...CudaProbeFields,
});
const decodeAxolotlProbe = Schema.decodeUnknownOption(Schema.fromJsonString(AxolotlProbeResult));

export interface ResolvedPython {
  readonly executable: string;
  readonly version: string;
}

export interface ResolvedRunner extends ResolvedPython {
  readonly runnerId: string;
  readonly runnerVersion: string;
  readonly environmentFingerprint: string;
  readonly environmentLock: RlEnvironmentLock | null;
}

export interface CapabilitiesShape {
  /**
   * Never cached: a researcher who just installed Python must see the change on
   * the next call rather than after a server restart.
   */
  readonly report: () => Effect.Effect<RlCapabilityReport>;
  readonly resolvePython: () => Effect.Effect<ResolvedPython, RlRunStartError>;
  readonly resolveRunner: (input: {
    readonly runnerId: string;
    readonly method?: "sft" | "dpo" | "grpo" | "rloo" | "ppo" | undefined;
  }) => Effect.Effect<ResolvedRunner, RlRunStartError>;
}

export class Capabilities extends Context.Service<Capabilities, CapabilitiesShape>()(
  "t3/rl/Capabilities",
) {}

/**
 * Runners whose dependencies cannot coexist get their own interpreter. The
 * variable is derived from the runner id so adding a backend needs no new
 * plumbing here.
 */
const runnerEnvironmentVariable = (runnerId: string): string =>
  `T3RL_PYTHON_${runnerId.toUpperCase().replaceAll("-", "_")}`;

/**
 * A dedicated interpreter is tried before the shared one. An explicitly
 * configured interpreter never falls back to whatever is on PATH: silently
 * running a different environment than the one a researcher named is how a run
 * ends up unreproducible.
 */
const candidateExecutables = (runnerId?: string): ReadonlyArray<string> => {
  const configured = [
    runnerId === undefined ? undefined : process.env[runnerEnvironmentVariable(runnerId)],
    process.env["T3RL_PYTHON"],
  ].flatMap((value) => (value !== undefined && value.trim().length > 0 ? [value.trim()] : []));
  return configured.length > 0 ? configured : ["python3", "python"];
};

const parseVersion = (
  stdout: string,
): { readonly version: string; readonly major: number; readonly minor: number } | null => {
  const match = PYTHON_VERSION_PATTERN.exec(stdout);
  if (match === null) return null;
  const major = Number(match[1]);
  const minor = Number(match[2]);
  if (!Number.isFinite(major) || !Number.isFinite(minor)) return null;
  return { version: `${major}.${minor}.${match[3] ?? "0"}`, major, minor };
};

const makeCapabilities = Effect.gen(function* () {
  const runner = yield* ProcessRunner.ProcessRunner;
  const crypto = yield* Crypto.Crypto;
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;

  const probe = (executable: string) =>
    runner.run({ command: executable, args: ["--version"], timeout: "10 seconds" }).pipe(
      Effect.map((output) => {
        // Older interpreters print the version on stderr.
        const parsed = parseVersion(`${output.stdout}\n${output.stderr}`);
        if (parsed === null) return null;
        if (
          parsed.major < MINIMUM_PYTHON.major ||
          (parsed.major === MINIMUM_PYTHON.major && parsed.minor < MINIMUM_PYTHON.minor)
        ) {
          return null;
        }
        return { executable, version: parsed.version } satisfies ResolvedPython;
      }),
      Effect.orElseSucceed(() => null),
    );

  const findPython = (runnerId?: string) =>
    Effect.gen(function* () {
      for (const executable of candidateExecutables(runnerId)) {
        const resolved = yield* probe(executable);
        if (resolved !== null) return resolved;
      }
      return null;
    });

  const resolveRunnerPython = (runnerId?: string) =>
    Effect.gen(function* () {
      const resolved = yield* findPython(runnerId);
      if (resolved === null) {
        return yield* new RlRunStartError({ code: "PythonNotFound", detail: REMEDY });
      }
      return resolved;
    });

  const resolvePython: CapabilitiesShape["resolvePython"] = () => resolveRunnerPython();

  const fingerprint = (value: string) =>
    crypto.digest("SHA-256", new TextEncoder().encode(value)).pipe(
      Effect.orDie,
      Effect.map((bytes) => Encoding.encodeHex(bytes)),
    );

  const findUvProject = (pythonExecutable: string) =>
    Effect.gen(function* () {
      if (!path.isAbsolute(pythonExecutable)) return null;
      let candidate = path.dirname(path.resolve(pythonExecutable));
      for (let depth = 0; depth < 6; depth += 1) {
        const pyprojectPath = path.join(candidate, "pyproject.toml");
        const lockfilePath = path.join(candidate, "uv.lock");
        const [hasProject, hasLockfile] = yield* Effect.all([
          fs.exists(pyprojectPath),
          fs.exists(lockfilePath),
        ]);
        if (hasProject || hasLockfile) {
          return { projectPath: candidate, lockfilePath, hasProject, hasLockfile };
        }
        const parent = path.dirname(candidate);
        if (parent === candidate) return null;
        candidate = parent;
      }
      return null;
    }).pipe(
      Effect.mapError(
        () =>
          new RlRunStartError({
            code: "RunnerUnavailable",
            detail: `Could not inspect the uv project for ${pythonExecutable}.`,
          }),
      ),
    );

  const resolveEnvironmentLock = (input: {
    readonly python: ResolvedPython;
    readonly frameworkId: string;
    readonly frameworkVersion: string;
    readonly probe: {
      readonly python: string;
      readonly platform: string;
      readonly torch: string;
      readonly cudaAvailable: boolean;
      readonly cudaRuntime: string | null;
      readonly cudaDeviceCount: number;
      readonly driverVersion: string | null;
    };
  }) =>
    Effect.gen(function* () {
      const project = yield* findUvProject(input.python.executable);
      if (project === null) return null;
      const checkCommand = `uv lock --check --project ${project.projectPath}`;
      const syncCommand = `uv sync --project ${project.projectPath} --locked`;
      if (!project.hasProject || !project.hasLockfile) {
        return null;
      }
      const checked = yield* runner
        .run({
          command: "uv",
          args: ["lock", "--check", "--project", project.projectPath],
          timeout: "30 seconds",
          maxOutputBytes: 16 * 1024,
        })
        .pipe(
          Effect.mapError(
            () =>
              new RlRunStartError({
                code: "RunnerUnavailable",
                detail: `The uv lock could not be checked. Run \`${checkCommand}\`; if needed, reproduce the environment with \`${syncCommand}\`.`,
              }),
          ),
        );
      if (checked.code !== 0) {
        return yield* new RlRunStartError({
          code: "RunnerUnavailable",
          detail: `The selected environment does not match its committed lockfile. Run \`${syncCommand}\`, then verify with \`${checkCommand}\`.`,
        });
      }
      const lockfile = yield* fs.readFile(project.lockfilePath).pipe(
        Effect.mapError(
          () =>
            new RlRunStartError({
              code: "RunnerUnavailable",
              detail: `The checked lockfile could not be read: ${project.lockfilePath}`,
            }),
        ),
      );
      const lockfileSha256 = yield* crypto
        .digest("SHA-256", lockfile)
        .pipe(Effect.map(Encoding.encodeHex), Effect.orDie);
      return {
        projectPath: project.projectPath,
        lockfilePath: project.lockfilePath,
        lockfileSha256,
        pythonExecutable: input.python.executable,
        pythonVersion: input.probe.python,
        platform: input.probe.platform,
        framework: { id: input.frameworkId, version: input.frameworkVersion },
        pytorchVersion: input.probe.torch,
        cudaAvailable: input.probe.cudaAvailable,
        cudaRuntime: input.probe.cudaRuntime,
        cudaDeviceCount: input.probe.cudaDeviceCount,
        driverVersion: input.probe.driverVersion,
      } satisfies RlEnvironmentLock;
    });

  const probeSb3 = (python: ResolvedPython) =>
    runner
      .run({
        command: python.executable,
        args: ["-c", SB3_PROBE],
        timeout: "15 seconds",
        maxOutputBytes: 16 * 1024,
      })
      .pipe(
        Effect.map((output) => {
          if (output.code !== 0) return null;
          return Option.getOrNull(decodeSb3Probe(output.stdout.trim()));
        }),
        Effect.orElseSucceed(() => null),
      );

  const probeTrl = (python: ResolvedPython) =>
    runner
      .run({
        command: python.executable,
        args: ["-c", TRL_PROBE],
        timeout: "20 seconds",
        maxOutputBytes: 16 * 1024,
      })
      .pipe(
        Effect.map((output) => {
          if (output.code !== 0) return null;
          return Option.getOrNull(decodeTrlProbe(output.stdout.trim()));
        }),
        Effect.orElseSucceed(() => null),
      );

  const probeAxolotl = (python: ResolvedPython) =>
    runner
      .run({
        command: python.executable,
        args: ["-c", AXOLOTL_PROBE],
        timeout: "60 seconds",
        maxOutputBytes: 16 * 1024,
      })
      .pipe(
        Effect.map((output) => {
          if (output.code !== 0) return null;
          return Option.getOrNull(decodeAxolotlProbe(output.stdout.trim()));
        }),
        Effect.orElseSucceed(() => null),
      );

  const resolveRunner: CapabilitiesShape["resolveRunner"] = (input) =>
    Effect.gen(function* () {
      const python = yield* resolveRunnerPython(input.runnerId);
      if (input.runnerId === "fake") {
        return {
          ...python,
          runnerId: "fake",
          runnerVersion: "0.1.0",
          environmentFingerprint: yield* fingerprint(`python=${python.version};runner=fake@0.1.0`),
          environmentLock: null,
        };
      }
      if (input.runnerId === "axolotl") {
        const probed = yield* probeAxolotl(python);
        if (probed === null || probed.axolotl !== SUPPORTED_AXOLOTL_VERSION) {
          return yield* new RlRunStartError({ code: "RunnerUnavailable", detail: AXOLOTL_REMEDY });
        }
        if (!probed.cudaAvailable && input.method !== "sft" && input.method !== "dpo") {
          return yield* new RlRunStartError({
            code: "RunnerUnavailable",
            detail: AXOLOTL_CUDA_REMEDY,
          });
        }
        const environmentLock = yield* resolveEnvironmentLock({
          python,
          frameworkId: input.runnerId,
          frameworkVersion: probed.axolotl,
          probe: probed,
        });
        return {
          executable: python.executable,
          version: probed.python,
          runnerId: input.runnerId,
          runnerVersion: probed.axolotl,
          environmentFingerprint: yield* fingerprint(
            // @effect-diagnostics-next-line preferSchemaOverJson:off - canonical bounded probe.
            JSON.stringify({ probed, lockfileSha256: environmentLock?.lockfileSha256 ?? null }),
          ),
          environmentLock,
        };
      }
      if (input.runnerId === "trl") {
        const probe = yield* probeTrl(python);
        if (probe === null || probe.trl !== SUPPORTED_TRL_VERSION) {
          return yield* new RlRunStartError({ code: "RunnerUnavailable", detail: TRL_REMEDY });
        }
        if (!probe.cudaAvailable && input.method !== "sft" && input.method !== "dpo") {
          return yield* new RlRunStartError({ code: "RunnerUnavailable", detail: TRL_CUDA_REMEDY });
        }
        const environmentLock = yield* resolveEnvironmentLock({
          python,
          frameworkId: input.runnerId,
          frameworkVersion: probe.trl,
          probe,
        });
        return {
          // Keep the configured launcher instead of the probe's realpath. Virtual
          // environments created by tools such as uv commonly symlink their
          // launcher to a base interpreter; spawning that realpath would bypass
          // the environment's sys.prefix and installed packages.
          executable: python.executable,
          version: probe.python,
          runnerId: input.runnerId,
          runnerVersion: probe.trl,
          environmentFingerprint: yield* fingerprint(
            // @effect-diagnostics-next-line preferSchemaOverJson:off - canonical bounded probe.
            JSON.stringify({ probe, lockfileSha256: environmentLock?.lockfileSha256 ?? null }),
          ),
          environmentLock,
        };
      }
      if (input.runnerId !== "stable-baselines3") {
        return yield* new RlRunStartError({
          code: "RunnerUnavailable",
          detail: `Unsupported RL runner: ${input.runnerId}`,
        });
      }

      const probe = yield* probeSb3(python);
      if (probe === null) {
        return yield* new RlRunStartError({ code: "RunnerUnavailable", detail: SB3_REMEDY });
      }
      const environmentLock = yield* resolveEnvironmentLock({
        python,
        frameworkId: input.runnerId,
        frameworkVersion: probe.stableBaselines3,
        probe,
      });
      return {
        executable: python.executable,
        version: probe.python,
        runnerId: input.runnerId,
        runnerVersion: probe.stableBaselines3,
        environmentFingerprint: yield* fingerprint(
          // @effect-diagnostics-next-line preferSchemaOverJson:off - canonical bounded probe.
          JSON.stringify({ probe, lockfileSha256: environmentLock?.lockfileSha256 ?? null }),
        ),
        environmentLock,
      };
    });

  const report: CapabilitiesShape["report"] = () =>
    Effect.gen(function* () {
      const resolved = yield* findPython();
      const sb3Python = yield* findPython("stable-baselines3");
      const sb3 = sb3Python === null ? null : yield* probeSb3(sb3Python);
      const trlPython = yield* findPython("trl");
      const trl = trlPython === null ? null : yield* probeTrl(trlPython);
      const trlVersionSupported = trl?.trl === SUPPORTED_TRL_VERSION;
      const axolotlPython = yield* findPython("axolotl");
      const axolotl = axolotlPython === null ? null : yield* probeAxolotl(axolotlPython);
      const axolotlVersionSupported = axolotl?.axolotl === SUPPORTED_AXOLOTL_VERSION;
      const lockFailure = (input: Parameters<typeof resolveEnvironmentLock>[0]) =>
        resolveEnvironmentLock(input).pipe(
          Effect.as(null as string | null),
          Effect.catchTag("RlRunStartError", (error) => Effect.succeed(error.detail)),
        );
      const sb3LockFailure =
        sb3Python === null || sb3 === null
          ? null
          : yield* lockFailure({
              python: sb3Python,
              frameworkId: "stable-baselines3",
              frameworkVersion: sb3.stableBaselines3,
              probe: sb3,
            });
      const trlLockFailure =
        trlPython === null || !trlVersionSupported || trl?.cudaAvailable !== true
          ? null
          : yield* lockFailure({
              python: trlPython,
              frameworkId: "trl",
              frameworkVersion: trl.trl,
              probe: trl,
            });
      const axolotlLockFailure =
        axolotlPython === null || !axolotlVersionSupported || axolotl?.cudaAvailable !== true
          ? null
          : yield* lockFailure({
              python: axolotlPython,
              frameworkId: "axolotl",
              frameworkVersion: axolotl.axolotl,
              probe: axolotl,
            });
      return {
        runners: [
          {
            runnerId: "fake",
            available: resolved !== null,
            version: resolved === null ? null : "0.1.0",
            failureCode: resolved === null ? ("PythonNotFound" as const) : null,
            remedy: resolved === null ? REMEDY : null,
            methods: [],
            methodCapabilities: [],
          },
          {
            runnerId: "stable-baselines3",
            available: sb3 !== null && sb3LockFailure === null,
            version: sb3?.stableBaselines3 ?? null,
            failureCode:
              sb3Python === null
                ? ("PythonNotFound" as const)
                : sb3 === null || sb3LockFailure !== null
                  ? ("RunnerUnavailable" as const)
                  : null,
            remedy: sb3Python === null ? REMEDY : sb3 === null ? SB3_REMEDY : sb3LockFailure,
            methods: ["ppo"],
            methodCapabilities: [
              {
                method: "ppo",
                available: sb3 !== null && sb3LockFailure === null,
                remedy: sb3Python === null ? REMEDY : sb3 === null ? SB3_REMEDY : sb3LockFailure,
              },
            ],
          },
          {
            runnerId: "trl",
            available: trlVersionSupported && trlLockFailure === null,
            version: trl?.trl ?? null,
            failureCode:
              trlPython === null
                ? ("PythonNotFound" as const)
                : !trlVersionSupported || trlLockFailure !== null
                  ? ("RunnerUnavailable" as const)
                  : null,
            remedy:
              trlPython === null ? REMEDY : !trlVersionSupported ? TRL_REMEDY : trlLockFailure,
            methods: ["sft", "dpo", "grpo", "rloo", "ppo"],
            methodCapabilities: [
              {
                method: "sft",
                available: trlVersionSupported && trlLockFailure === null,
                remedy: trlVersionSupported ? trlLockFailure : TRL_REMEDY,
              },
              {
                method: "dpo",
                available: trlVersionSupported && trlLockFailure === null,
                remedy: trlVersionSupported ? trlLockFailure : TRL_REMEDY,
              },
              {
                method: "grpo",
                available:
                  trlVersionSupported && trl?.cudaAvailable === true && trlLockFailure === null,
                remedy: trl?.cudaAvailable === true ? trlLockFailure : TRL_CUDA_REMEDY,
              },
              { method: "rloo", available: false, remedy: "RLOO is planned for milestone 6." },
              { method: "ppo", available: false, remedy: "TRL PPO is planned for milestone 6." },
            ],
          },
          {
            runnerId: "axolotl",
            available: axolotlVersionSupported && axolotlLockFailure === null,
            version: axolotl?.axolotl ?? null,
            failureCode:
              axolotlPython === null
                ? ("PythonNotFound" as const)
                : !axolotlVersionSupported || axolotlLockFailure !== null
                  ? ("RunnerUnavailable" as const)
                  : null,
            remedy:
              axolotlPython === null
                ? REMEDY
                : !axolotlVersionSupported
                  ? AXOLOTL_REMEDY
                  : axolotlLockFailure,
            methods: ["sft", "dpo", "grpo"],
            methodCapabilities: [
              {
                method: "sft",
                available: axolotlVersionSupported && axolotlLockFailure === null,
                remedy: axolotlVersionSupported ? axolotlLockFailure : AXOLOTL_REMEDY,
              },
              {
                method: "dpo",
                available: axolotlVersionSupported && axolotlLockFailure === null,
                remedy: axolotlVersionSupported ? axolotlLockFailure : AXOLOTL_REMEDY,
              },
              {
                method: "grpo",
                available:
                  axolotlVersionSupported &&
                  axolotl?.cudaAvailable === true &&
                  axolotlLockFailure === null,
                remedy: axolotl?.cudaAvailable === true ? axolotlLockFailure : AXOLOTL_CUDA_REMEDY,
              },
            ],
          },
        ],
        experiments: [],
      } satisfies RlCapabilityReport;
    });

  return Capabilities.of({ report, resolvePython, resolveRunner });
});

export const CapabilitiesLive = Layer.effect(Capabilities, makeCapabilities);
