/**
 * Capabilities - what the host can actually run.
 *
 * Discovery only. The lab never installs Python, native packages, or framework
 * wheels; an unavailable runner is reported with actionable remedy text so the
 * researcher decides what to install.
 *
 * @module Capabilities
 */
import { RlCapabilityReport, RlRunStartError } from "@t3tools/contracts";
import * as Context from "effect/Context";
import * as Crypto from "effect/Crypto";
import * as Effect from "effect/Effect";
import * as Encoding from "effect/Encoding";
import * as Layer from "effect/Layer";
import * as Option from "effect/Option";
import * as Schema from "effect/Schema";

import * as ProcessRunner from "../processRunner.ts";

/** Gymnasium and Stable-Baselines3 both require 3.10 or newer. */
const MINIMUM_PYTHON = { major: 3, minor: 10 } as const;
const SUPPORTED_TRL_VERSION = "1.10.0";

const PYTHON_VERSION_PATTERN = /^Python (\d+)\.(\d+)(?:\.(\d+))?/m;

const REMEDY = `Install Python ${MINIMUM_PYTHON.major}.${MINIMUM_PYTHON.minor} or newer, or set T3RL_PYTHON to an interpreter path.`;
const SB3_REMEDY =
  "Select a Python 3.10–3.13 virtual environment with Gymnasium, Stable-Baselines3, NumPy, and PyTorch by setting T3RL_PYTHON.";
const TRL_REMEDY =
  "Create a uv environment and install the optional LLM runtime with `uv pip install --python .venv/bin/python -e './python[llm]'`, then set T3RL_PYTHON to that interpreter.";
const TRL_CUDA_REMEDY =
  "The bundled GRPO/RLVR experiment requires a CUDA GPU visible to the configured PyTorch runtime.";

const SB3_PROBE = `
import json, os, platform, sys
import gymnasium, stable_baselines3, numpy, torch
env = gymnasium.make("CartPole-v1")
try:
    env.reset(seed=0)
    env.action_space.seed(0)
    env.step(env.action_space.sample())
finally:
    env.close()
print(json.dumps({"executable": os.path.realpath(sys.executable), "python": platform.python_version(), "platform": platform.platform(), "stableBaselines3": stable_baselines3.__version__, "gymnasium": gymnasium.__version__, "numpy": numpy.__version__, "torch": torch.__version__}, sort_keys=True, separators=(",", ":")))
`;

const TRL_PROBE = `
import importlib.metadata as metadata, json, os, platform, sys
import accelerate, datasets, torch, transformers, trl
print(json.dumps({"executable": os.path.realpath(sys.executable), "python": platform.python_version(), "platform": platform.platform(), "trl": metadata.version("trl"), "transformers": metadata.version("transformers"), "datasets": metadata.version("datasets"), "accelerate": metadata.version("accelerate"), "torch": metadata.version("torch"), "cudaAvailable": torch.cuda.is_available(), "cudaDeviceCount": torch.cuda.device_count()}, sort_keys=True, separators=(",", ":")))
`;

const Sb3ProbeResult = Schema.Struct({
  executable: Schema.String.check(Schema.isMaxLength(1024)),
  python: Schema.String.check(Schema.isMaxLength(64)),
  platform: Schema.String.check(Schema.isMaxLength(256)),
  stableBaselines3: Schema.String.check(Schema.isMaxLength(64)),
  gymnasium: Schema.String.check(Schema.isMaxLength(64)),
  numpy: Schema.String.check(Schema.isMaxLength(64)),
  torch: Schema.String.check(Schema.isMaxLength(64)),
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
  cudaAvailable: Schema.Boolean,
  cudaDeviceCount: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
});
const decodeTrlProbe = Schema.decodeUnknownOption(Schema.fromJsonString(TrlProbeResult));

export interface ResolvedPython {
  readonly executable: string;
  readonly version: string;
}

export interface ResolvedRunner extends ResolvedPython {
  readonly runnerId: string;
  readonly runnerVersion: string;
  readonly environmentFingerprint: string;
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
  }) => Effect.Effect<ResolvedRunner, RlRunStartError>;
}

export class Capabilities extends Context.Service<Capabilities, CapabilitiesShape>()(
  "t3/rl/Capabilities",
) {}

const candidateExecutables = (): ReadonlyArray<string> => {
  const configured = process.env["T3RL_PYTHON"];
  return configured !== undefined && configured.trim().length > 0
    ? [configured.trim()]
    : ["python3", "python"];
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

  const findPython = Effect.gen(function* () {
    for (const executable of candidateExecutables()) {
      const resolved = yield* probe(executable);
      if (resolved !== null) return resolved;
    }
    return null;
  });

  const resolvePython: CapabilitiesShape["resolvePython"] = () =>
    Effect.gen(function* () {
      const resolved = yield* findPython;
      if (resolved === null) {
        return yield* new RlRunStartError({ code: "PythonNotFound", detail: REMEDY });
      }
      return resolved;
    });

  const fingerprint = (value: string) =>
    crypto.digest("SHA-256", new TextEncoder().encode(value)).pipe(
      Effect.orDie,
      Effect.map((bytes) => Encoding.encodeHex(bytes)),
    );

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

  const resolveRunner: CapabilitiesShape["resolveRunner"] = (input) =>
    Effect.gen(function* () {
      const python = yield* resolvePython();
      if (input.runnerId === "fake") {
        return {
          ...python,
          runnerId: "fake",
          runnerVersion: "0.1.0",
          environmentFingerprint: yield* fingerprint(`python=${python.version};runner=fake@0.1.0`),
        };
      }
      if (input.runnerId === "trl") {
        const probe = yield* probeTrl(python);
        if (probe === null || probe.trl !== SUPPORTED_TRL_VERSION) {
          return yield* new RlRunStartError({ code: "RunnerUnavailable", detail: TRL_REMEDY });
        }
        if (!probe.cudaAvailable) {
          return yield* new RlRunStartError({ code: "RunnerUnavailable", detail: TRL_CUDA_REMEDY });
        }
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
            JSON.stringify(probe),
          ),
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
      return {
        executable: python.executable,
        version: probe.python,
        runnerId: input.runnerId,
        runnerVersion: probe.stableBaselines3,
        environmentFingerprint: yield* fingerprint(
          // @effect-diagnostics-next-line preferSchemaOverJson:off - canonical bounded probe.
          JSON.stringify(probe),
        ),
      };
    });

  const report: CapabilitiesShape["report"] = () =>
    Effect.gen(function* () {
      const resolved = yield* findPython;
      const sb3 = resolved === null ? null : yield* probeSb3(resolved);
      const trl = resolved === null ? null : yield* probeTrl(resolved);
      const trlVersionSupported = trl?.trl === SUPPORTED_TRL_VERSION;
      return {
        runners: [
          {
            runnerId: "fake",
            available: resolved !== null,
            version: resolved === null ? null : "0.1.0",
            failureCode: resolved === null ? ("PythonNotFound" as const) : null,
            remedy: resolved === null ? REMEDY : null,
          },
          {
            runnerId: "stable-baselines3",
            available: sb3 !== null,
            version: sb3?.stableBaselines3 ?? null,
            failureCode:
              resolved === null
                ? ("PythonNotFound" as const)
                : sb3 === null
                  ? ("RunnerUnavailable" as const)
                  : null,
            remedy: resolved === null ? REMEDY : sb3 === null ? SB3_REMEDY : null,
          },
          {
            runnerId: "trl",
            available: trlVersionSupported && trl?.cudaAvailable === true,
            version: trl?.trl ?? null,
            failureCode:
              resolved === null
                ? ("PythonNotFound" as const)
                : !trlVersionSupported || trl?.cudaAvailable !== true
                  ? ("RunnerUnavailable" as const)
                  : null,
            remedy:
              resolved === null
                ? REMEDY
                : !trlVersionSupported
                  ? TRL_REMEDY
                  : trl?.cudaAvailable === true
                    ? null
                    : TRL_CUDA_REMEDY,
          },
        ],
        experiments: [],
      } satisfies RlCapabilityReport;
    });

  return Capabilities.of({ report, resolvePython, resolveRunner });
});

export const CapabilitiesLive = Layer.effect(Capabilities, makeCapabilities);
