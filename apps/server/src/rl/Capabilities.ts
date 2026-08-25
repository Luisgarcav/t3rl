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
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";

import * as ProcessRunner from "../processRunner.ts";

/** Gymnasium and Stable-Baselines3 both require 3.10 or newer. */
const MINIMUM_PYTHON = { major: 3, minor: 10 } as const;

const PYTHON_VERSION_PATTERN = /^Python (\d+)\.(\d+)(?:\.(\d+))?/m;

const REMEDY = `Install Python ${MINIMUM_PYTHON.major}.${MINIMUM_PYTHON.minor} or newer, or set T3RL_PYTHON to an interpreter path.`;

export interface ResolvedPython {
  readonly executable: string;
  readonly version: string;
}

export interface CapabilitiesShape {
  /**
   * Never cached: a researcher who just installed Python must see the change on
   * the next call rather than after a server restart.
   */
  readonly report: () => Effect.Effect<RlCapabilityReport>;
  readonly resolvePython: () => Effect.Effect<ResolvedPython, RlRunStartError>;
}

export class Capabilities extends Context.Service<Capabilities, CapabilitiesShape>()(
  "t3/rl/Capabilities",
) {}

const candidateExecutables = (): ReadonlyArray<string> => {
  const configured = process.env["T3RL_PYTHON"];
  return configured !== undefined && configured.trim().length > 0
    ? [configured.trim(), "python3", "python"]
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

  const report: CapabilitiesShape["report"] = () =>
    Effect.gen(function* () {
      const resolved = yield* findPython;
      return {
        runners: [
          {
            runnerId: "fake",
            available: resolved !== null,
            version: resolved?.version ?? null,
            failureCode: resolved === null ? ("PythonNotFound" as const) : null,
            remedy: resolved === null ? REMEDY : null,
          },
        ],
      } satisfies RlCapabilityReport;
    });

  return Capabilities.of({ report, resolvePython });
});

export const CapabilitiesLive = Layer.effect(Capabilities, makeCapabilities);
