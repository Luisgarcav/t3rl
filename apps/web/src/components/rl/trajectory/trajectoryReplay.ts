export const DEFAULT_MAX_REPLAY_BYTES = 4 * 1024 * 1024;
export const DEFAULT_MAX_TRAJECTORY_STEPS = 10_000;

const MAX_ENVIRONMENT_LENGTH = 128;
const MAX_JSON_DEPTH = 6;
const MAX_JSON_COLLECTION_ITEMS = 256;
const MAX_JSON_OBJECT_KEYS = 64;
const MAX_JSON_STRING_LENGTH = 1_024;

export type TrajectoryJsonValue =
  | null
  | boolean
  | number
  | string
  | ReadonlyArray<TrajectoryJsonValue>
  | { readonly [key: string]: TrajectoryJsonValue };

export interface TrajectoryStep {
  readonly step: number;
  readonly observation: TrajectoryJsonValue;
  readonly action: TrajectoryJsonValue;
  readonly reward: number;
  readonly terminated: boolean;
  readonly truncated: boolean;
}

export interface TrajectoryReplay {
  readonly environment: string;
  readonly evaluationSeed: number;
  readonly trajectory: ReadonlyArray<TrajectoryStep>;
}

export type TrajectoryReplayErrorCode =
  | "http-error"
  | "invalid-json"
  | "invalid-schema"
  | "network-error"
  | "too-large";

export class TrajectoryReplayError extends Error {
  readonly code: TrajectoryReplayErrorCode;

  constructor(code: TrajectoryReplayErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "TrajectoryReplayError";
    this.code = code;
  }
}

export interface ParseTrajectoryReplayOptions {
  readonly maxSteps?: number | undefined;
}

export interface LoadTrajectoryReplayOptions extends ParseTrajectoryReplayOptions {
  readonly fetchImpl?: typeof fetch | undefined;
  readonly maxBytes?: number | undefined;
  readonly signal?: AbortSignal | undefined;
}

function schemaError(message: string): never {
  throw new TrajectoryReplayError("invalid-schema", message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (!isRecord(value)) schemaError(`${path} must be an object.`);
  return value;
}

function requireBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") schemaError(`${path} must be a boolean.`);
  return value;
}

function requireFiniteNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    schemaError(`${path} must be a finite number.`);
  }
  return value;
}

function requireSafeInteger(value: unknown, path: string, minimum?: number): number {
  if (!Number.isSafeInteger(value) || (minimum !== undefined && Number(value) < minimum)) {
    schemaError(
      minimum === undefined
        ? `${path} must be a safe integer.`
        : `${path} must be a safe integer greater than or equal to ${minimum}.`,
    );
  }
  return Number(value);
}

function requireJsonValue(value: unknown, path: string, depth = 0): TrajectoryJsonValue {
  if (depth > MAX_JSON_DEPTH) {
    schemaError(`${path} exceeds the maximum nesting depth of ${MAX_JSON_DEPTH}.`);
  }
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "number") return requireFiniteNumber(value, path);
  if (typeof value === "string") {
    if (value.length > MAX_JSON_STRING_LENGTH) {
      schemaError(`${path} exceeds ${MAX_JSON_STRING_LENGTH} characters.`);
    }
    return value;
  }
  if (Array.isArray(value)) {
    if (value.length > MAX_JSON_COLLECTION_ITEMS) {
      schemaError(`${path} exceeds ${MAX_JSON_COLLECTION_ITEMS} items.`);
    }
    return value.map((item, index) => requireJsonValue(item, `${path}[${index}]`, depth + 1));
  }
  if (isRecord(value)) {
    const entries = Object.entries(value);
    if (entries.length > MAX_JSON_OBJECT_KEYS) {
      schemaError(`${path} exceeds ${MAX_JSON_OBJECT_KEYS} properties.`);
    }
    return Object.fromEntries(
      entries.map(([key, item]) => {
        if (key.length > MAX_JSON_STRING_LENGTH) {
          schemaError(`${path} contains a property name longer than ${MAX_JSON_STRING_LENGTH}.`);
        }
        return [key, requireJsonValue(item, `${path}.${key}`, depth + 1)];
      }),
    );
  }
  return schemaError(`${path} must be JSON-compatible data.`);
}

function normalizePositiveLimit(
  value: number | undefined,
  fallback: number,
  label: string,
): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved <= 0) {
    throw new TrajectoryReplayError("invalid-schema", `${label} must be a positive safe integer.`);
  }
  return resolved;
}

export function parseTrajectoryReplay(
  value: unknown,
  options: ParseTrajectoryReplayOptions = {},
): TrajectoryReplay {
  const root = requireRecord(value, "Replay");
  const environment = root.environment;
  if (
    typeof environment !== "string" ||
    environment.trim().length === 0 ||
    environment.length > MAX_ENVIRONMENT_LENGTH
  ) {
    schemaError(
      `Replay.environment must be a non-empty string of at most ${MAX_ENVIRONMENT_LENGTH} characters.`,
    );
  }
  const evaluationSeed = requireSafeInteger(root.evaluationSeed, "Replay.evaluationSeed");
  const rawTrajectory = root.trajectory;
  if (!Array.isArray(rawTrajectory)) schemaError("Replay.trajectory must be an array.");
  const maxSteps = normalizePositiveLimit(
    options.maxSteps,
    DEFAULT_MAX_TRAJECTORY_STEPS,
    "maxSteps",
  );
  if (rawTrajectory.length === 0) schemaError("Replay.trajectory must contain at least one step.");
  if (rawTrajectory.length > maxSteps) {
    schemaError(`Replay.trajectory exceeds the limit of ${maxSteps} steps.`);
  }

  let previousStep = -1;
  const trajectory = rawTrajectory.map((rawStep, index): TrajectoryStep => {
    const path = `Replay.trajectory[${index}]`;
    const stepRecord = requireRecord(rawStep, path);
    const step = requireSafeInteger(stepRecord.step, `${path}.step`, 0);
    if (step <= previousStep) {
      schemaError(`${path}.step must be greater than the previous step (${previousStep}).`);
    }
    previousStep = step;
    return {
      step,
      observation: requireJsonValue(stepRecord.observation, `${path}.observation`),
      action: requireJsonValue(stepRecord.action, `${path}.action`),
      reward: requireFiniteNumber(stepRecord.reward, `${path}.reward`),
      terminated: requireBoolean(stepRecord.terminated, `${path}.terminated`),
      truncated: requireBoolean(stepRecord.truncated, `${path}.truncated`),
    };
  });

  return { environment, evaluationSeed, trajectory };
}

export function parseTrajectoryReplayJson(
  text: string,
  options: ParseTrajectoryReplayOptions = {},
): TrajectoryReplay {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new TrajectoryReplayError("invalid-json", "The replay artifact is not valid JSON.", {
      cause: error,
    });
  }
  return parseTrajectoryReplay(value, options);
}

export async function readBoundedResponseText(
  response: Response,
  maxBytes = DEFAULT_MAX_REPLAY_BYTES,
): Promise<string> {
  const byteLimit = normalizePositiveLimit(maxBytes, DEFAULT_MAX_REPLAY_BYTES, "maxBytes");
  const declaredLength = response.headers.get("content-length");
  if (declaredLength !== null) {
    const parsedLength = Number(declaredLength);
    if (Number.isFinite(parsedLength) && parsedLength > byteLimit) {
      throw new TrajectoryReplayError(
        "too-large",
        `The replay artifact exceeds the ${byteLimit.toLocaleString()} byte limit.`,
      );
    }
  }
  if (response.body === null) {
    throw new TrajectoryReplayError("invalid-json", "The replay artifact has no response body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const chunks: string[] = [];
  let bytesRead = 0;
  try {
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      bytesRead += result.value.byteLength;
      if (bytesRead > byteLimit) {
        await reader.cancel();
        throw new TrajectoryReplayError(
          "too-large",
          `The replay artifact exceeds the ${byteLimit.toLocaleString()} byte limit.`,
        );
      }
      chunks.push(decoder.decode(result.value, { stream: true }));
    }
    chunks.push(decoder.decode());
  } catch (error) {
    if (error instanceof TrajectoryReplayError) throw error;
    throw new TrajectoryReplayError("invalid-json", "The replay artifact is not valid UTF-8.", {
      cause: error,
    });
  } finally {
    reader.releaseLock();
  }
  return chunks.join("");
}

export async function loadTrajectoryReplay(
  url: string,
  options: LoadTrajectoryReplayOptions = {},
): Promise<TrajectoryReplay> {
  if (url.trim().length === 0) {
    throw new TrajectoryReplayError("network-error", "The replay artifact URL is empty.");
  }
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  let response: Response;
  try {
    response = await fetchImpl(url, {
      headers: { Accept: "application/json" },
      signal: options.signal ?? null,
    });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw new TrajectoryReplayError("network-error", "The replay artifact could not be loaded.", {
      cause: error,
    });
  }
  if (!response.ok) {
    throw new TrajectoryReplayError(
      "http-error",
      `The replay artifact request failed with HTTP ${response.status}.`,
    );
  }
  const text = await readBoundedResponseText(response, options.maxBytes);
  return parseTrajectoryReplayJson(text, { maxSteps: options.maxSteps });
}

export function clampTrajectoryIndex(index: number, stepCount: number): number {
  if (stepCount <= 0) return 0;
  return Math.min(stepCount - 1, Math.max(0, Math.trunc(index)));
}

export type TrajectoryTerminalKind = "continuing" | "terminated" | "truncated";

export function trajectoryTerminalKind(step: TrajectoryStep): TrajectoryTerminalKind {
  if (step.terminated) return "terminated";
  if (step.truncated) return "truncated";
  return "continuing";
}
