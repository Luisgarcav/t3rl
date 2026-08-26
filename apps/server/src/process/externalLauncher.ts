/**
 * ExternalLauncher - external application launch service interface.
 *
 * Owns process launch helpers for browser URLs and workspace paths
 * in configured editor integrations.
 *
 * @module ExternalLauncher
 */
import {
  EDITORS,
  ExternalLauncherError,
  ExternalLauncherBrowserSpawnError,
  ExternalLauncherCommandNotFoundError,
  ExternalLauncherEditorSpawnError,
  ExternalLauncherUnknownEditorError,
  ExternalLauncherUnsupportedEditorError,
  type EditorId,
  type LaunchEditorInput,
} from "@t3tools/contracts";
import { HostProcessPlatform } from "@t3tools/shared/hostProcess";
import { isCommandAvailable, resolveCommandPath, resolveSpawnCommand } from "@t3tools/shared/shell";
import * as Clock from "effect/Clock";
import * as Config from "effect/Config";
import * as Context from "effect/Context";
import * as Effect from "effect/Effect";
import * as Encoding from "effect/Encoding";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Option from "effect/Option";
import * as Path from "effect/Path";
import * as Ref from "effect/Ref";
import * as ChildProcess from "effect/unstable/process/ChildProcess";
import * as ChildProcessSpawner from "effect/unstable/process/ChildProcessSpawner";

// ==============================
// Definitions
// ==============================

export {
  ExternalLauncherError,
  ExternalLauncherBrowserSpawnError,
  ExternalLauncherCommandNotFoundError,
  ExternalLauncherEditorSpawnError,
  ExternalLauncherUnknownEditorError,
  ExternalLauncherUnsupportedEditorError,
  isExternalLauncherError,
} from "@t3tools/contracts";
export type { LaunchEditorInput };
interface EditorLaunch {
  readonly editor: EditorId;
  readonly label: string;
  readonly target: string;
  readonly command: string;
  readonly args: ReadonlyArray<string>;
  readonly requiresTerminal: boolean;
  readonly workingDirectory?: string;
}

interface ProcessLaunch {
  readonly command: string;
  readonly args: ReadonlyArray<string>;
  readonly options: ChildProcess.CommandOptions;
}

interface TargetPathAndPosition {
  readonly path: string;
  readonly line: string;
  readonly column: Option.Option<string>;
}

const TARGET_WITH_POSITION_PATTERN = /^(.*?):(\d+)(?::(\d+))?$/;
const POWERSHELL_ARGUMENTS_PREFIX = [
  "-NoProfile",
  "-NonInteractive",
  "-ExecutionPolicy",
  "Bypass",
  "-EncodedCommand",
] as const;

const DETACHED_IGNORE_STDIO_OPTIONS = {
  detached: true,
  stdin: "ignore",
  stdout: "ignore",
  stderr: "ignore",
} as const satisfies ChildProcess.CommandOptions;

interface TerminalLauncher {
  readonly command: string;
  readonly argsBeforeCommand: ReadonlyArray<string>;
  readonly kind: "argv" | "macos-terminal" | "windows-terminal";
}

const LINUX_TERMINAL_LAUNCHERS = [
  { commands: ["x-terminal-emulator"], argsBeforeCommand: ["-e"] },
  { commands: ["ghostty"], argsBeforeCommand: ["-e"] },
  { commands: ["alacritty"], argsBeforeCommand: ["-e"] },
  { commands: ["kitty"], argsBeforeCommand: [] },
  { commands: ["wezterm"], argsBeforeCommand: ["start", "--"] },
  { commands: ["footclient", "foot"], argsBeforeCommand: [] },
  { commands: ["kgx", "gnome-terminal"], argsBeforeCommand: ["--"] },
  { commands: ["konsole"], argsBeforeCommand: ["-e"] },
  { commands: ["xterm"], argsBeforeCommand: ["-e"] },
] as const;

const compactEnv = (input: Record<string, Option.Option<string>>): NodeJS.ProcessEnv =>
  Object.fromEntries(
    Object.entries(input).flatMap(([key, value]) =>
      Option.match(value, {
        onNone: () => [],
        onSome: (resolved) => [[key, resolved]],
      }),
    ),
  );

const BrowserLaunchEnvConfig = Config.all({
  SYSTEMROOT: Config.string("SYSTEMROOT").pipe(Config.option),
  windir: Config.string("windir").pipe(Config.option),
  WSL_DISTRO_NAME: Config.string("WSL_DISTRO_NAME").pipe(Config.option),
  WSL_INTEROP: Config.string("WSL_INTEROP").pipe(Config.option),
  SSH_CONNECTION: Config.string("SSH_CONNECTION").pipe(Config.option),
  SSH_TTY: Config.string("SSH_TTY").pipe(Config.option),
  container: Config.string("container").pipe(Config.option),
}).pipe(Config.map(compactEnv));

const CommandLookupEnvConfig = Config.all({
  PATH: Config.string("PATH").pipe(Config.option),
  Path: Config.string("Path").pipe(Config.option),
  path: Config.string("path").pipe(Config.option),
  PATHEXT: Config.string("PATHEXT").pipe(Config.option),
}).pipe(Config.map(compactEnv));

const readBrowserLaunchEnv = BrowserLaunchEnvConfig.pipe(Effect.orElseSucceed(() => ({})));
const readCommandLookupEnv = CommandLookupEnvConfig.pipe(Effect.orElseSucceed(() => ({})));

function parseTargetPathAndPosition(target: string): Option.Option<TargetPathAndPosition> {
  const match = TARGET_WITH_POSITION_PATTERN.exec(target);
  if (!match?.[1] || !match[2]) {
    return Option.none();
  }

  return Option.some({
    path: match[1],
    line: match[2],
    column: Option.fromUndefinedOr(match[3]),
  });
}

function resolveCommandEditorArgs(
  editor: (typeof EDITORS)[number],
  target: string,
): ReadonlyArray<string> {
  const parsedTarget = parseTargetPathAndPosition(target);

  switch (editor.launchStyle) {
    case "direct-path":
      return [target];
    case "goto":
      return Option.isSome(parsedTarget) ? ["--goto", target] : [target];
    case "line-column":
      return Option.match(parsedTarget, {
        onNone: () => [target],
        onSome: ({ path, line, column }) => [
          "--line",
          line,
          ...Option.match(column, {
            onNone: () => [],
            onSome: (value) => ["--column", value],
          }),
          path,
        ],
      });
    case "vim":
      return Option.match(parsedTarget, {
        onNone: () => [target],
        onSome: ({ path, line, column }) => [
          Option.match(column, {
            onNone: () => `+${line}`,
            onSome: (value) => `+call cursor(${line},${value})`,
          }),
          path,
        ],
      });
  }
}

function resolveEditorArgs(
  editor: (typeof EDITORS)[number],
  target: string,
): ReadonlyArray<string> {
  const baseArgs = "baseArgs" in editor ? editor.baseArgs : [];
  return [...baseArgs, ...resolveCommandEditorArgs(editor, target)];
}

const resolveAvailableCommand = Effect.fn("externalLauncher.resolveAvailableCommand")(function* (
  commands: ReadonlyArray<string>,
  env: NodeJS.ProcessEnv,
): Effect.fn.Return<Option.Option<string>, never, FileSystem.FileSystem | Path.Path> {
  for (const command of commands) {
    if (yield* isCommandAvailable(command, { env })) {
      return Option.some(command);
    }
  }
  return Option.none();
});

const resolveTerminalLauncher = Effect.fn("externalLauncher.resolveTerminalLauncher")(function* (
  platform: NodeJS.Platform,
  env: NodeJS.ProcessEnv,
): Effect.fn.Return<Option.Option<TerminalLauncher>, never, FileSystem.FileSystem | Path.Path> {
  if (platform === "darwin") {
    const command = yield* resolveAvailableCommand(["osascript"], env);
    return Option.map(command, (resolved) => ({
      command: resolved,
      argsBeforeCommand: [],
      kind: "macos-terminal" as const,
    }));
  }

  if (platform === "win32") {
    const command = yield* resolveAvailableCommand(["wt.exe", "wt"], env);
    return Option.map(command, (resolved) => ({
      command: resolved,
      argsBeforeCommand: [],
      kind: "windows-terminal" as const,
    }));
  }

  for (const candidate of LINUX_TERMINAL_LAUNCHERS) {
    const command = yield* resolveAvailableCommand(candidate.commands, env);
    if (Option.isSome(command)) {
      return Option.some({
        command: command.value,
        argsBeforeCommand: candidate.argsBeforeCommand,
        kind: "argv",
      });
    }
  }

  return Option.none();
});

function shellQuotePosix(value: string): string {
  return `'${value.replaceAll("'", `'\\''`)}'`;
}

function appleScriptString(value: string): string {
  return `"${value.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
}

function macosTerminalScript(
  editorCommand: string,
  editorArgs: ReadonlyArray<string>,
  workingDirectory?: string,
): string {
  const editorShellCommand = `exec ${[editorCommand, ...editorArgs].map(shellQuotePosix).join(" ")}`;
  const shellCommand =
    workingDirectory === undefined
      ? editorShellCommand
      : `cd -- ${shellQuotePosix(workingDirectory)} && ${editorShellCommand}`;
  return [
    'tell application "Terminal"',
    "activate",
    `do script ${appleScriptString(shellCommand)}`,
    "end tell",
  ].join("\n");
}

function terminalLauncherHint(platform: NodeJS.Platform): string {
  if (platform === "darwin") return "osascript";
  if (platform === "win32") return "wt.exe";
  return "terminal-emulator";
}

function encodeUtf16LeBase64(input: string): string {
  const bytes = new Uint8Array(input.length * 2);
  for (let index = 0; index < input.length; index += 1) {
    const code = input.charCodeAt(index);
    bytes[index * 2] = code & 0xff;
    bytes[index * 2 + 1] = code >>> 8;
  }
  return Encoding.encodeBase64(bytes);
}

function escapePowerShellStringLiteral(input: string): string {
  return `'${input.replaceAll("'", "''")}'`;
}

function resolvePowerShellPath(env: NodeJS.ProcessEnv = {}): string {
  return `${env.SYSTEMROOT || env.windir || String.raw`C:\Windows`}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe`;
}

function resolveWslPowerShellPath(): string {
  return "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe";
}

function shouldUseWindowsBrowserFromWsl(
  platform: NodeJS.Platform,
  env: NodeJS.ProcessEnv = {},
): boolean {
  return (
    platform === "linux" &&
    (env.WSL_DISTRO_NAME !== undefined || env.WSL_INTEROP !== undefined) &&
    env.SSH_CONNECTION === undefined &&
    env.SSH_TTY === undefined &&
    env.container === undefined
  );
}

function resolveWindowsBrowserLaunch(target: string, command: string): ProcessLaunch {
  const encodedCommand = encodeUtf16LeBase64(
    `$ProgressPreference = 'SilentlyContinue'; Start ${escapePowerShellStringLiteral(target)}`,
  );
  return {
    command,
    args: [...POWERSHELL_ARGUMENTS_PREFIX, encodedCommand],
    options: {
      detached: true,
      shell: false,
      stdin: "ignore",
      stdout: "ignore",
      stderr: "ignore",
    },
  };
}

function fileManagerCommandForPlatform(platform: NodeJS.Platform): string {
  switch (platform) {
    case "darwin":
      return "open";
    case "win32":
      return "explorer";
    default:
      return "xdg-open";
  }
}

function buildBrowserLaunch(
  target: string,
  platform: NodeJS.Platform,
  env: NodeJS.ProcessEnv = {},
): ProcessLaunch {
  if (platform === "darwin") {
    return {
      command: "open",
      args: [target],
      options: DETACHED_IGNORE_STDIO_OPTIONS,
    };
  }

  if (platform === "win32") {
    return resolveWindowsBrowserLaunch(target, resolvePowerShellPath(env));
  }

  if (shouldUseWindowsBrowserFromWsl(platform, env)) {
    return resolveWindowsBrowserLaunch(target, resolveWslPowerShellPath());
  }

  return {
    command: "xdg-open",
    args: [target],
    options: DETACHED_IGNORE_STDIO_OPTIONS,
  };
}

const buildAvailableEditors = Effect.fn("externalLauncher.buildAvailableEditors")(function* (
  platform: NodeJS.Platform,
  env: NodeJS.ProcessEnv,
): Effect.fn.Return<ReadonlyArray<EditorId>, never, FileSystem.FileSystem | Path.Path> {
  const available: EditorId[] = [];
  let terminalLauncher: Option.Option<TerminalLauncher> | undefined;

  for (const editor of EDITORS) {
    if (editor.commands === null) {
      const command = fileManagerCommandForPlatform(platform);
      if (yield* isCommandAvailable(command, { env })) {
        available.push(editor.id);
      }
      continue;
    }

    const command = yield* resolveAvailableCommand(editor.commands, env);
    if (Option.isNone(command)) continue;

    const requiresTerminal = "requiresTerminal" in editor && editor.requiresTerminal === true;
    if (requiresTerminal) {
      terminalLauncher ??= yield* resolveTerminalLauncher(platform, env);
      if (Option.isNone(terminalLauncher)) continue;
    }

    available.push(editor.id);
  }

  return available;
});

const resolveBrowserLaunch = Effect.fn("externalLauncher.resolveBrowserLaunch")(function* (
  target: string,
) {
  const platform = yield* HostProcessPlatform;
  const env = yield* readBrowserLaunchEnv;
  return buildBrowserLaunch(target, platform, env);
});

const resolveAvailableEditors = Effect.fn("externalLauncher.resolveAvailableEditors")(function* () {
  const platform = yield* HostProcessPlatform;
  const env = yield* readCommandLookupEnv;
  return yield* buildAvailableEditors(platform, env);
});

// Editor discovery walks PATH for every known editor and runs for every
// client connect (the server config embeds the available editors). Memoize
// the discovered set for a bounded window so repeat connects skip even the
// per-command cache lookups in @t3tools/shared/shell.
//
// This deliberately does not use `Effect.cachedWithTTL`: that memoizes the
// first caller's Exit whatever it is, including an interrupt. Callers run this
// on the connection fiber under a timeout (`resolveAvailableEditorsForConfig`),
// so one client disconnecting mid-scan would cache the interrupt and replay it
// to every later connect for the whole TTL, breaking `server.getConfig`
// permanently. Storing only on success means an interrupted scan leaves the
// cache untouched and the next connect simply rescans.
// Expiry uses the monotonic clock (Clock.currentTimeNanos), matching the
// command-resolution cache in @t3tools/shared/shell, so a backward wall-clock
// adjustment cannot keep an expired entry alive.
const EDITOR_DISCOVERY_CACHE_TTL_NANOS = 60_000_000_000n;

interface EditorDiscoveryCacheEntry {
  readonly editors: ReadonlyArray<EditorId>;
  readonly expiresAtNanos: bigint;
}

/**
 * ExternalLauncher - Service tag for browser/editor launch operations.
 */
export class ExternalLauncher extends Context.Service<
  ExternalLauncher,
  {
    readonly resolveAvailableEditors: () => Effect.Effect<ReadonlyArray<EditorId>>;
    /** Launch a URL target in the default browser. */
    readonly launchBrowser: (target: string) => Effect.Effect<void, ExternalLauncherError>;
    /**
     * Launch a workspace path in a selected editor integration.
     *
     * Launches the editor as a detached process so server startup is not blocked.
     */
    readonly launchEditor: (input: LaunchEditorInput) => Effect.Effect<void, ExternalLauncherError>;
  }
>()("t3/process/externalLauncher") {}

// ==============================
// Implementations
// ==============================

const resolveEditorLaunch = Effect.fn("resolveEditorLaunch")(function* (
  input: LaunchEditorInput,
): Effect.fn.Return<EditorLaunch, ExternalLauncherError, FileSystem.FileSystem | Path.Path> {
  const platform = yield* HostProcessPlatform;
  const env = yield* readCommandLookupEnv;
  yield* Effect.annotateCurrentSpan({
    "externalLauncher.editor": input.editor,
    "externalLauncher.cwd": input.cwd,
    "externalLauncher.platform": platform,
  });
  const editorDef = EDITORS.find((editor) => editor.id === input.editor);
  if (!editorDef) {
    return yield* new ExternalLauncherUnknownEditorError({ editor: input.editor });
  }

  if (editorDef.commands) {
    const command = Option.getOrElse(
      yield* resolveAvailableCommand(editorDef.commands, env),
      () => editorDef.commands[0],
    );
    const opensNeovimProject =
      editorDef.id === "neovim" &&
      (yield* Effect.gen(function* () {
        const fileSystem = yield* FileSystem.FileSystem;
        const info = yield* fileSystem.stat(input.cwd).pipe(Effect.orElseSucceed(() => null));
        return info?.type === "Directory";
      }));
    return {
      editor: editorDef.id,
      label: editorDef.label,
      target: input.cwd,
      command,
      args: opensNeovimProject ? ["."] : resolveEditorArgs(editorDef, input.cwd),
      requiresTerminal: "requiresTerminal" in editorDef && editorDef.requiresTerminal === true,
      ...(opensNeovimProject ? { workingDirectory: input.cwd } : {}),
    };
  }

  if (editorDef.id !== "file-manager") {
    return yield* new ExternalLauncherUnsupportedEditorError({ editor: input.editor });
  }

  return {
    editor: editorDef.id,
    label: editorDef.label,
    target: input.cwd,
    command: fileManagerCommandForPlatform(platform),
    args: [input.cwd],
    requiresTerminal: false,
  };
});

const launchAndUnref = Effect.fn("externalLauncher.launchAndUnref")(function* (
  launch: ProcessLaunch,
  onError: (cause: unknown) => ExternalLauncherError,
): Effect.fn.Return<void, ExternalLauncherError, ChildProcessSpawner.ChildProcessSpawner> {
  const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;
  const command = ChildProcess.make(launch.command, launch.args, launch.options);

  yield* spawner.spawn(command).pipe(
    Effect.flatMap((handle) => handle.unref),
    Effect.asVoid,
    Effect.scoped,
    Effect.mapError(onError),
  );
});

const launchBrowser = Effect.fn("externalLauncher.launchBrowser")(function* (
  target: string,
): Effect.fn.Return<void, ExternalLauncherError, ChildProcessSpawner.ChildProcessSpawner> {
  const launch = yield* resolveBrowserLaunch(target);
  return yield* launchAndUnref(
    launch,
    (cause) =>
      new ExternalLauncherBrowserSpawnError({
        target,
        command: launch.command,
        args: launch.args,
        cause,
      }),
  );
});

const launchEditorProcess = Effect.fn("externalLauncher.launchEditorProcess")(function* (
  launch: EditorLaunch,
): Effect.fn.Return<
  void,
  ExternalLauncherError,
  ChildProcessSpawner.ChildProcessSpawner | FileSystem.FileSystem | Path.Path
> {
  const platform = yield* HostProcessPlatform;
  const env = yield* readCommandLookupEnv;
  if (!(yield* isCommandAvailable(launch.command, { env }))) {
    return yield* new ExternalLauncherCommandNotFoundError({
      editor: launch.editor,
      command: launch.command,
    });
  }

  let command = launch.command;
  let args = launch.args;
  if (launch.requiresTerminal) {
    const editorCommand = yield* resolveCommandPath(launch.command, { env }).pipe(
      Effect.mapError(
        () =>
          new ExternalLauncherCommandNotFoundError({
            editor: launch.editor,
            command: launch.command,
          }),
      ),
    );
    const terminal = yield* resolveTerminalLauncher(platform, env);
    if (Option.isNone(terminal)) {
      return yield* new ExternalLauncherCommandNotFoundError({
        editor: launch.editor,
        command: terminalLauncherHint(platform),
      });
    }
    const terminalCommand = yield* resolveCommandPath(terminal.value.command, { env }).pipe(
      Effect.mapError(
        () =>
          new ExternalLauncherCommandNotFoundError({
            editor: launch.editor,
            command: terminal.value.command,
          }),
      ),
    );
    command = terminalCommand;
    switch (terminal.value.kind) {
      case "macos-terminal":
        args = ["-e", macosTerminalScript(editorCommand, launch.args, launch.workingDirectory)];
        break;
      case "windows-terminal":
        args = [
          "new-tab",
          "--title",
          launch.label,
          ...(launch.workingDirectory === undefined
            ? []
            : ["--startingDirectory", launch.workingDirectory]),
          editorCommand,
          ...launch.args,
        ];
        break;
      case "argv":
        args = [...terminal.value.argsBeforeCommand, editorCommand, ...launch.args];
        break;
    }
  }

  const spawnCommand = yield* resolveSpawnCommand(command, args, { env });
  yield* launchAndUnref(
    {
      command: spawnCommand.command,
      args: spawnCommand.args,
      options: {
        detached: true,
        shell: spawnCommand.shell,
        stdin: "ignore",
        stdout: "ignore",
        stderr: "ignore",
        ...(launch.workingDirectory === undefined ? {} : { cwd: launch.workingDirectory }),
      },
    },
    (cause) =>
      new ExternalLauncherEditorSpawnError({
        editor: launch.editor,
        target: launch.target,
        command: spawnCommand.command,
        args: spawnCommand.args,
        cause,
      }),
  );
});

export const make = Effect.gen(function* () {
  const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;
  const fileSystem = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;

  const provideCommandResolutionServices = <A, E, R>(
    effect: Effect.Effect<A, E, R | FileSystem.FileSystem | Path.Path>,
  ) =>
    effect.pipe(
      Effect.provideService(FileSystem.FileSystem, fileSystem),
      Effect.provideService(Path.Path, path),
    );

  const editorDiscoveryCache = yield* Ref.make<Option.Option<EditorDiscoveryCacheEntry>>(
    Option.none(),
  );
  const cachedAvailableEditors = Effect.gen(function* () {
    const nowNanos = yield* Clock.currentTimeNanos;
    const entry = yield* Ref.get(editorDiscoveryCache);
    if (Option.isSome(entry) && entry.value.expiresAtNanos > nowNanos) {
      return entry.value.editors;
    }
    const editors = yield* provideCommandResolutionServices(resolveAvailableEditors());
    yield* Ref.set(
      editorDiscoveryCache,
      Option.some({
        editors,
        expiresAtNanos: nowNanos + EDITOR_DISCOVERY_CACHE_TTL_NANOS,
      }),
    );
    return editors;
  });

  return ExternalLauncher.of({
    resolveAvailableEditors: () => cachedAvailableEditors,
    launchBrowser: (target) =>
      launchBrowser(target).pipe(
        Effect.provideService(ChildProcessSpawner.ChildProcessSpawner, spawner),
      ),
    launchEditor: (input) =>
      provideCommandResolutionServices(
        Effect.flatMap(resolveEditorLaunch(input), (launch) =>
          launchEditorProcess(launch).pipe(
            Effect.provideService(ChildProcessSpawner.ChildProcessSpawner, spawner),
          ),
        ),
      ),
  });
});

export const layer = Layer.effect(ExternalLauncher, make);
