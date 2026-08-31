# T3RL research lab architecture

> For maintainers. The Phase 1 backend and initial web/desktop client are implemented; later
> research phases remain incremental work.

Status: Phase 1 backend and initial RL Lab UI implemented; validation and later research phases remain

Delivery plan: [T3RL phased development plan](./rl-lab-roadmap.md)

## Purpose

T3RL turns T3 Code into an agent-assisted workspace for reinforcement learning research. A researcher
should be able to change an environment or algorithm, launch a reproducible experiment, observe its
behavior, compare the result with earlier runs, and ask an agent to investigate the evidence without
leaving the project.

The first implementation is deliberately narrow: train PPO on `CartPole-v1` with Gymnasium and
Stable-Baselines3, stream bounded progress metrics, persist the resolved run manifest, and retain an
evaluation artifact. This vertical slice proves the execution and data boundaries before more
algorithms, environments, or distributed backends are introduced.

## Product boundary

T3RL is a research control plane, not a new reinforcement learning framework. Algorithms and
environments remain ordinary Python code. The lab owns experiment orchestration, reproducibility,
observation, comparison, and agent-facing research context.

The interface must not hide behavior that changes a result. Every UI choice resolves to an explicit,
exportable configuration, and a run records the defaults ultimately used by the backend.

## Why T3 Code is the foundation

T3 Code already provides the control surface around which T3RL's research workflow can be built:

- a server execution boundary that owns processes, terminals, Git, and filesystem access;
- authenticated, typed RPC shared by local and remote clients;
- durable project and thread state with explicit command and event semantics;
- web, desktop, and mobile surfaces backed by a shared client runtime;
- remote access that lets a client disconnect without moving execution into the browser;
- agent sessions that can inspect and edit the same research workspace;
- process resource telemetry and established performance constraints.

Those capabilities are valuable because T3RL is agent-assisted research infrastructure. If the goal
were only a standalone chart dashboard, the cost of this architecture would not be justified. The
foundation is a good fit when experiments, source changes, agents, execution, evidence, and remote
control form one workflow.

### Goals

- Preserve Python and versioned experiment definitions as the source of truth.
- Make a run reproducible enough to inspect, share, and repeat.
- Keep training on the server-side execution environment so local and remote projects behave the
  same way.
- Let researchers use custom environments, algorithms, callbacks, and metrics as the product grows.
- Bound metric traffic, memory, and rendering cost independently of training speed.
- Give agents structured access to configurations, code changes, metrics, logs, and artifacts.
- Degrade explicitly when Python, an accelerator, or a runner backend is unavailable.

### Non-goals for the first vertical slice

- Distributed or multi-node training.
- Hyperparameter sweeps and multi-seed aggregation.
- Live video streaming from every environment step.
- Editing arbitrary reward functions through form controls.
- Installing Python, GPU drivers, or project dependencies automatically.
- Supporting every Gymnasium environment or Stable-Baselines3 algorithm.
- Resuming a process after the T3 server itself restarts.
- A complete mobile experiment-authoring interface.

## Design principles

### The server owns execution

The browser never starts Python or reads artifacts directly from the host. The T3 server validates a
request, resolves the project and runner, supervises the child process, records lifecycle changes,
and exposes bounded data through authenticated RPC. This preserves T3's local, remote, relay, and
desktop execution model.

### Python stays replaceable

The first control worker uses Gymnasium and Stable-Baselines3, and the first LLM post-training
worker uses TRL/GRPO with a deterministic verifier and held-out before/after evaluation. The server
talks to both through the same versioned process protocol rather than importing framework-specific
concepts throughout the application. Axolotl, Accelerate launchers, FSDP, DeepSpeed, and vLLM stay
behind adapters or sidecar ownership; they do not add framework-shaped lifecycle states to clients.
See [Framework adapters](./rl-framework-adapters.md) for the compatibility boundary and status.

### Durable state is sparse; telemetry is bounded

Run lifecycle transitions are durable domain facts. Training samples are high-volume observations
and do not belong in the orchestration event log. The server persists a compact metric series with
the run artifacts and publishes batches only to active subscribers.

### Reproducibility is a default

A run receives an immutable resolved manifest before training starts. The manifest records the seed,
algorithm, environment, effective hyperparameters, source revision, dirty-worktree evidence, Python
environment, runner version, and relevant hardware. A run never silently changes when an experiment
definition is edited later.

### The UI explains state; it does not invent it

Clients render the server projection and subscription state. They do not infer that a run completed
because metrics stopped arriving, and they do not maintain a second lifecycle state machine.

## Runtime topology

```text
apps/web and apps/mobile
  RL Lab view + shared run state
              │
              │ authenticated Effect RPC
              ▼
apps/server
  RL run lifecycle + process supervision + artifact access
              │
              │ argv + versioned NDJSON over stdout
              ▼
Selected Python worker
  Stable-Baselines3 control adapter or post-training framework adapter
              │
              ├─ metrics
              ├─ optional final model
              ├─ before/after evaluation and behavior replay
              └─ resolved environment metadata
```

Desktop receives the web implementation through its existing wrapper. Mobile initially supports a
read-only run list and run status; authoring can follow after the shared runtime model is stable.

## Domain language

### Experiment

A versionable definition of an intended investigation: environment, algorithm, parameters,
evaluation policy, and requested seeds. Editing an experiment does not mutate prior runs.

### Run

One concrete execution of a resolved experiment definition for one seed. A run has an immutable ID,
manifest, lifecycle, metric series, logs, and zero or more artifacts.

### Evaluation

A policy assessment performed separately from training collection. Evaluation metrics and recordings
are labeled as evaluation output so they are not confused with training rewards.

### Artifact

A file produced by a run, such as a checkpoint, resolved manifest, log, evaluation video, or summary.
Artifacts have a media type, size, and logical kind. Clients receive artifacts through authorized
server endpoints rather than arbitrary filesystem paths. Content hashes remain a later evidence
extension.

### Runner

An adapter that turns a resolved manifest into a training process and translates native output into
the T3RL worker protocol. The first runner is `stable-baselines3`; runner identity and version are
part of the manifest.

## Run lifecycle

```text
requested ──▶ preparing ──▶ running ──▶ cancelling
                                │            │
                                ▼            ▼
                          completed      cancelled
                             failed

any active state ──(server restart)──▶ interrupted
any active state ──────(error)───────▶ failed
```

- `requested`: durable intent exists, but no process has started.
- `preparing`: the server is resolving capabilities, paths, and the immutable manifest.
- `running`: the worker handshake succeeded and training is active.
- `cancelling`: cancellation was accepted and process shutdown is in progress.
- `completed`, `failed`, `cancelled`, and `interrupted`: terminal states with an explicit reason or
  result.

`interrupted` is deliberately distinct from `failed`. A run whose server died has an unknown outcome,
and reporting it as a failure would fabricate a scientific claim about a trajectory nobody observed.

The first terminal fact wins: a worker that finishes on its own before a pending cancellation reaches
it records the result it reported, not `cancelled`.

Each start carries a client request id. Retrying the same request for a project returns the original
run id without spawning again; a deliberate rerun uses a new request id and produces a new run.
Cancelling a terminal run succeeds without effect. A disconnected client does not affect the
process. Process adoption and checkpoint resume after a restart are later capabilities.

## Contracts and persistence

Client/server schemas belong in `packages/contracts`. Shared connection and cached domain state
belong in `packages/client-runtime`. Visual state stays in the relevant client.

The implemented RPC surface is intentionally small:

- `rl.capabilities`: report available runners, actionable setup failures, and bundled experiments.
- `rl.listRuns`: return durable run summaries for one project.
- `rl.getRun`: return the manifest, bounded metrics, and artifact metadata for one run.
- `rl.startRun`: request a run from an experiment, seed, and client request id.
- `rl.cancelRun`: request cancellation of a non-terminal run.
- `rl.subscribeRun`: send a durable snapshot followed by live lifecycle, manifest, metric, and
  artifact events.

Artifact bytes use the existing signed asset URL boundary.

Names may change to match the concrete Effect RPC grouping, but the capability boundaries should not
expand during the first slice.

Run intent and lifecycle transitions use an RL-owned in-place projection rather than the
orchestration event log. Metric points must not become orchestration events. A compact run
projection supports lists and reconnection; metrics and artifact metadata use dedicated tables.

### Experiment definitions

Phase 1 definitions are bundled, version-controlled JSON:

```text
python/t3rl_worker/experiments/
  cartpole-ppo.json
```

Project-owned definitions are a later extension. When added, UI edits must remain ordinary,
reviewable workspace changes.

### Run artifacts

Generated artifacts default to environment-local T3 state rather than the Git workspace:

```text
<stateDir>/rl/<run-id>/
  manifest.json
  worker.log
  summary.json
  evaluation.json
  replay.json
  model.zip
```

This avoids adding large binary output to Git and prevents run output from contaminating thread
checkpoints. Exporting selected artifacts into the workspace is an explicit future action.

The Phase 1 immutable manifest contains:

- experiment id and resolved configuration;
- runner id and version plus worker protocol version;
- random seed and runner-resolved evaluation policy;
- Git commit and dirty-worktree status;
- Python executable, version, and environment fingerprint;
- instrumentation level and a host hardware summary.

Content hashes, retained dirty patches, accelerator evidence, and exported source snapshots are later
research-evidence extensions.

Secrets and environment-variable values are excluded. The manifest may record the names of declared
inputs, but never credentials or raw tokens.

## Worker protocol

The server spawns the worker with an argument vector, never a shell-composed command. Standard output
is reserved for newline-delimited JSON protocol messages; human-readable logs go to standard error
and the bounded worker log.

The worker must first emit a `hello` message containing its protocol, runner, and runner version. The
server rejects an incompatible protocol before marking the run as `running`.

Implemented message kinds are:

- `hello`
- `manifest`
- `metrics`
- `artifact`
- `error`
- `done`

The initial worker receives its immutable inputs through an argument vector. Cancellation is a
process-tree signal, not a protocol message. Messages are validated at the server boundary; unknown
message kinds and invalid required fields fail the run with a protocol error.

## Metrics and rendering

Training can produce more samples than a client can render or a remote connection should carry. The
worker therefore aggregates before emission and the server enforces an additional bound.

For the first slice:

- publish at most two metric batches per second;
- separate training and evaluation namespaces;
- retain raw episode summaries on disk, not individual environment steps;
- bound each batch by point count and encoded byte size;
- downsample historical series before returning them to a chart;
- coalesce server-side metric bursts independently of subscriber count;
- record evaluation video after a configured interval or at completion, not during every training
  step.

The UI updates charts from batches and avoids continuously repainting animation. A stale-data marker
is derived from server timestamps, while lifecycle status remains authoritative.

## Algorithm observability and debugging

Visualizations are research instruments, not decoration. They should help a researcher locate a
learning failure, inspect the evidence that produced it, relate it to code and configuration, and run
a controlled correction or ablation.

### Common diagnostics

Every runner should expose a small framework-independent core when the underlying algorithm provides
the data:

- episodic return, episode length, success rate, and termination reason;
- training and evaluation sample counts on separate axes;
- optimizer step, learning rate, throughput, and elapsed wall time;
- observation and action distributions with explicit sampling metadata;
- gradient norms, parameter norms, clipping, and non-finite value detection;
- environment-state coverage or visitation summaries where a representation is available;
- system utilization correlated with training phases without treating utilization as a learning
  metric.

### Algorithm-specific diagnostics

Runner adapters may declare typed metric groups and visualization hints without putting
framework-specific fields into the core run lifecycle.

- Policy-gradient methods may expose actor loss, critic loss, entropy, approximate KL divergence,
  clipping fraction, explained variance, return estimates, and advantage distributions.
- Value-based methods may expose Q-value and target distributions, temporal-difference error,
  exploration schedule, target-network updates, and replay-buffer age, priority, and coverage.
- Off-policy actor-critic methods may expose actor and critic losses, Q estimates, target values,
  policy temperature, and replay sampling diagnostics.
- Custom algorithms may publish namespaced scalar, histogram, image, table, graph, and sampled
  trajectory data through a versioned extension schema.

Absence is explicit: a chart says that a metric is unsupported or was not captured rather than
showing a misleading zero.

### Evidence navigation

A diagnostic view should preserve this path:

```text
chart anomaly
      ↓
metric window and sampled episodes
      ↓
checkpoint, manifest, and source snapshot
      ↓
agent or researcher diagnosis
      ↓
proposed change or ablation
      ↓
comparable run
```

Replay views align observations, chosen actions, action probabilities or values, rewards,
terminations, and selected model estimates on one timeline. A researcher can move from a point or
range in a chart to the corresponding sampled trajectories and retained checkpoint when those
artifacts were captured.

Agent-generated diagnoses reference stable run IDs, metric ranges, artifacts, and source evidence.
The agent may prepare a code or configuration change, but the change remains an ordinary reviewable
workspace diff and the follow-up run remains permission-aware.

### Instrumentation levels

Instrumentation is selected explicitly and recorded in the resolved manifest:

- `minimal`: lifecycle, evaluation, episodic summaries, and lightweight optimizer metrics;
- `standard`: algorithm-specific scalars, bounded distributions, and periodic evaluation replay;
- `deep`: selected activations, gradients, replay-buffer samples, state maps, and additional model
  snapshots.

`standard` is the PPO first-slice default. `deep` is opt-in because collection, serialization, storage,
and synchronization can change training throughput or behavior. Comparisons show instrumentation
differences alongside code and configuration differences.

### First-slice visualization scope

The PPO `CartPole-v1` slice includes return, episode length, evaluation return, actor/policy loss,
value loss, entropy, approximate KL divergence when reported by the runner, elapsed time, and a final
evaluation replay. It also surfaces worker errors and non-finite metrics.

The web/desktop client also provides two manifest-backed inspection surfaces without widening the
wire contract. The data explorer discovers arbitrary metric keys from the bounded run projection
and renders only the selected mode. The algorithm visualizer maps the resolved algorithm to a
declarative stage graph and labels its playback as conceptual; it never infers a live optimizer
phase from metric arrival. Both expose `Visual`, `Data`, and `Source` modes so the rendered claim can
be checked against its retained inputs.

Replay-buffer inspection, state visitation maps, gradient and activation distributions, checkpoint
alignment, custom panels, and cross-run diagnostic overlays follow after the metric and artifact
contracts have proven stable.

## Agent integration

Agent assistance operates on structured evidence rather than screenshots or copied chart text. The
product-native `t3-code` MCP server exposes the same project-scoped toolkit to Codex, Claude,
Cursor, Grok, and OpenCode. It allows an agent to:

- inspect experiment definitions and resolved manifests;
- query bounded metric summaries and selected time ranges;
- read run logs and failure reasons;
- compare two or more run summaries;
- locate the source revision and dirty-worktree evidence recorded by a run;
- propose a new experiment definition or ablation as an ordinary workspace edit;
- start a run only through an explicit, permission-aware action.

The concrete tools are `rl_capabilities`, `rl_list_runs`, `rl_get_run`, `rl_query_metrics`,
`rl_compare_runs`, `rl_read_artifact`, `rl_start_run`, and `rl_cancel_run`. The MCP credential derives
the project from its thread; callers cannot supply a different project ID, and foreign run IDs are
reported as missing. Metric queries, summaries, comparisons, and textual artifact reads are bounded.
Binary models are never copied into model context.

RL access is attached by default and is independent of agent browser access. Disabling
`enableAgentBrowserAccess` removes only the `preview` capability; it does not remove the `rl`
capability. Starting and cancelling runs are annotated as destructive permission-aware operations,
while every evidence tool is read-only, idempotent, and closed-world.

### Right-panel research workbenches

The web/desktop right-panel selector exposes three project-scoped workbenches without changing the
ordinary Agents surface:

- `experiments` embeds the existing RL Lab beside a conversation;
- `specialists` edits reusable scientific role instructions;
- `autoresearch` prepares one evidence-backed, review-gated research iteration.

The specialist and Autoresearch slice persists one `ResearchWorkspaceDocument` at
`.t3rl/research.json` through the existing authorized project file RPC. This makes configuration
branch-versioned and remote-ready without adding a second settings database. Prompts contain the
effective specialist instructions visibly, so provider adapters do not need to pretend that Codex,
Claude, Cursor, Grok, and OpenCode share one raw system-prompt primitive.

Version 2 adds a `target` with three orthogonal fields: stable algorithm ID, task topology, and
reward or feedback source. The client-runtime algorithm catalog derives family, learning mode,
action space, required evidence, suggested adapters, and honest native-execution status. Version 1
files migrate in memory and acquire the new default specialists without discarding edited profiles.
See [RL algorithm coverage](./rl-algorithm-coverage.md) for the taxonomy and adapter boundary.

When a baseline is selected, the client derives a bounded evidence block from the authoritative run
projection: run identity and lifecycle, resolved manifest evidence, first/last/min/max scalar
summaries with non-finite counts, and artifact identities. Project-authored evidence is delimited as
untrusted data in the prepared prompt. The prepared first turn must stop before edits or training and
request explicit approval.

This is an assisted single-iteration seam, not the Phase 2B autonomous controller. Server-enforced
multi-iteration budgets, durable hypothesis-to-outcome records, and automatic authorized run
execution remain Phase 2B work.

## Security and permissions

Starting training executes project-controlled code and must be treated as a side effect comparable to
starting a terminal command, not as a read-only chart action.

- RPC methods use the least-privileged existing scope or a dedicated scope if existing semantics do
  not fit.
- A run is always bound to one project and one execution environment.
- Project-relative paths are resolved and checked by the server.
- Artifact access uses IDs and server-side lookup, never a client-supplied absolute path.
- The runner receives an explicit environment allowlist; credentials are not copied wholesale.
- Capability detection is read-only and never installs packages.
- Cancellation targets only the process identity captured when that run was spawned.
- Remote clients follow the same authorization checks as local clients.

## Failure behavior

Failures should leave a useful scientific record. A failed run retains its resolved manifest,
available metrics, bounded logs, worker exit information, and any completed artifacts.

The server distinguishes at least:

- capability failure before spawn;
- invalid experiment configuration;
- worker protocol incompatibility;
- training exception reported by the runner;
- unexpected worker exit;
- user cancellation;
- server interruption.

Errors shown to users contain an actionable summary and a stable machine-readable code. Raw stack
traces remain available in the run log without being pushed repeatedly over the WebSocket.

## First vertical slice

The initial end-to-end implementation is complete when a researcher can:

1. open an existing project on web or desktop;
2. select the checked-in `CartPole-v1` PPO experiment;
3. choose an explicit seed and start a run;
4. observe authoritative lifecycle status and bounded reward metrics;
5. disconnect and reconnect without losing the run or duplicating it;
6. cancel an active run without affecting unrelated processes;
7. open the resolved manifest, summary, log, and final evaluation artifact;
8. reload T3RL and find the completed or failed run in project history.

The slice uses one worker process, one environment, one algorithm, one seed per run, CPU execution,
and local artifact storage. It validates remote browser control because execution remains on the T3
server machine.

## Next phase: open-ended learning research

After the first vertical slice validates execution, persistence, telemetry, and artifacts, the next
phase shifts the lab from running isolated benchmarks to studying agents that continuously acquire
capabilities from experience.

The goal is a closed research loop in which T3RL can propose tasks, run controlled experiments,
measure what an agent learned, and use that evidence to design the next investigation:

```text
hypothesis → task or environment → training → evaluation → diagnosis
     ▲                                                        │
     └──────────────── next experiment or curriculum ─────────┘
```

This phase adds:

- automatic generation and mutation of environments and tasks;
- curriculum management that adapts difficulty to demonstrated competence;
- continual-learning runs that preserve knowledge across a sequence of tasks;
- self-play and multi-agent experiment definitions;
- evaluations for transfer, generalization, forgetting, and newly acquired capabilities;
- reward-hacking, specification-gaming, exploration-collapse, and regression diagnostics;
- structured lineage from a research hypothesis through code, experience, checkpoints, and results;
- agent-assisted experiment and ablation design based on prior run evidence;
- explicit approval boundaries for agent-proposed code changes and experiment execution;
- runner and scheduler adapters for distributed training when an experiment requires more compute.

Automatic research actions remain ordinary, inspectable domain operations. An agent may propose an
experiment, environment change, or next curriculum stage, but T3RL records the proposal, rationale,
inputs, authorization, and resulting run. Generated tasks and evaluations are versioned artifacts so
the system cannot silently redefine success after observing a result.

The phase is successful when a researcher can define a capability objective, let the system produce
and execute a bounded curriculum, and inspect a reproducible account of which experiences produced
which measured capabilities. Scale alone is not a success criterion; the same workflow must remain
usable with deterministic fake runners and small local environments.

## Fork risks and engineering guardrails

T3RL is a long-lived fork of an actively changing application. Its implementation must protect both
research correctness and the ability to integrate upstream improvements.

### Upstream drift

Broad mechanical changes create avoidable merge conflicts and make upstream fixes harder to adopt.

- Keep an explicit upstream-tracking workflow and integrate upstream changes frequently.
- Develop T3RL changes on focused feature branches rather than mixing them with upstream sync work.
- Keep RL code behind narrow `rl` modules and shared contracts.
- Avoid a repository-wide rename, package rename, or visual rebrand during the first milestones.
- Do not modify provider adapters unless a provider-specific research capability actually requires it.
- Record any intentional divergence from upstream in an architecture decision or this document.

### Event-sourcing pressure

It is tempting to place every metric in the existing orchestration model because it is durable and
already reaches clients. Doing so would inflate the event log, projections, database, and WebSocket
traffic.

- Persist run intent and lifecycle facts in the RL run projection, outside orchestration events.
- Store metrics, logs, checkpoints, and videos in the bounded run artifact store.
- Project only the compact metadata required for lists, status, reconnection, and authorization.
- Never infer lifecycle transitions from missing telemetry.
- Load and downsample historical metrics on demand.

### Long-running process ownership

Training processes outlive individual browser connections and may spawn descendants.

- The server owns every worker for its complete lifecycle.
- Capture the exact spawned process identity and never cancel by name or search pattern.
- Isolate standard output for protocol data and retain bounded standard-error logs.
- Define shutdown escalation, orphan prevention, crash behavior, and server-restart semantics in tests.
- Apply resource limits and concurrency controls before enabling multiple simultaneous runs.
- Treat a missing heartbeat as diagnostic evidence, not by itself as proof that a process exited.

### Python and accelerator distribution

Python, native Gymnasium dependencies, CUDA, and framework wheels vary across operating systems and
hardware. Bundling them prematurely would turn the first experiment slice into a packaging project.

- Capability detection remains read-only and reports exact missing requirements.
- Never install packages, drivers, or toolchains as a side effect of opening the lab.
- Keep the worker protocol independent of a particular environment manager.
- Make the selected Python executable and environment fingerprint visible in every run.
- Support CPU execution first; introduce accelerator selection only after capability reporting is
  trustworthy.
- Test packaging separately on supported desktop and headless-server targets before claiming support.
- Fail explicitly when a runner cannot satisfy the resolved experiment instead of silently changing
  an algorithm, device, precision, or dependency version.

### Chat-first versus run-first interaction

T3 Code is organized primarily around conversations, while research also needs persistent run lists,
comparisons, artifacts, and dashboards.

- Add an RL Lab surface with its own run-centered information architecture.
- Keep chat available beside research evidence without forcing every run to be represented as a
  conversation thread.
- Let agents reference stable run and artifact IDs rather than copied chart text.
- Ensure a user can operate and inspect a run without invoking an agent.
- Preserve ordinary project, terminal, and source-control workflows when the RL Lab is unused.

### Multi-surface scope

Implementing separate authoring experiences for web and mobile before the domain stabilizes would
multiply product and testing work.

- Web is the primary implementation and desktop inherits it through the existing wrapper.
- Mobile begins as a bounded status, monitoring, and cancellation surface.
- Shared connection-backed state belongs in `packages/client-runtime`; visual layout does not.
- Every server capability must still behave correctly across local, bearer, relay, and SSH-connected
  environments.
- Client disconnection, backgrounding, or sleep must not cancel a run.

### Performance and observability overhead

Instrumentation can change the experiment it is intended to measure, and high-frequency rendering can
degrade the rest of T3 Code.

- Default to episode summaries and bounded batches rather than step-level streaming.
- Make expensive captures such as activations, gradient distributions, environment frames, and replay
  buffer samples explicit and periodic.
- Record instrumentation settings in the resolved manifest.
- Set byte, point-count, and file-size limits at the server boundary; add retention before enabling
  broad concurrent use.
- Render charts incrementally and downsample before sending large histories to a client.
- Measure worker, server, WebSocket, and renderer overhead before increasing telemetry detail.

### Research correctness

A polished visualization can still present misleading scientific evidence.

- Label training and evaluation metrics separately.
- Preserve seeds, sample counts, aggregation methods, and evaluation policy with every result.
- Never compare runs as equivalent when their environment, code snapshot, instrumentation, or resolved
  defaults differ without showing those differences.
- Use multiple seeds and uncertainty estimates before presenting comparative conclusions in later
  phases.
- Retain raw bounded summaries needed to reproduce a displayed aggregate.
- Treat agent-generated diagnoses as hypotheses linked to evidence, not authoritative conclusions.

### Security and artifact growth

Research environments execute arbitrary project code and can produce unbounded checkpoints, videos,
and logs.

- Starting or cancelling code remains an explicit permission-aware action.
- Do not expose absolute paths or inherit the server's complete environment into a worker.
- Enforce per-run and per-project artifact quotas with visible retention behavior.
- Cleanup must target resolved run IDs and remain recoverable where practical.
- Never delete a checkpoint that is the only retained source for a resumable or cited result without
  explicit confirmation.
- Keep credentials and secret values out of manifests, logs, metrics, and agent context.

### Scope control

The largest project risk is attempting the open-ended research vision before proving the execution
boundary.

- The first product proof remains one PPO `CartPole-v1` run on CPU.
- New algorithms, distributed scheduling, self-play, automatic curricula, and deep instrumentation do
  not enter that slice unless required to validate a foundational contract.
- Each later capability must reuse or intentionally revise the manifest, lifecycle, artifact, and
  authorization models.
- Prefer one complete, reconnectable, diagnosable run over a broad catalog of partially supported
  environments and algorithms.

## Implementation seams

Expected ownership follows existing repository boundaries:

```text
packages/contracts/src/rl.ts           client/server schemas and RPC contracts
packages/client-runtime/src/rl/        shared connection-backed run state
apps/server/src/rl/                    lifecycle, supervision, persistence, artifacts
apps/web/src/rl/                       desktop/web lab interface
apps/mobile/src/features/rl/           initial read-only surface
python/t3rl_worker/                    versioned worker and SB3 runner
python/t3rl_worker/experiments/        version-controlled experiment definitions
```

`apps/web/src` has no `features/` directory: it is organised flat with per-domain folders such as
`terminal/` and `browser/`, and the lab surface should follow that shape.

Exact filenames should follow adjacent code when implementation begins. The Python worker is not a
Node workspace package and must not leak Python framework types into `packages/contracts`.

## Verification strategy

- Contract tests reject malformed run requests and protocol messages.
- Pure lifecycle tests cover valid transitions and idempotent cancellation behavior; manager and
  store tests cover request-id start deduplication.
- Server tests use a deterministic fake worker for success, failure, cancellation, malformed output,
  and unexpected exit.
- Worker tests validate manifest resolution, seeding, metric aggregation, and a short CartPole smoke
  run.
- Persistence tests prove that run summaries survive server restart and active runs become explicitly
  interrupted.
- Subscription tests prove batching, byte limits, reconnection, and subscriber cleanup.
- Artifact tests cover authorization, media metadata, lexical traversal, and symlink escape attempts.
- Client tests render empty, unavailable, preparing, running, failed, cancelled, and completed states.

## Later milestones

After the open-ended learning phase is measured and stable:

1. richer statistical analysis across experiment families and long-running curricula;
2. checkpoint branching and counterfactual experiment comparison;
3. reusable environment, evaluator, and curriculum registries;
4. heterogeneous resource scheduling across local, remote, and cluster environments;
5. runner adapters for additional research frameworks and external simulators;
6. collaboration, review, and publication workflows for reproducible research bundles.

Each milestone should preserve the run manifest and lifecycle model rather than adding
backend-specific concepts to the client.

## Decisions required before implementation

The run kernel increment settled the first two. Full reasoning lives in
[the run kernel design](../superpowers/specs/2026-08-24-t3rl-run-kernel-design.md).

- **Settled (D1).** Run lifecycle lives in a manager with an in-place projection, modelled on
  `apps/server/src/terminal/Manager.ts` — not in the orchestration aggregate and not in a sibling
  event-sourced domain. A run changes state about six times and has exactly one writer, so a
  decider, projector, and reactors would be machinery with no load to carry. Adding a transition log
  later is additive.
- **Settled (D2).** `rl.*` reuses `orchestration:read` and `orchestration:operate` rather than
  introducing scope literals. Scopes are frozen per session in `auth_sessions.scopes`, so a new one
  would force every paired device to re-pair, and a client that can dispatch an orchestration command
  already runs arbitrary code on the server.
- **Settled for development.** `python/pyproject.toml` declares the optional worker environment and
  `T3RL_PYTHON` selects it; capability detection remains read-only. Desktop release distribution is
  still a later packaging decision.
- **Open.** The retention limit and explicit cleanup behavior for environment-local run artifacts.
  The kernel ships no way to delete a run, so this stays deferred rather than half-built.

These decisions affect persistence, security, and distribution. Algorithm catalogs, visual design,
and distributed execution do not need to be settled before the vertical slice begins.
