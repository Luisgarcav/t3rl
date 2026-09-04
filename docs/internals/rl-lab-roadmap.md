# T3RL phased development plan

> For maintainers. This roadmap describes delivery order and remaining gates, not calendar
> commitments.

Status: Phase 1 backend and initial RL Lab client complete; end-to-end validation and measurement remain active work

Architecture: [T3RL research lab architecture](./rl-lab.md)

## How to use this roadmap

This document is for maintainers choosing and reviewing implementation increments. The architecture
document remains the source of truth for system boundaries and intended behavior; this roadmap owns
delivery order, phase scope, and entry and exit gates. Update the architecture when a technical
boundary changes, and update this roadmap when sequencing or acceptance criteria change.

Each phase should produce a runnable demonstration, focused verification, and a transition review.
User documentation is added only for behavior that actually ships.

## Outcome

T3RL should become agent-assisted infrastructure for reproducible reinforcement learning research.
The delivery plan starts with one complete local experiment, then builds the scientific evidence and
automation required for open-ended learning, and only then expands to distributed execution and a
public research platform.

The phases are gates. A phase is complete when its exit criteria are demonstrated in code and tests,
not when its feature list has been partially implemented.

## Assumptions

- A small initial team benefits from narrow vertical slices and explicit deferral.
- T3 Code remains the server, client, agent, project, and remote-control foundation.
- Web is the primary client; desktop inherits it; mobile begins as a monitoring surface.
- Python algorithms and environments remain replaceable runner implementations.
- CPU-only `CartPole-v1` is sufficient to validate the first execution boundary.
- Experiment definitions are version-controlled; generated run artifacts are environment-local.
- Run lifecycle facts are durable but not event-sourced; high-volume metrics are neither.
- Upstream T3 Code continues to change and must remain practical to integrate.
- No phase silently installs Python, native packages, CUDA, or framework dependencies.

## Phase map

```text
Phase 0             Phase 1                    Phase 2
Decisions and  ───▶ First vertical slice ───▶ Open-ended research
contracts           PPO + CartPole             2A evidence and debugging
                                                2B agent-assisted research
                                                2C continual curricula
                                                       │
                                                       ▼
                 Phase 4                    Phase 3
                 Hardening and release ◀── Scale and backend ecosystem
```

## Phase 0: decisions and executable contracts

> **Status: implemented** as the run kernel — contracts, pure lifecycle, worker protocol, artifact
> confinement, storage, host boundary, supervision, and a deterministic Python fake worker, reachable
> over six RPC methods. Design and reasoning:
> [run kernel design](../superpowers/specs/2026-08-24-t3rl-run-kernel-design.md).
> The checked-in PPO/`CartPole-v1` definition now ships with the backend worker bundle. Artifact
> retention and cleanup remain open.

### Goal

Remove the architectural uncertainty that would otherwise be expensive to reverse after data and
process lifecycles exist.

### Deliverables

- Architecture decisions for:
  - RL lifecycle ownership — **settled**: a manager with an in-place projection, not an event-sourced
    domain. See [the run kernel design](../superpowers/specs/2026-08-24-t3rl-run-kernel-design.md);
  - authorization for starting and cancelling project-controlled training — **settled**: `rl.*`
    reuses the existing orchestration scopes so no paired device has to re-pair;
  - environment-local artifact storage, quotas, retention, and cleanup;
  - Python executable discovery and environment fingerprinting;
  - worker protocol versioning and compatibility;
  - server restart behavior for active runs.
- Effect schemas for the initial experiment, resolved manifest, run summary, lifecycle, metric batch,
  artifact metadata, capability report, and stable error codes.
- A versioned NDJSON worker protocol with TypeScript validation and Python fixtures.
- A deterministic fake worker that can complete, fail, emit malformed output, stall, and respond to
  cancellation.
- A checked-in example definition for PPO on `CartPole-v1`.
- A concrete storage layout and migration strategy for run projections and artifacts.
- A targeted verification matrix for local web, desktop wrapper, remote browser, and server restart.

### Implementation order

1. Record the decisions and name the domain concepts.
2. Define schemas and state transitions without UI or Stable-Baselines3.
3. Prove protocol compatibility with fixtures from TypeScript and Python.
4. Prove the lifecycle using the deterministic fake worker.
5. Freeze the minimum Phase 1 contract; additive compatible fields remain possible.

### Exit criteria

- Every lifecycle transition has a pure test and invalid transitions are rejected.
- Start and cancel requests have defined idempotency behavior.
- The server rejects malformed and incompatible worker messages deterministically.
- The fake worker can exercise success, failure, cancellation, stall, and unexpected exit without
  sleeps or process-name matching.
- Artifact IDs cannot escape a run or resolve a client-supplied absolute path.
- No unresolved decision can change the Phase 1 manifest, lifecycle, authorization, or storage model.

### Explicit deferrals

- Real RL training.
- Charts and videos.
- Multiple seeds or algorithms.
- Agent tools.
- GPU and distributed execution.

## Phase 1: first vertical slice

> **Backend status: implemented.** Workstreams 1A and 1B are complete, including transport
> idempotency, reconnectable metric snapshots, the bundled Stable-Baselines3 runner, separate seeded
> evaluation, production artifact budgets, and a gated real-worker smoke test. Workstream 1C now
> provides the project-scoped launcher, reconnectable run view, charts, manifest, and artifact access.
> The client/performance portions of 1D remain before Phase 1 can pass its full exit gate.

### Goal

Deliver one complete, reconnectable, reproducible experiment through the real T3RL stack: PPO on
`CartPole-v1`, one seed per run, CPU execution, web and desktop.

### Workstream 1A: durable run kernel

- Persist run request, preparation, running, cancelling, and terminal lifecycle facts.
- Build compact run projections for project history and reconnection.
- Supervise the exact worker process identity and bounded stderr log.
- Mark active runs as explicitly interrupted after server restart.
- Implement capability, list, start, cancel, subscribe, and artifact access boundaries.
- Deduplicate transport retries by `(projectId, requestId)` while preserving deliberate reruns under
  a new request id.
- Persist metrics outside the orchestration event log.

### Workstream 1B: Stable-Baselines3 worker

- Resolve the checked-in experiment definition into an immutable manifest.
- Verify Python, Gymnasium, Stable-Baselines3, and environment capability without installing them.
- Seed Python, NumPy, Gymnasium, and the runner wherever supported and record limitations.
- Train PPO on `CartPole-v1` and perform a separate final evaluation.
- Emit at most two metric batches per second through the versioned protocol.
- Retain manifest, episode summaries, bounded log, final model, evaluation summary, and replay.
- Report non-finite values and runner exceptions explicitly.

### Workstream 1C: RL Lab client

> **Status: implemented for the initial web/desktop surface.** Visual validation, remote-browser
> control, and renderer performance measurement remain in Workstream 1D.

- Add a run-centered RL Lab surface without representing each run as a conversation thread.
- Display capability failures before the user attempts to start training.
- Select the fixed experiment, choose an explicit seed, and start or cancel a run.
- Render authoritative lifecycle, return, episode length, evaluation return, policy loss, value loss,
  entropy, approximate KL when available, elapsed time, and worker errors.
- Reconnect to an active run without duplicating it.
- Open the resolved manifest, summary, log, model metadata, and evaluation replay.
- Preserve ordinary chat, terminal, project, and source-control behavior when the lab is unused.

### Workstream 1D: verification and measurement

- Use the fake worker for server lifecycle, subscription, cancellation, and protocol tests.
- Add a short Python worker test and a separately gated real CartPole smoke test.
- Measure server CPU, retained memory, metric bytes, artifact growth, and renderer update cost.
- Validate local web and desktop behavior; validate a remote browser controls execution on the server
  machine.
- Confirm disconnect, browser backgrounding, or client sleep does not cancel training.

### Exit criteria

- A fresh project can run the checked-in experiment to completion and reopen it after a T3RL restart.
- The same run ID is preserved across client reconnects and retries.
- Cancellation stops only the captured worker process tree and reaches a durable terminal state.
- No metric point is written as an orchestration event.
- The UI never reports completion from a quiet metric stream.
- The manifest contains the effective configuration, seed, source revision and dirty evidence, Python
  environment fingerprint, runner versions, instrumentation level, and relevant hardware.
- Phase 1 has a recorded performance baseline and no unbounded queue, log, chart, or artifact path.
- The end-to-end path is documented and demonstrable without modifying source during the demo.

### Explicit deferrals

- User-extensible algorithm and environment catalogs beyond the bundled Phase 1 definition.
- Multi-seed conclusions and statistical comparison.
- Deep gradients, activations, replay-buffer inspection, and state maps.
- Automatic experiment execution by agents.
- Checkpoint resume after server failure.
- Mobile authoring.

## Phase 2: open-ended research

Phase 2 is the next product phase after the vertical slice. It is divided into three ordered gates so
automation is built on trustworthy evidence rather than on a larger collection of opaque runs.

## Phase 2A: research evidence and algorithm debugging

### Goal

Make experiments scientifically comparable and make algorithm failures traceable from a visualization
to the run, trajectory, checkpoint, manifest, and source evidence that produced them.

### Deliverables

- Run groups for multiple seeds with explicit aggregation and uncertainty estimates.
- Side-by-side comparison of resolved manifests, code evidence, instrumentation, metrics, and
  artifacts.
- `minimal`, `standard`, and `deep` instrumentation levels recorded in the manifest.
- Typed, namespaced diagnostic adapters for policy-gradient, value-based, off-policy actor-critic, and
  custom algorithms.
- Sampled distributions for actions, observations, advantages, returns, values, Q targets, temporal-
  difference errors, gradients, and parameters where applicable.
- Explicit unsupported and not-captured states instead of synthetic zero values.
- Timeline-aligned evaluation replay with actions, rewards, probabilities or values, termination,
  and selected model estimates.
- Evidence navigation from a chart range to sampled episodes, retained checkpoint, manifest, and
  source snapshot.
- Project-defined Gymnasium environments, callbacks, metrics, and evaluation policies.
- Exportable research bundles containing the configuration and bounded evidence needed to reproduce a
  displayed result.

### Exit criteria

- T3RL reproduces a known baseline over multiple seeds and reports its aggregation method.
- A deliberately injected exploration or value-learning failure is visible and traceable to retained
  evidence.
- A researcher can compare two runs without overlooking changes in defaults, source, seed,
  instrumentation, or evaluation policy.
- Deep instrumentation is opt-in, bounded, measured, and visibly distinguished from standard runs.
- Custom metrics cannot modify the core run lifecycle or bypass storage and transport limits.
- At least one non-PPO algorithm uses the same lifecycle, manifest, and artifact contracts.

## Phase 2B: agent-assisted research loop

> **Initial client seam implemented.** The right-panel selector now provides project-versioned
> specialist instructions and a single review-gated Autoresearch brief with bounded baseline
> evidence. Its algorithm catalog spans the major RL families and treats RLVR, RLHF, and RLAIF as
> reward regimes orthogonal to PPO, GRPO, RLOO, and related optimizers. This does not satisfy the
> Phase 2B gate: most entries still need worker adapters, execution is not yet an autonomous
> controller, budgets are not yet server-enforced across turns, and durable hypothesis-to-outcome
> records and agent-facing evidence tools remain open.

### Goal

Let a coding agent investigate structured run evidence and prepare the next controlled experiment
without granting it an invisible or unbounded automation path.

### Deliverables

- Stable query boundaries for manifests, metric summaries, selected ranges, logs, artifacts, source
  evidence, and run comparison.
- Agent-accessible tools or commands built on the same authorized server services as the UI.
- Stable run, metric-range, checkpoint, artifact, and experiment identifiers in agent responses.
- A workflow for an agent to formulate a diagnosis, cite evidence, prepare a reviewable code or
  experiment diff, and propose an ablation.
- Explicit approval before starting project-controlled training or expanding a compute budget.
- A durable research record linking hypothesis, evidence, proposal, authorization, code change, run,
  and outcome.
- Evaluation tasks for diagnosis quality, evidence citation, false conclusions, and unnecessary
  experiment generation.

### Exit criteria

- Given a controlled failing experiment, an agent identifies the relevant evidence and proposes a
  falsifiable next experiment rather than only summarizing charts.
- Every diagnosis cites stable evidence that remains inspectable after the conversation ends.
- Proposed changes appear as ordinary workspace diffs and can be rejected without mutating a run.
- Agent retries cannot duplicate runs or silently increase the approved budget.
- The same research workflow works with more than one provider without embedding provider-specific
  fields in RL contracts.

## Phase 2C: continual curricula and open-ended learning

### Goal

Close a bounded research loop in which T3RL can generate tasks, measure acquired capabilities, and
select the next curriculum stage while preserving a reproducible account of the process.

### Deliverables

- Versioned task, environment, evaluator, capability, and curriculum definitions.
- Automatic task generation and mutation with recorded parentage and rationale.
- Curriculum policies that adapt difficulty to measured competence.
- Continual-learning runs that preserve and evaluate knowledge across task sequences.
- Transfer, generalization, regression, and catastrophic-forgetting evaluations.
- Self-play and multi-agent experiment definitions with explicit opponent and population lineage.
- Reward-hacking, specification-gaming, exploration-collapse, and evaluator-regression diagnostics.
- Bounded autonomous loops with limits on runs, wall time, compute, artifacts, and task depth.
- Human approval boundaries for generated code, evaluator changes, budget expansion, and external
  execution.
- Deterministic fake domains for testing curriculum behavior without expensive training.

### Exit criteria

- A researcher defines a capability objective and T3RL executes a bounded curriculum in a small
  domain.
- Generated tasks and evaluators are immutable, versioned, and attributable to their parents.
- The system reports which experiences produced which measured capability changes.
- Changing an evaluator cannot rewrite the interpretation of completed runs silently.
- A continual-learning demonstration measures both acquisition and forgetting.
- A self-play demonstration preserves opponent lineage and avoids comparing incompatible populations
  without warning.
- The complete loop can be replayed with fake runners and audited without an LLM.

### Explicit non-claims

- Completing Phase 2 does not demonstrate general intelligence or autonomous scientific discovery.
- Improvement on generated tasks does not establish broad generalization without independent
  evaluations.
- Agent-generated hypotheses remain hypotheses even when the system executes them automatically.

## Phase 3: scale and backend ecosystem

The detailed dependency order for LLM post-training, including reproducible `uv` environments,
checkpoint lineage, TRL/Axolotl training methods, distributed launchers, vLLM supervision, restart
adoption, retention, and mobile delivery, lives in the
[serious LLM post-training delivery plan](../superpowers/plans/2026-09-04-serious-llm-post-training.md).

### Goal

Run the same research model on larger hardware and external schedulers without turning T3RL into a
new cluster scheduler or leaking backend-specific concepts into clients.

### Deliverables

- A resource inventory for CPU, memory, accelerators, and runner availability.
- Per-environment concurrency, compute-budget, artifact, and retention policies.
- A scheduler/runner adapter contract for local processes, existing cluster schedulers, and managed
  research backends.
- Checkpoint publication, explicit resume, process adoption where safe, and preemption reporting.
- Runner adapters selected by demonstrated research need, such as CleanRL or RLlib.
- External artifact storage behind the existing artifact identity and authorization model.
- Backpressure and aggregation for many simultaneous metric streams.
- Failure classification for queueing, placement, worker, network, scheduler, and storage failures.
- Cost and resource attribution attached to runs and research loops.

### Exit criteria

- One experiment definition runs through local and external adapters with the same public manifest,
  lifecycle, metrics, and artifact model.
- Backend failure never becomes a false successful or user-cancelled run.
- Preemption and resume preserve explicit checkpoint lineage.
- Concurrency limits prevent one project from exhausting the T3 server environment.
- Clients require no scheduler-specific state machine.
- A load test demonstrates bounded server and WebSocket behavior at the declared supported scale.

### Trade-off

T3RL should integrate schedulers rather than reimplement placement, autoscaling, or cluster health.
Backend adapters add operational dependencies, but preserve a smaller and more maintainable research
control plane.

## Phase 4: hardening and public research platform

### Goal

Make the proven workflow installable, maintainable, secure, and understandable by researchers who did
not build it.

### Deliverables

- Supported Python and native-dependency setup paths for declared operating systems.
- Cross-platform web, desktop, and headless-server verification.
- Mobile monitoring, alerting, artifact summaries, and safe cancellation.
- Storage migrations, protocol compatibility policy, and backward-compatible run readers.
- Retention, export, archival, and recoverable cleanup workflows.
- Security review of process spawning, environment inheritance, project paths, artifacts, remote
  authorization, and agent-triggered execution.
- Performance budgets and regression benchmarks for worker overhead, server memory, transport, charts,
  and artifact indexing.
- User documentation, researcher tutorials, example environments, failure guides, and architecture
  references.
- Reproducible public case studies that include baselines, multiple seeds, limitations, and raw bounded
  evidence.
- A documented upstream-sync and release process for the fork.

### Exit criteria

- A new researcher can install T3RL, run the reference experiment, inspect a failure, and export a
  research bundle using only published documentation.
- Upgrade and rollback tests preserve existing run history.
- Artifact cleanup is bounded, visible, targeted by resolved IDs, and recoverable where practical.
- Security and performance findings have owners or explicit accepted-risk records.
- The project publishes at least one technically substantive case study rather than only a product
  demonstration.
- Every advertised platform and runner combination has an automated or documented verification path.

## Cross-phase rules

These requirements apply to every phase:

### Upstream maintainability

- Keep RL behavior isolated behind narrow modules and contracts.
- Integrate upstream regularly and separate sync changes from feature changes.
- Avoid global rename churn until the research architecture is proven.
- Record intentional divergences and their removal or maintenance cost.

### Research correctness

- Separate training and evaluation evidence.
- Preserve effective defaults, seeds, sample counts, instrumentation, aggregation, and source evidence.
- Never present a single run as a statistically established conclusion.
- Treat agent interpretations as evidence-linked hypotheses.

### Security and control

- Starting code, changing evaluators, expanding budgets, and deleting unique artifacts are explicit
  actions.
- Resolve processes, projects, runs, and artifacts by captured identities rather than search patterns.
- Keep credentials and secret values out of manifests, logs, metrics, artifacts, and agent context.

### Performance

- Bound queues, batches, retained history, log bytes, artifacts, and renderer work.
- Measure instrumentation overhead and record its level in the manifest.
- Keep lifecycle correctness independent of telemetry delivery.

### Verification

- Prefer deterministic fake workers and fake research domains for lifecycle and automation tests.
- Gate costly real-training tests separately.
- Test failure and cancellation as first-class outcomes.
- Verify the smallest affected repository surface and avoid unrelated full-suite work during
  development.

### Documentation

- Update internal architecture when a boundary changes.
- Add user documentation only when behavior ships.
- Keep a runnable reference experiment for every supported public workflow.

## Phase transition review

Before starting the next phase, maintainers record:

1. the working demonstration and its exact version;
2. exit criteria that passed and any explicitly accepted gaps;
3. performance, storage, and failure measurements;
4. new security or research-validity risks;
5. upstream divergence introduced by the phase;
6. contracts now considered stable and contracts still experimental;
7. scope removed from or added to the next phase, with rationale.

An accepted gap that changes manifest meaning, lifecycle semantics, authorization, or scientific
interpretation blocks the next phase. Cosmetic gaps and explicitly unsupported optional artifacts do
not necessarily block it.

## Decisions to revisit as the system grows

- Split RL lifecycle into a separate service only when process or cluster ownership can no longer be
  safely supervised by the T3 server boundary.
- Move artifacts from local storage when declared retention or remote execution requirements exceed
  local capacity, not merely in anticipation of scale.
- Replace NDJSON only when measured serialization or transport cost becomes material; preserve message
  semantics across transports.
- Expand mobile authoring only after run and experiment contracts stabilize and researcher demand is
  demonstrated.
- Add multi-tenant isolation only with an explicit shared-server product requirement; remote access by
  itself is not multi-tenancy.
- Reconsider the fork boundary if upstream develops an extension mechanism capable of hosting the RL
  server, client, and worker lifecycle without losing required guarantees.

## Initial public success definition

The project is ready to be presented as serious RL research infrastructure when it can demonstrate:

- one complete and reproducible local experiment workflow;
- multi-seed comparison with scientifically honest uncertainty;
- one real algorithm failure diagnosed through retained evidence;
- one agent-assisted ablation with cited evidence and explicit approval;
- one bounded continual-curriculum demonstration;
- measured performance and failure behavior;
- a technical case study that another researcher can reproduce.

This definition values research evidence and systems quality over the number of supported algorithms,
environments, charts, or integrations.
