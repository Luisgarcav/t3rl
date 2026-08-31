# t3rl

**An open, agent-assisted control plane for reproducible reinforcement learning research.**

t3RL brings experiment execution, live evidence, run comparison, diagnostics, and coding agents
into one desktop workspace. It builds on [T3 Code](https://github.com/pingdotgg/t3code), preserving
its fast, remote-ready agent interface while adding a project-scoped RL Lab backed by Python,
Gymnasium, Stable-Baselines3, and an optional TRL adapter for bounded LLM post-training.

> [!WARNING]
> t3RL is alpha software. The current release is intended for local research and development, not
> unattended or production training workloads.

## What t3rl does

- Runs RL experiments on the server attached to a project, so training continues if the desktop or
  browser client disconnects.
- Streams bounded lifecycle and metric updates without placing high-volume telemetry in the agent
  conversation.
- Records the resolved seed, configuration, source revision, Python environment, runner version,
  metrics, and artifacts for each run.
- Replays evaluation trajectories step by step and exposes observations, actions, rewards, and
  episode boundaries; LLM runs reuse the same bounded view for prompts, completions, verifier
  decisions, and rewards.
- Compares compatible runs across seeds while surfacing configuration, source, and environment
  drift.
- Provides explainable diagnostic signals for return collapse, non-finite values, excessive KL,
  low entropy, divergent value loss, stalled streams, and train/evaluation gaps.
- Explores retained metrics as a chart, bounded data table, or declarative source, and explains the
  resolved algorithm through a conceptual stage visualizer.
- Adds reusable research specialists and a review-gated Autoresearch workflow that prepares one
  falsifiable iteration at a time.
- Gives Codex, Claude Code, Cursor, Grok, and OpenCode project-scoped tools for inspecting RL
  evidence and, with the active permission mode, starting or cancelling runs.

## RL execution coverage

The bundled CPU runner uses `MlpPolicy` and ships with these experiments:

| Algorithm | Learning style                  | Action space | Bundled environment |
| --------- | ------------------------------- | ------------ | ------------------- |
| PPO       | On-policy actor-critic          | Discrete     | `CartPole-v1`       |
| A2C       | On-policy actor-critic          | Discrete     | `CartPole-v1`       |
| DQN       | Off-policy value-based          | Discrete     | `CartPole-v1`       |
| SAC       | Off-policy, entropy-regularized | Continuous   | `Pendulum-v1`       |
| TD3       | Off-policy actor-critic         | Continuous   | `Pendulum-v1`       |
| DDPG      | Off-policy actor-critic         | Continuous   | `Pendulum-v1`       |

An optional single-GPU preview adds one executable LLM post-training experiment:

| Algorithm | Reward regime                | Model                        | Bundled dataset      |
| --------- | ---------------------------- | ---------------------------- | -------------------- |
| GRPO      | RLVR, exact-integer verifier | `Qwen/Qwen2.5-0.5B-Instruct` | `arithmetic-rlvr-v1` |

The TRL worker records the resolved model commit, dataset SHA-256, split policy, verifier identity,
dependency fingerprint, execution backend, and token/wall-clock/GPU-hour limits. It evaluates a
versioned holdout before and after training, emits optimizer and resource metrics, and retains
bounded prompt/completion evidence plus a separate evaluation artifact. It deliberately does not
retain a model checkpoint in this slice. vLLM, distributed scheduling, arbitrary models, and
production-scale datasets are not yet integrated.

The research catalog is intentionally broader than the execution catalog. It can help plan and
review work involving tabular RL, model-based RL, offline and imitation learning, multi-agent RL,
bandits, evolutionary methods, RLHF/RLAIF/RLVR, and LLM policy optimization. Execution currently
covers the table above; every other catalog method still needs an adapter. The UI reports this
distinction and does not claim that an unavailable runner is installed.

## Run the desktop app from source

### Prerequisites

- Git
- Node.js `24.13.1` or newer within the Node 24 release line
- [Vite+](https://viteplus.dev/) (`vp`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python environments
- Python `3.10`–`3.13` for real RL runs (`uv` can install it on demand)
- At least one authenticated provider CLI if you also want to use the agent workspace

Install Vite+ on macOS or Linux:

```bash
curl -fsSL https://vite.plus | bash
```

On Windows PowerShell:

```powershell
irm https://vite.plus/ps1 | iex
```

### 1. Clone and install JavaScript dependencies

```bash
git clone https://github.com/Luisgarcav/t3rl.git
cd t3rl
vp i
```

### 2. Install the optional RL runtime

On macOS or Linux:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ./python
export T3RL_PYTHON="$PWD/.venv/bin/python"
```

For the optional GRPO/RLVR experiment, install the LLM extra into that environment and use a CUDA
PyTorch runtime:

```bash
uv pip install --python .venv/bin/python -e './python[llm]'
.venv/bin/python -c 'import torch; assert torch.cuda.is_available()'
```

On Windows PowerShell:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -e .\python
$env:T3RL_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
```

The optional LLM extra can be installed with
`uv pip install --python .\.venv\Scripts\python.exe -e '.\python[llm]'`; the configured PyTorch
build must report an available CUDA device before RL Lab enables the TRL runner.

The Python environment is optional if you only want to open the app or use its agent features.
RL Lab checks capabilities when it opens and shows an actionable message instead of installing
packages automatically.

### 3. Launch Electron

Run this from the same terminal in which `T3RL_PYTHON` is set:

```bash
vp run dev:desktop
```

This starts the Vite renderer and the Electron desktop host together. Development state is isolated
under this checkout's gitignored `.t3/` directory. Stop the process with `Ctrl+C`.

To build an installer for the current platform:

```bash
vp run dist:desktop:artifact
```

Platform-specific build commands are also available:

```bash
vp run dist:desktop:dmg    # macOS
vp run dist:desktop:linux  # Linux AppImage
vp run dist:desktop:win    # Windows NSIS installer
```

## Use RL Lab

1. Open or create a project in t3RL.
2. Select the flask button next to the project, or open an empty right panel and choose
   **Experiments**.
3. Confirm that the experiment's `stable-baselines3` or `trl` runner is available.
4. Choose a bundled experiment, set an integer seed, and select **Start run**.
5. Use **Overview** for lifecycle, live metrics, manifests, cancellation, and artifacts; use
   **Data** to inspect raw metric observations and **Algorithm** to explore the resolved training
   flow.
6. Use **Behavior** to inspect evaluation trajectories or LLM prompt/completion verifier samples,
   **Compare** to aggregate compatible runs, and **Diagnostics** to review bounded heuristic
   findings.
7. From the right panel, use **Specialists** to prepare a reusable research role or
   **Autoresearch** to draft a budgeted, approval-gated iteration in the agent composer.

Every intentional rerun receives a new run ID. Reconnecting to an existing run resumes its live
view without duplicating metric points or artifacts.

## Use RL Lab with an agent

The default agent receives the product-native `t3-code` RL toolkit for the current project. The same
tools are available to every built-in provider without additional RL-specific configuration.

| Research task                                           | Agent tools                     |
| ------------------------------------------------------- | ------------------------------- |
| Check runners and experiments                           | `rl_capabilities`               |
| Find and inspect retained runs                          | `rl_list_runs`, `rl_get_run`    |
| Explore metrics and diagnostics evidence                | `rl_query_metrics`              |
| Compare configurations and results                      | `rl_compare_runs`               |
| Read logs, summaries, evaluations, and behavior replays | `rl_read_artifact`              |
| Execute or stop training                                | `rl_start_run`, `rl_cancel_run` |

The server derives project scope from the agent's thread, so a tool call cannot select another
project. Metric responses and textual artifacts are bounded; binary models are not copied into the
agent context. Starting and cancelling training remain permission-aware actions. Disabling agent
browser access removes only the `preview_*` tools and does not disable the RL toolkit.

## Architecture

```mermaid
flowchart TB
    subgraph control["Project-scoped control surfaces"]
        direction LR
        clients["Web & desktop<br/>Launch · Metrics · Algorithm · Behavior · Compare · Diagnostics"]
        agents["Default coding agent<br/>Codex · Claude · Cursor · Grok · OpenCode"]
    end

    server["Node server — execution authority<br/>Capability probes · Lifecycle · Supervision · Artifact authorization"]
    store[("Project run store<br/>Manifest · Metrics · Artifacts")]

    subgraph adapters["Replaceable Python runner adapters"]
        direction LR
        sb3["Stable-Baselines3<br/>Control tasks · Evaluation trajectories · Models"]
        trl["TRL native<br/>GRPO/RLVR · Before/after evaluation · Completion evidence"]
        axolotl["Axolotl adapter — planned<br/>Config translation · Plugins · Async rollouts"]
        distributed["Distributed launchers — planned<br/>Accelerate · FSDP · DeepSpeed · vLLM sidecar"]
    end

    clients <-->|"Authenticated typed RPC"| server
    agents <-->|"Project-scoped rl_* tools"| server
    server <-->|"Durable bounded evidence"| store
    server <-->|"Spawn + versioned NDJSON"| sb3
    server <-->|"Spawn + versioned NDJSON"| trl
    server -.->|"Same protocol"| axolotl
    server -.->|"Launcher-owned topology"| distributed
```

The server owns execution. The client renders authoritative state, and the Python worker stays
replaceable behind a small process protocol. This keeps local, desktop, and remote behavior aligned
without coupling the UI to a particular RL framework.

## Current limits

- Each run uses one seed and one local worker process. Control experiments use CPU; the bundled
  GRPO preview requires one CUDA GPU. Multi-seed comparisons combine separate retained runs.
- Active processes are marked `interrupted` after a server restart; checkpoint resume is not yet
  implemented.
- The bundled experiment catalog is fixed and intentionally small.
- The GRPO preview uses a small fixed holdout and does not retain checkpoints. It does not yet
  support vLLM, arbitrary models, Axolotl execution, or distributed launchers.
- Diagnostics are inspection heuristics, not causal conclusions or universal RL thresholds.
- Autoresearch prepares a single reviewed iteration; it does not edit code, launch training, expand
  budgets, or loop autonomously without explicit approval.
- RL experiment authoring is currently centered on the web/desktop client.

## Documentation

- [RL Lab user guide](./docs/user/rl-lab.md)
- [RL Lab architecture](./docs/internals/rl-lab.md)
- [Framework adapter architecture](./docs/internals/rl-framework-adapters.md)
- [Algorithm coverage](./docs/internals/rl-algorithm-coverage.md)
- [Development roadmap](./docs/internals/rl-lab-roadmap.md)
- [Install and first run](./docs/user/install.md)
- [Remote access](./docs/user/remote-access.md)
- [Contributor guide](./CONTRIBUTING.md)

## Project status and attribution

t3RL is an independent open-source fork of [T3 Code](https://github.com/pingdotgg/t3code). We are
grateful to its maintainers and contributors for the multi-surface agent platform on which this
research workspace is built.

Licensed under the [MIT License](./LICENSE).
