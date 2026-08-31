# RL Lab

RL Lab is a project-scoped workspace for launching and monitoring reproducible reinforcement
learning experiments. Runs are managed by the connected server rather than by a chat thread, so
closing or reconnecting the client does not stop training.

## Open the lab

You can open RL Lab in either place:

- Open the project selector in the thread sidebar, find the project, and select the flask button
  labeled **Open RL Lab**.
- Open an empty right panel from a project thread and choose **Experiments**. This keeps the
  conversation and lab evidence beside each other.

The right-panel launcher groups its surfaces into **Runtime**, **Workspace**, and **Research**.
Research is ordered as **Experiments**, **Autoresearch**, then **Specialists**: inspect evidence,
prepare the next study, and edit reusable roles only when needed.

The selected environment controls where Python runs and where run history and artifacts are stored.
A remote browser therefore controls the runner on its connected server, not on the browser device.

## Start a run

The launch screen checks the server's runner capabilities before enabling **Start run**. If Python,
Stable-Baselines3, TRL, Axolotl, or the CUDA runtime required by an experiment is unavailable, the
warning includes the server-side remedy; RL Lab never installs packages automatically.

Axolotl pins dependency versions that cannot share an environment with the TRL runner, so it needs
its own Python environment. Point `T3RL_PYTHON_AXOLOTL` at that interpreter on the server; runners
without a dedicated variable keep using the shared `T3RL_PYTHON`. Each run records the interpreter
that served it, so a result always says which environment produced it.

Choose a catalog experiment, enter an integer seed, and start the run. The bundled
Stable-Baselines3 catalog covers PPO, A2C, and DQN on `CartPole-v1`, plus SAC, TD3, and DDPG on
continuous-control `Pendulum-v1`. The optional TRL catalog includes a bounded single-GPU GRPO/RLVR
samples using `Qwen/Qwen2.5-0.5B-Instruct`, a versioned arithmetic dataset, and an exact-integer
verifier, and the same task runs through Axolotl as well. Each keeps a versioned holdout out of the
optimizer dataset, evaluates it before and after training, and records the comparison as metrics and
an evaluation artifact. A new request creates a new run, while transport retries are deduplicated by
the server.

Two dataset versions ship. The `v1` samples evaluate four held-out rows twice, which the base model
already answers almost perfectly, so the before/after comparison there cannot resolve a change
smaller than one sampled generation. The `v2` samples use a harder dataset and evaluate 64 held-out
rows four times, giving a measurement with room to move in either direction. Prefer `v2` when the
question is whether training changed anything; the run's recorded dataset checksum tells you which
one produced a result.

## Monitor and reopen runs

The project history shows every retained run and its authoritative lifecycle state. Selecting a run
opens its live view. The navigation under the lab header separates six investigation surfaces:

- **Overview** shows lifecycle status, cancellation, live training metrics, the immutable resolved
  manifest, and signed artifact links.
- **Data** explores every retained metric, including custom worker keys. Switch between a bounded
  line chart, the latest raw observations, and the declarative source used by the renderer.
- **Algorithm** walks through a conceptual stage graph derived from the immutable resolved
  manifest. Its playback controls explain the algorithm; the highlighted stage is not presented as
  live worker execution unless later instrumentation explicitly reports that state.
- **Behavior** replays bounded evidence artifacts. Control runs show observations, actions, rewards,
  and episode boundaries; LLM post-training runs show prompts, references, completions, parsed
  answers, verifier decisions, and scalar rewards.
- **Compare** aggregates exact-step observations across verified seeds for one experiment. Missing
  values remain missing, duplicate seeds are not counted twice, and configuration or environment
  drift is called out before interpretation.
- **Diagnostics** screens retained evidence for non-finite values, return collapse, excessive KL,
  low entropy, divergent value loss, stalled streams, and train/evaluation gaps. These are
  explainable inspection signals, not causal conclusions.

RL Lab receives a bounded snapshot before live updates. Reopening the page or reconnecting to the
server resumes the same run ID without duplicating metric points or artifacts. A quiet metric stream
does not imply completion; only the lifecycle status does.

The default coding agent can use these same project-scoped utilities through the product-native RL
Lab tools. It can inspect the experiment catalog, manifests, metrics, comparisons, textual artifacts,
logs, evaluations, environment trajectories, and LLM prompt/completion verifier replays without
relying on screenshots. Starting or cancelling a run remains a permission-aware action. Turning
off agent browser access disables only browser control; it does not remove the agent's RL Lab
evidence tools.

From a run detail, **Use as baseline** saves that run into the workspace research configuration and
opens Autoresearch. From Autoresearch, **Open run** returns to the selected baseline evidence.

## Interpretation limits

Each run uses one seed and one local worker process, and does not resume after a server failure.
Control experiments run on CPU; the bundled GRPO preview requires CUDA, records token, wall-clock,
and GPU-hour limits, and intentionally retains no checkpoint. Its small fixed holdout demonstrates
the before/after evaluation path but is not a statistically strong benchmark. It does not yet
support arbitrary models, vLLM, or distributed training. Multi-seed comparison
combines separate runs rather than launching a sweep, and uses the resolved seed as the statistical
unit.
Automatic diagnostics use configurable heuristics and should be checked against the task,
algorithm, reward scale, and emission cadence.

## Research specialists

Open an empty right panel and choose **Specialists** to edit reusable research roles. The bundled
profiles cover research leadership, algorithm debugging, environment and reward auditing, ablation
design, and evaluation review. Profiles are saved in `.t3rl/research.json` inside the active
workspace, so they follow the branch and work remotely through the connected server.

**Use in composer** prepares the selected role in the current thread. The instructions remain
visible and editable in the composer; they are project-authored context rather than a hidden,
provider-specific system prompt. The current thread's provider and model remain in control.

## Prepare an Autoresearch iteration

The **Autoresearch** surface prepares one bounded scientific iteration:

1. Define an objective and explicit success metric.
2. Select a retained baseline run. T3 Code adds a bounded manifest and metric summary to the
   prepared turn rather than relying on screenshots.
3. Choose an algorithm, task type, and reward or feedback source.
4. Set the maximum run count, wall-clock minutes, and exact seeds.
5. Choose the specialist profiles that should review the evidence. Use **Manage** only when their
   reusable instructions need editing.
6. Select **Prepare iteration** from the persistent action bar and inspect the resulting composer
   text before sending it.

The research catalog covers representative methods across tabular and value-based RL, policy
gradients and actor-critic, entropy-regularized and model-based RL, offline and imitation learning,
multi-agent RL, bandits, evolutionary methods, LLM policy optimization, and adjacent preference
optimization. A custom entry keeps project-specific algorithms possible without changing the file
format.

Algorithm and reward regime are separate choices. For example, GRPO, PPO for LLMs, RLOO,
REINFORCE++, or DAPO can use verifiable rewards; choosing **Verifiable reward (RLVR)** adds verifier
integrity, leakage, determinism, false-acceptance, and held-out evaluation checks to the prepared
turn. Human feedback (RLHF), AI feedback (RLAIF), learned reward models, environment rewards, and
rule-based or hybrid rewards remain distinct options.

Every research-catalog entry supports planning and evidence review. The surface separately reports
execution availability: the integrated Stable-Baselines3 path supports PPO and A2C for discrete or
continuous control, DQN for discrete control, and SAC, TD3, and DDPG for continuous control. A run
still requires a matching version-controlled experiment and a successful server capability probe.
The integrated TRL preview adds GRPO with verifiable rewards for its bundled model and dataset.
Other methods require a worker adapter; selecting one never pretends that its dependencies are
installed.

The first Autoresearch slice is deliberately review-gated. The agent must stop after proposing one
falsifiable hypothesis, a minimal diff, and an exact run plan. It cannot apply changes, start
training, expand the budget, change the evaluator, or begin another iteration without a later
explicit approval. This is not yet an unattended multi-iteration controller.
