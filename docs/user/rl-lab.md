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

Stable-Baselines3, TRL, and Axolotl use independent locked environments because their framework
constraints can conflict. The server administrator selects them with
`T3RL_PYTHON_STABLE_BASELINES3`, `T3RL_PYTHON_TRL`, or `T3RL_PYTHON_AXOLOTL`; a shared
`T3RL_PYTHON` remains a fallback. RL Lab checks a discovered lock without changing it and records
its SHA-256 together with the interpreter, framework, platform, PyTorch, CUDA, and driver evidence.

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

- **Overview** shows lifecycle status, cancellation, continuation actions, live training metrics,
  the immutable resolved manifest, signed artifact links, and run lineage.
- **Data** explores every retained metric, including custom worker keys. Switch between a bounded
  line chart, the latest raw observations, and the declarative source used by the renderer.
- **Algorithm** walks through a conceptual stage graph derived from the immutable resolved
  manifest. Its playback controls explain the algorithm; the highlighted stage is not presented as
  live worker execution unless later instrumentation explicitly reports that state.
- **Behavior** replays bounded evidence artifacts. Control runs show observations, actions, rewards,
  and episode boundaries; LLM post-training runs show prompts, references, completions, parsed
  answers, verifier decisions, and scalar rewards.
- **Compare** shows descriptive exact-step observations across verified seeds and can request an
  authoritative paired study comparison from retained evaluation evidence. Missing values stay
  missing; the result identifies its statistical unit, uncertainty, and excluded runs or samples.
- **Diagnostics** screens retained evidence for non-finite values, return collapse, excessive KL,
  low entropy, divergent value loss, stalled streams, and train/evaluation gaps. These are
  explainable inspection signals, not causal conclusions.

RL Lab receives a bounded snapshot before live updates. Artifact-heavy runs use a separate paginated
artifact inventory. New artifacts show their server-computed SHA-256 and `ready` state; artifacts
from older runs remain readable and are labeled **Legacy · unverified**. Reopening the page or
reconnecting to the server resumes the same run ID without duplicating metric points or artifacts.
A quiet metric stream does not imply completion; only the lifecycle status does.

### Continue from a checkpoint or adapter

Post-training runs publish two deliberately different outputs:

- **Trainer-state resume** checkpoints include the LoRA weights plus trainer, optimizer, scheduler, random
  number generator, data cursor, and—when needed—gradient scaler state. RL Lab only offers
  **Resume step N** after the server has verified the directory, its SHA-256, and all compatibility
  evidence.
- **PEFT adapters** contain portable LoRA weights and configuration tied to a pinned base model.
  **Start from adapter** creates a warm start; it does not claim to preserve optimizer or data
  position.

Both actions create a new child run. The parent run and its artifacts remain immutable. The child
records the relation, parent ID, source artifact ID, source step, and exact source hash; **Run
lineage** shows that chain. A changed model or tokenizer revision, PEFT configuration, precision,
quantization, trainable module set, framework, environment, or lock causes an explicit compatibility
error instead of a best-effort continuation.

Restoring trainer state and reproducing numerical results are separate guarantees. Numerical
agreement requires a comparison with an uninterrupted run on a recorded platform using a declared
tolerance. Matching hashes or seeds alone does not establish that agreement.

Cancelling a post-training run first requests a graceful checkpoint. The worker gets the deadline
declared in its resolved policy; the server then terminates the exact process it started if the
deadline expires.

The default coding agent can use these same project-scoped utilities through the product-native RL
Lab tools. It can inspect the experiment catalog, manifests, metrics, comparisons, textual artifacts,
logs, evaluations, environment trajectories, and LLM prompt/completion verifier replays without
relying on screenshots. `rl_list_artifacts` traverses a large inventory one bounded page at a time;
`rl_resume_run` and `rl_warm_start_run` preserve the same distinction and lineage as the UI.
Starting, resuming, warm-starting, or cancelling a run remains a permission-aware action. Turning
off agent browser access disables only browser control; it does not remove the agent's RL Lab
evidence tools.

From a run detail, **Use as baseline** saves that run into the workspace research configuration and
opens Autoresearch. From Autoresearch, **Open run** returns to the selected baseline evidence.

### Export evidence

For a finished, failed, cancelled, or interrupted run, select **Export evidence** in its detail
header. The download contains the selected evidence and an offline verifier. The export dialog shows
the archive and index hashes, file count, and omissions; the bundle lists the omitted evidence so
another researcher can identify what is still needed. Exporting keeps the source run unchanged.

Artifact integrity, environment reconstruction, trainer-state resume, numerical reproducibility,
and repeatability on independent evaluation data require distinct evidence. See
[Research records and evidence exports](./rl-evidence.md) for the verification workflow and scope.

## Interpretation limits

Each run uses one seed and one local worker process. The server does not yet adopt a still-running
worker after its own restart; it marks that attempt `interrupted`, and a verified checkpoint from the
attempt may be used to create a child run. Control experiments run on CPU; the bundled GRPO path
requires CUDA and records token, wall-clock, GPU-hour, checkpoint, and retention limits. Its small
fixed holdout demonstrates the before/after evaluation path but is not a statistically strong
benchmark. Project-defined models and offline methods must pass their runner's capability and
validation checks; vLLM and distributed training remain unavailable. Descriptive comparison combines
existing snapshots. Studies schedule separate runs and report the selected statistical unit:
training-seed pairs, or paired evaluation samples within those seed pairs.
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
The integrated TRL paths include GRPO with verifiable rewards and project-defined SFT/DPO. Other
methods require a worker adapter; selecting one never pretends that its dependencies are installed.

The first Autoresearch slice is deliberately review-gated. The agent must stop after proposing one
falsifiable hypothesis, a minimal diff, and an exact run plan. It cannot apply changes, start
training, expand the budget, change the evaluator, or begin another iteration without a later
explicit approval. This is not yet an unattended multi-iteration controller.

## Studies and paired comparisons

A study groups two or more experiment variants under one immutable evaluation protocol and an
explicit set of training, data, evaluation-sample, and generation seeds. Study runs are scheduled
with bounded concurrency. A failed or cancelled member leaves the study partial instead of making
it look complete.

Use **Compare** with a study ID and its baseline/candidate labels to request paired analysis. Select
an evaluation metric such as `eval_after/loss` or DPO's `eval_after/preference_accuracy`, the
statistical unit, resample count, and missing-pair policy. The study result is independent of the
descriptive chart metric selected below it.

The result shows the paired seed count, exact seed set, estimator version, actual statistical unit,
mean difference, confidence interval when available, and unmatched samples or excluded runs with
their reasons. Version 2 honors the selected unit; old version-1 results remain historical evidence
and are not recalculated as version 2 silently.

Missing evaluation evidence, incompatible protocols, unsupported estimators, incomplete required
pairs, and computation limits have distinct explanations. If the resampling budget is exceeded,
reduce the requested count or study size. Legacy aggregate-only evaluation cannot supply paired
sample evidence. A negative effect or inconclusive comparison remains a valid research outcome.

## Project-owned experiments

Put versioned definitions in `.t3rl/experiments/<id>.json` and address them as `project__<id>`.
Bundled definitions use `bundled__<id>`. Project definitions choose a server-supported adapter
(`trl` or `axolotl`) and method; they cannot provide a process command or entrypoint.

Each definition declares its model and tokenizer revisions, dataset, verifier, split policy,
evaluation protocol, budgets, instrumentation, and runner configuration. Reproducible definitions
must pin every external revision. Exploratory definitions may float, but validation reports a
visible warning.

Run `rl_validate_experiment` from the project agent before training. Validation reads the current
definition, checks runner availability and budgets, rejects paths or symlinks outside the project,
and reports hashes without modifying the environment. When training starts, local dataset, verifier,
and definition files are copied into the run evidence and re-hashed. Later edits therefore cannot
change an active or completed run.

### SFT and DPO methods

Set `method` to `sft` with `datasetFormat` `sft-text` or `sft-conversation`, and select the
`held-out-loss` evaluation claim. Text rows contain `text`; conversational rows contain `prompt`
and `completion`.

Set `method` to `dpo` with `datasetFormat` `dpo-preference`, and select
`preference-accuracy`. Every row must contain `prompt`, `chosen`, and `rejected`; equal or missing
preference responses are rejected before model allocation. Training loss is shown as diagnostics,
not treated as the evaluation claim.

Runner availability is method-specific. SFT and DPO can be available on a CPU environment while
GRPO still reports its CUDA requirement. A completed SFT adapter can be selected as the explicit
warm-start input of a DPO experiment; this creates a new run with immutable lineage rather than
changing the SFT run.
