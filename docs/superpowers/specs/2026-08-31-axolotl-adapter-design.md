# Axolotl adapter design

> For maintainers. Design for the second RL Lab execution backend.

Status: approved, not implemented
Date: 2026-08-31
Adapter context: [RL framework adapters](../../internals/rl-framework-adapters.md)
Architecture context: [T3RL research lab architecture](../../internals/rl-lab.md)

## Purpose

Add Axolotl as a second executable backend for one version-pinned GRPO configuration, so the adapter
boundary the native TRL runner defined is proven against a framework that did not shape it.

A boundary exercised by exactly one implementation is a guess. This increment turns it into a
tested claim: two independent backends resolve the same task through the same protocol, emit the
same artifacts, and normalize into the same metric namespaces.

## Scope

### In scope

- `apps/server/src/rl/Capabilities.ts`: per-runner interpreter resolution, an Axolotl probe, and an
  `axolotl` entry in the capability report.
- `python/t3rl_worker/rlvr.py`: the verifier, dataset loading and hashing, split policy, evidence
  ledger, and protocol emission, extracted from `trl_worker.py` and shared by both workers.
- `python/t3rl_worker/axolotl_worker.py`: config translation and execution against Axolotl 0.18.0.
- `python/t3rl_worker/experiments/arithmetic-grpo-axolotl.json`.
- An `axolotl` optional extra in `python/pyproject.toml`, separate from `llm`.
- Focused tests, none of which require Axolotl installed.
- Documentation moving Axolotl from planned to implemented.

### Out of scope

No checkpoint retention or resume. No Accelerate launcher or multi-GPU execution. No FSDP or
DeepSpeed. No vLLM. No asynchronous GRPO. No second Axolotl configuration. No change to the
WebSocket contract, the run lifecycle, or any client.

## Findings that shaped this design

Both come from a feasibility spike run on 2026-08-31.

### F1: Axolotl cannot share an environment with the TRL runner

Axolotl pins exact versions rather than ranges. Release 0.18.0 requires `trl==1.8.0`,
`transformers==5.14.1`, `torch==2.12.1`, `accelerate==1.13.0`, and `datasets==4.8.4`. Every one is
below the pin the TRL runner was verified against.

The resolver proves the conflict is not negotiable: `axolotl` plus `trl==1.10.0` is unsatisfiable,
and no published Axolotl release accepts any TRL above `1.8.0`. Installing Axolotl beside the TRL
runner means downgrading a backend that already works.

### F2: Axolotl GRPO does not require vLLM

The Axolotl GRPO documentation describes a two-process architecture with a vLLM server and implies
two GPUs. The source disagrees. `axolotl/utils/schemas/trl.py` declares `use_vllm: bool =
Field(default=False)`, `core/trainers/grpo/trainer.py` guards every vLLM call behind
`if self.args.use_vllm:`, and a `colocate` mode exists for single-GPU use.

GRPO therefore runs on the native TRL generation path without vLLM, on one GPU. The documented
delivery order stands: vLLM stays a later, separately supervised increment rather than a
prerequisite for this one.

## Decisions

### D1: interpreters are resolved per runner, through the environment

`candidateExecutables` gains the requesting runner id and consults `T3RL_PYTHON_AXOLOTL` first,
then the shared `T3RL_PYTHON`, then `python3` and `python`. A runner with no dedicated variable
behaves exactly as it does today.

This extends the convention already in the file rather than introducing a second one. No contract
changes: `RlResolvedManifest` already records `pythonExecutable` per run, so which interpreter
served a run is already durable evidence.

Rejected: a `runnerId`-to-path map in `settings.json`. It is more discoverable, and it costs a
contract field, a settings surface, and client work in web, desktop, and mobile to configure
something that is a property of the server host, not of the user. If researchers later ask to manage
environments from Settings, this design does not block it — the resolution point stays one function.

Rejected: one environment with TRL downgraded to 1.8.0. It breaks a verified backend, ties every
future TRL upgrade to Axolotl's release cadence, and makes each adapter's dependencies the other's
problem. That is the coupling an adapter boundary exists to prevent.

### D2: both workers resolve the same task

The Axolotl experiment uses the same model, model revision, dataset, verifier, split policy, and
seed handling as `arithmetic-grpo-rlvr`. Only the backend differs.

This is what makes the existing run-comparison view a test of the adapter boundary. If each backend
brought its own task, a difference in results could not be attributed: adapter defect, framework
difference, or a different problem being solved would all look alike.

### D3: shared behavior is extracted, not duplicated

The verifier, dataset loading and hashing, split policy, `EvidenceLedger`, and protocol emission
move to `python/t3rl_worker/rlvr.py`, imported by both workers.

Duplicating them would let two implementations of the same measurement drift silently, which
defeats D2. The module imports no framework at module level — `trl_worker.py` already loads its
dependencies lazily — so it stays importable from both environments.

The extraction is behavior-preserving. Its proof is that the TRL suite stays green and a 20-seed
sweep on the bundled configuration reproduces byte-identical evaluation results.

### D4: the first comparison proves the protocol, not equivalent results

The bundled holdout is four rows sampled twice: eight observations, resolution 0.125, against a base
model that already passes roughly 0.87 of it. A 20-seed sweep on 2026-08-31 measured the noise floor
of a single evaluation of an untrained policy at `sd = 0.111`, matching the binomial prediction of
0.119 for eight samples. Run-to-run variation is entirely sampling noise.

At that resolution the two backends will produce numbers that are indistinguishable from each other
and from noise. This increment therefore claims only that the backends agree on protocol, artifacts,
metric namespaces, and evidence semantics. It does not claim their learning outcomes are equivalent,
and no such claim may be added to the UI or documentation on this evidence.

Raising holdout resolution needs a larger and harder dataset. It is deliberately a separate
increment, because doing it here would change what a run measures and what a backend does in the
same change, and a disagreement between the two backends could not then be attributed.

## Architecture

### Module layout

```
python/t3rl_worker/
  rlvr.py                                 # shared: verifier, dataset, split, ledger, protocol
  trl_worker.py                           # TRL backend, imports rlvr
  axolotl_worker.py                       # Axolotl backend, imports rlvr
  experiments/arithmetic-grpo-rlvr.json
  experiments/arithmetic-grpo-axolotl.json
```

Neither worker imports the other. `rlvr.py` imports no framework at module level.

### Capability resolution

`resolveRunner({ runnerId: "axolotl" })` resolves the Axolotl interpreter, runs the probe, and
fails with `RunnerUnavailable` when Axolotl is absent, at an unsupported version, or without a
visible CUDA device. The report lists `axolotl` beside `trl` and `stable-baselines3`, each with its
own version, failure code, and remedy.

The remedy names `T3RL_PYTHON_AXOLOTL` and the command that creates the environment, because the lab
never installs anything itself.

### Data flow

Unchanged. The manager spawns the experiment's entrypoint with `--run-dir`, `--seed`, and
`--config-json`, reads NDJSON on stdout, and treats stderr as human logs. Same protocol version,
same artifact kinds: manifest, summary, evaluation, replay, logs.

## Error handling

| Condition                       | Code                | Remedy                                            |
| ------------------------------- | ------------------- | ------------------------------------------------- |
| No interpreter found            | `PythonNotFound`    | Install Python or set the interpreter variable    |
| Axolotl absent or wrong version | `RunnerUnavailable` | Create the environment, set `T3RL_PYTHON_AXOLOTL` |
| No CUDA device visible          | `RunnerUnavailable` | The GRPO experiment requires a CUDA GPU           |

Codes are the ones the contract already defines. The adapter introduces no new error vocabulary.

## Testing

- `Capabilities.test.ts`: the Axolotl variable wins for the `axolotl` runner, is ignored for `trl`,
  and resolution falls back to the shared variable when it is unset; version and CUDA gating.
- `test_rlvr.py`: the extracted module, carrying over the existing evidence and verifier tests.
- `test_axolotl_worker.py`: config translation — `use_vllm` false, pinned model and revision, the
  split the experiment declares, and rejection of unknown or unbounded values.
- `test_trl_worker.py`: unchanged in behavior, proving the extraction preserved it.

All Python tests run without Axolotl installed, because framework imports are lazy.

Verification runs targeted: the touched test files plus lint and typecheck for the changed scope.

## Implementation risk

The Axolotl environment has not been created. Its pinned `torch==2.12.1` and `xformers==0.0.35` must
have wheels for the host CUDA runtime. Creating and probing that environment is the first
implementation step precisely so the risk lands before anything is built on it.

If it fails, this design does not change — the adapter boundary and the per-runner interpreter model
are independent of whether one particular Axolotl release installs — but the scope does, and that
becomes a decision to bring back rather than work around.

## Delivery order after this increment

Unchanged from the adapter document: checkpoint retention and resume, then an Accelerate launcher
with a two-GPU smoke fixture, then FSDP or DeepSpeed as strategies, then a separately supervised
vLLM sidecar.
