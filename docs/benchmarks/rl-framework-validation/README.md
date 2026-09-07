# Real framework validation

[The recorded result](2026-09-05-linux-x64.json) exercises native TRL and Axolotl training on
Linux x64 with an NVIDIA RTX 4060 Laptop GPU (8,188 MiB, driver 610.57.04). Each framework was
installed into a separate disposable virtual environment using its checked-in lock. The
application's selected Python interpreter and live T3 state were not changed.

The SFT/DPO case creates a deterministic 13,024-parameter Llama model and a local WordLevel
tokenizer. It trains on eight arithmetic records and evaluates two disjoint held-out records.
TRL uses three seeds (7, 19, 41) on CUDA and seed 7 on CPU. Axolotl uses training seed 7 and data
seed 19 on CUDA. Every
method/device/seed case completes six optimizer steps, reloads its PEFT adapter independently,
interrupts a separate run with SIGTERM after a checkpoint receipt, and resumes its complete
trainer state. Final adapter parameters and held-out loss must agree with the uninterrupted run
within absolute tolerance `1e-6` and relative tolerance `1e-5`. All recorded parameter differences
were zero. Only the original three-seed TRL CUDA reference matrix includes zero-learning-rate
ablations, one for each method and seed; their losses must remain unchanged. The later checks with
distinct training/data seeds do not include ablations.

The three-seed reference matrix used equal training/data seeds. Its exact source snapshot was
retained before adding independent seed transport. The final code was then checked separately
with training seed 7 and data seed 19 on TRL CPU/CUDA and Axolotl CUDA. The workers apply the data
seed to the training sampler and record that evaluation/generation seeds are unused by fixed
held-out, teacher-forced evaluation. The result identifies the source snapshot for each suite.

Each completed worker writes per-sample evaluation evidence whose mean is checked against the
reported scalar. The protocol freezes dataset bytes, held-out IDs and order, deterministic
evaluation, evaluator source hashes, method, tokenization/truncation settings, precision, and
framework versions. A requested study protocol that differs is rejected before training.

The separate native GRPO case uses the pinned `Qwen/Qwen2.5-0.5B-Instruct` revision and the bundled
`arithmetic-rlvr-v2` dataset. Four optimizer steps produced nonzero gradients, full checkpoints,
an independently loaded adapter, evaluation, and replay. Resuming at step 2 produced the same
final parameters. Held-out verifier pass rate changed from `0.5` to `0.4375`; this negative
observation is retained. One seed and 16 held-out generations do not support a model-quality
claim.

## Reproduce the bounded SFT/DPO cases

Run from the repository root with `uv` and a CUDA-capable host. Use fresh output directories for
each invocation; the harness refuses to overwrite an existing model or run.

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venv-trl-validation" uv sync --project python/environments/trl --frozen --python 3.12
UV_PROJECT_ENVIRONMENT="$PWD/.venv-axolotl-validation" uv sync --project python/environments/axolotl --frozen --python 3.12

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-trl-validation/bin/python python/t3rl_worker/offline_validation.py --device cuda --output-dir .t3/serious-validation/trl-gpu --seeds 7 19 41 --ablation
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-trl-validation/bin/python python/t3rl_worker/offline_validation.py --device cpu --output-dir .t3/serious-validation/trl-cpu --seeds 7
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-trl-validation/bin/python python/t3rl_worker/offline_validation.py --device cpu --output-dir .t3/serious-validation/seed-set-trl-cpu --seeds 7 --data-seed-offset 12
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-trl-validation/bin/python python/t3rl_worker/offline_validation.py --device cuda --output-dir .t3/serious-validation/seed-set-trl-gpu --seeds 7 --data-seed-offset 12
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-axolotl-validation/bin/python python/t3rl_worker/offline_validation.py --backend axolotl --device cuda --output-dir .t3/serious-validation/axolotl-gpu --seeds 7 --data-seed-offset 12
```

The harness requires the installed core framework versions to match the selected lock. It
disables Hub network access, writes a local model instead of downloading weights, and records
the trainer's actual device and CUDA allocations. Each output directory retains the model,
datasets, source snapshots, NDJSON protocol, stderr logs, checkpoints, adapters, sample evaluations, and
`validation-report.json`. A worker error, missing state file, non-JSON protocol line, missing
sample, or resume mismatch fails the command.

The same bounded cases back `test_offline_gpu_smoke.py`. Set `T3RL_RUN_GPU_SMOKE=1` for native
TRL CUDA, `T3RL_RUN_FRAMEWORK_SMOKE=1` for native TRL CPU, or
`T3RL_RUN_AXOLOTL_GPU_SMOKE=1` for Axolotl CUDA and run that file with its matching interpreter.
An ordinary test run reports these cases as skipped; a skip is not GPU evidence.

## Reproduce the native GRPO case

The exact configuration is retained under `grpo.config` in the result JSON. The pinned Qwen model
must already be cached, or be fetched explicitly before an offline run. This command starts only
the worker and stores all output under its new run directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-trl-validation/bin/python - <<'PY'
import json
import os
import pathlib
import subprocess
import sys

record = json.loads(pathlib.Path("docs/benchmarks/rl-framework-validation/2026-09-05-linux-x64.json").read_text())
run_dir = pathlib.Path(".t3/serious-validation/grpo-reproduction").resolve()
run_dir.mkdir(parents=True)
with (run_dir / "protocol.ndjson").open("w") as out, (run_dir / "stderr.log").open("w") as err:
    subprocess.run([
        sys.executable, "python/t3rl_worker/trl_worker.py", "--run-dir", str(run_dir),
        "--seed", "7", "--config-json", json.dumps(record["grpo"]["config"]),
    ], stdout=out, stderr=err, check=True, timeout=240,
        env={**os.environ, "T3RL_ENVIRONMENT_LOCK_SHA256": record["grpo"]["environmentLockSha256"]})
PY
```

To check continuation, start a child run with the same seed/config and
`--resume-checkpoint <parent>/checkpoints/checkpoint-2-intermediate`, then independently load
the two final `adapter_model.safetensors` files and compare corresponding tensors using the
declared tolerances. The checked-in result records the verification performed for this run.

## Limits

These are real framework lifecycle checks on one host. The tiny SFT/DPO model and two held-out
records establish neither useful arithmetic ability nor statistically defensible improvement.
CPU and GPU are each checked against their own uninterrupted runs; cross-device equality is not
asserted. A fresh virtual environment on the same machine does not establish reproduction by
another researcher, on a clean machine, or on another GPU. Distributed training, vLLM, DeepSpeed,
Axolotl GRPO, and full model training beyond LoRA remain outside this evidence.

The committed JSON is a compact, path-sanitized record. Full local evidence lives in the
gitignored run directories. A portable bundle must include the generated tiny base model and
tokenizer in addition to its adapter; local model paths must be remapped when loading elsewhere.
