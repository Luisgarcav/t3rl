#!/usr/bin/env python3
"""Bounded Axolotl worker for the same GRPO/RLVR slice the TRL worker runs.

The task, the dataset, the verifier, the split policy, and the evidence
semantics come from `rlvr` and are therefore identical to the native TRL
adapter. Only the framework that performs the optimization differs, which is
what makes a comparison between the two runners attributable to the backend.

Axolotl pins exact dependency versions that no TRL adapter environment can
satisfy, so this worker runs from its own interpreter, selected by
`T3RL_PYTHON_AXOLOTL`. Framework imports stay lazy so configuration and
translation remain testable without that environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

import axolotl_reward

from rlvr import (
    EvidenceLedger,
    PROTOCOL_VERSION,
    SUPPORTED_DATASET,
    SUPPORTED_MODEL,
    SUPPORTED_MODEL_REVISION,
    _boolean,
    _float,
    _int,
    _string,
    emit,
    load_builtin_dataset,
    make_exact_integer_reward,
    split_dataset,
    write_json,
)

RUNNER_ID = "axolotl"
SUPPORTED_AXOLOTL = "0.18.0"

# Set once per process by `run`. The reward itself is installed on
# `axolotl_reward`, which Axolotl imports by path.
LEDGER: EvidenceLedger | None = None
TRAINER: Any = None

DEFAULT_CONFIG: dict[str, Any] = {
    "algorithm": "GRPO",
    "taskType": "llm-reasoning",
    "rewardSource": "verifiable",
    "backend": "axolotl",
    "launcher": "direct",
    "distributedStrategy": "single-process",
    "worldSize": 1,
    "modelId": SUPPORTED_MODEL,
    "modelRevision": SUPPORTED_MODEL_REVISION,
    "datasetId": SUPPORTED_DATASET,
    "systemPrompt": "Answer the arithmetic question. Return only the final integer.",
    "maxSteps": 8,
    "learningRate": 0.000005,
    "perDeviceTrainBatchSize": 2,
    "gradientAccumulationSteps": 1,
    "numGenerations": 2,
    "maxPromptLength": 128,
    "maxCompletionLength": 64,
    "temperature": 1.0,
    "beta": 0.001,
    "loggingSteps": 1,
    "evaluationRows": 4,
    "evaluationBatchSize": 2,
    "evaluationNumGenerations": 2,
    "retainSampleCount": 64,
    "maxGeneratedTokens": 2048,
    "maxWallClockSeconds": 1800,
    "maxGpuHours": 0.5,
    "gradientCheckpointing": True,
    "useVllm": False,
    "requireCuda": True,
}


def resolve_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("config must be a JSON object")
    unknown = sorted(set(raw) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    config = {**DEFAULT_CONFIG, **raw}

    for key, expected in {
        "algorithm": "GRPO",
        "taskType": "llm-reasoning",
        "rewardSource": "verifiable",
        "backend": "axolotl",
        "launcher": "direct",
        "distributedStrategy": "single-process",
        "modelId": SUPPORTED_MODEL,
        "datasetId": SUPPORTED_DATASET,
    }.items():
        if config[key] != expected:
            raise ValueError(f"{key} must be {expected}")

    config["modelRevision"] = _string("modelRevision", config["modelRevision"], 128)
    config["systemPrompt"] = _string("systemPrompt", config["systemPrompt"], 1024)
    for name, maximum in {
        "maxSteps": 10_000,
        "perDeviceTrainBatchSize": 128,
        "gradientAccumulationSteps": 1024,
        "numGenerations": 64,
        "maxPromptLength": 16_384,
        "maxCompletionLength": 16_384,
        "loggingSteps": 10_000,
        "evaluationRows": 1024,
        "evaluationBatchSize": 128,
        "evaluationNumGenerations": 64,
        "retainSampleCount": 256,
        "maxGeneratedTokens": 100_000_000,
        "maxWallClockSeconds": 86_400,
    }.items():
        config[name] = _int(name, config[name], 1, maximum)
    for name, minimum, maximum in [
        ("learningRate", 0.0, 1.0),
        ("temperature", 0.01, 10.0),
        ("beta", 0.0, 10.0),
        ("maxGpuHours", 0.001, 24.0),
    ]:
        config[name] = _float(name, config[name], minimum, maximum)
    config["worldSize"] = _int("worldSize", config["worldSize"], 1, 4096)
    if config["worldSize"] != 1:
        raise ValueError("the direct Axolotl adapter currently requires worldSize=1")
    for name in ["gradientCheckpointing", "useVllm", "requireCuda"]:
        config[name] = _boolean(name, config[name])
    if config["useVllm"]:
        # Axolotl's GRPO documentation presents vLLM as required; its schema
        # defaults `use_vllm` to false and guards every vLLM call behind it.
        # The sidecar is a later, separately supervised increment.
        raise ValueError("useVllm is not supported by the first Axolotl adapter")

    effective_batch = (
        config["perDeviceTrainBatchSize"] * config["gradientAccumulationSteps"]
    )
    if effective_batch % config["numGenerations"] != 0:
        raise ValueError(
            "perDeviceTrainBatchSize × gradientAccumulationSteps must be divisible by numGenerations"
        )
    training_token_bound = (
        config["maxSteps"]
        * config["perDeviceTrainBatchSize"]
        * config["gradientAccumulationSteps"]
        * config["maxCompletionLength"]
    )
    evaluation_token_bound = (
        2
        * config["evaluationRows"]
        * config["evaluationNumGenerations"]
        * config["maxCompletionLength"]
    )
    if training_token_bound + evaluation_token_bound > config["maxGeneratedTokens"]:
        raise ValueError(
            f"configured run may generate {training_token_bound + evaluation_token_bound} tokens, "
            f"above maxGeneratedTokens={config['maxGeneratedTokens']}"
        )
    if config["maxWallClockSeconds"] / 3600 > config["maxGpuHours"]:
        raise ValueError(
            "maxWallClockSeconds exceeds the declared single-GPU maxGpuHours budget"
        )
    return config


def build_axolotl_config(
    config: dict[str, Any],
    *,
    run_dir: str,
    train_path: str,
    eval_path: str,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Translates the bounded experiment config into an Axolotl config.

    Nothing is inferred: every value either comes from the resolved config or is
    a constant this slice deliberately fixes. `use_vllm` is one of those
    constants, and no `vllm` block is emitted at all.
    """
    dataset = {"path": train_path, "ds_type": "json", "split": "train"}
    return {
        "base_model": config["modelId"],
        "revision_of_model": config["modelRevision"],
        "rl": "grpo",
        "trl": {
            "use_vllm": False,
            "num_generations": config["numGenerations"],
            "max_completion_length": config["maxCompletionLength"],
            "temperature": config["temperature"],
            "beta": config["beta"],
            "reward_funcs": ["axolotl_reward.exact_integer_reward"],
        },
        "datasets": [dataset],
        "test_datasets": [{**dataset, "path": eval_path}],
        "max_steps": config["maxSteps"],
        "num_epochs": 1,
        # Axolotl prepares the model inside train(), so both evaluations are
        # driven by the trainer rather than called around it.
        "eval_on_start": True,
        "eval_strategy": "steps",
        "eval_steps": config["maxSteps"],
        # The verifier reads `answer` and `evidencePhase` off each row, so the
        # trainer must not drop columns it does not recognize itself.
        "remove_unused_columns": False,
        "learning_rate": config["learningRate"],
        "micro_batch_size": config["perDeviceTrainBatchSize"],
        "gradient_accumulation_steps": config["gradientAccumulationSteps"],
        "sequence_len": config["maxPromptLength"] + config["maxCompletionLength"],
        "gradient_checkpointing": config["gradientCheckpointing"],
        "logging_steps": config["loggingSteps"],
        "output_dir": os.path.join(run_dir, "axolotl-output"),
        # Axolotl caches its prepared dataset next to the config by default,
        # which would write outside the run directory the worker owns.
        "dataset_prepared_path": os.path.join(run_dir, "prepared"),
        "seed": seed,
        # This slice retains no checkpoint, matching the TRL adapter.
        "save_strategy": "no",
        "bf16": True,
    }


def dataset_rows(
    records: list[dict[str, str]], system_prompt: str, evidence_phase: str
) -> list[dict[str, Any]]:
    return [
        {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["prompt"]},
            ],
            "answer": row["answer"],
            "evidencePhase": evidence_phase,
        }
        for row in records
    ]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def current_phase(requested_phase: str) -> str:
    """
    Axolotl runs both evaluations inside `train()`, so the phase is read from
    the dataset column and the live optimizer step rather than from the call
    order. TRL 1.8 does not hand reward functions a trainer state.
    """
    if requested_phase != "evaluation":
        return "training"
    step = 0 if TRAINER is None else int(getattr(TRAINER.state, "global_step", 0))
    return "evaluation-before" if step == 0 else "evaluation-after"


def load_dependencies() -> dict[str, Any]:
    import torch
    from axolotl.cli.config import load_cfg
    from axolotl.common.datasets import load_preference_datasets
    from axolotl.train import setup_model_and_trainer
    from axolotl.utils.dict import DictDefault
    from transformers import TrainerCallback

    return {
        "torch": torch,
        "load_cfg": load_cfg,
        "load_preference_datasets": load_preference_datasets,
        "setup_model_and_trainer": setup_model_and_trainer,
        "DictDefault": DictDefault,
        "TrainerCallback": TrainerCallback,
    }


def run(args: argparse.Namespace) -> int:
    global LEDGER, TRAINER

    import hashlib
    import importlib.metadata as metadata
    import platform
    import random
    import time
    from contextlib import redirect_stdout

    started = time.monotonic()
    # Axolotl and its dependencies write warnings to stdout while importing, and
    # stdout is the versioned protocol stream. rlvr captured the real protocol
    # handle at import time, so emit() is unaffected by this redirect.
    with redirect_stdout(sys.stderr):
        deps = load_dependencies()
    runner_version = metadata.version("axolotl")
    emit(
        {
            "type": "hello",
            "protocol": PROTOCOL_VERSION,
            "runner": RUNNER_ID,
            "runnerVersion": runner_version,
        }
    )
    args.hello_sent = True

    config = resolve_config(args.config_json)
    torch = deps["torch"]
    if config["requireCuda"] and not torch.cuda.is_available():
        raise RuntimeError("this bundled GRPO experiment requires a CUDA GPU")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.run_dir, exist_ok=True)
    source_records, dataset_sha = load_builtin_dataset(config["datasetId"])
    training_records, evaluation_records = split_dataset(
        source_records, config["evaluationRows"]
    )
    train_path = os.path.join(args.run_dir, "train.jsonl")
    eval_path = os.path.join(args.run_dir, "eval.jsonl")
    write_jsonl(
        train_path, dataset_rows(training_records, config["systemPrompt"], "training")
    )
    write_jsonl(
        eval_path, dataset_rows(evaluation_records, config["systemPrompt"], "evaluation")
    )

    LEDGER = EvidenceLedger(config["retainSampleCount"])
    axolotl_reward.REWARD = make_exact_integer_reward(
        LEDGER, phase_resolver=current_phase
    )

    evidence = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "axolotl": runner_version,
        "trl": metadata.version("trl"),
        "transformers": metadata.version("transformers"),
        "torch": metadata.version("torch"),
        "accelerate": metadata.version("accelerate"),
    }
    evidence["fingerprint"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    emit(
        {
            "type": "manifest",
            "values": {
                **config,
                "seed": args.seed,
                "modelRevisionResolved": config["modelRevision"],
                "datasetSha256": dataset_sha,
                "datasetRows": len(source_records),
                "trainingRows": len(training_records),
                "evaluationRows": len(evaluation_records),
                "evaluationPolicy": "held-out-tail-v1; before-and-after; seeded-sampling",
                "verifierId": "exact-integer-v1",
                "dependencies": evidence,
                "execution": {
                    "adapter": "axolotl-direct-v1",
                    "backend": config["backend"],
                    "launcher": config["launcher"],
                    "distributedStrategy": config["distributedStrategy"],
                    "worldSize": config["worldSize"],
                },
            },
        }
    )

    class MetricsCallback(deps["TrainerCallback"]):  # type: ignore[misc]
        def on_log(self, callback_args, state, control, logs=None, **_):
            if not logs:
                return
            values = {
                f"{'eval' if key.startswith('eval_') else 'train'}/{key}": value
                for key, value in logs.items()
                if isinstance(value, (int, float))
            }
            if values:
                emit(
                    {
                        "type": "metrics",
                        "step": int(getattr(state, "global_step", 0)),
                        "wallClockMs": int((time.monotonic() - started) * 1000),
                        "values": values,
                    }
                )

    axolotl_config = build_axolotl_config(
        config,
        run_dir=args.run_dir,
        train_path=train_path,
        eval_path=eval_path,
        seed=args.seed,
    )
    write_json(args.run_dir, "axolotl-config.json", axolotl_config)
    emit({"type": "artifact", "kind": "config", "path": "axolotl-config.json"})

    # Axolotl and Transformers write human progress to stdout; stdout here is
    # the versioned worker protocol, so their output goes to stderr instead.
    with redirect_stdout(sys.stderr):
        cfg = deps["load_cfg"](deps["DictDefault"](axolotl_config))
        dataset_meta = deps["load_preference_datasets"](cfg=cfg)
        trainer, _model, _tokenizer, _peft, _processor = deps[
            "setup_model_and_trainer"
        ](cfg, dataset_meta)
        TRAINER = trainer
        trainer.add_callback(MetricsCallback())
        trainer.train()

    elapsed_ms = int((time.monotonic() - started) * 1000)
    training = LEDGER.summarize("training")
    evaluation_before = LEDGER.summarize("evaluation-before")
    evaluation_after = LEDGER.summarize("evaluation-after")
    before_rate = evaluation_before["verifierPassRate"]
    after_rate = evaluation_after["verifierPassRate"]
    delta = (
        float(after_rate) - float(before_rate)
        if isinstance(before_rate, (int, float)) and isinstance(after_rate, (int, float))
        else None
    )

    write_json(
        args.run_dir,
        "summary.json",
        {
            "algorithm": config["algorithm"],
            "runner": RUNNER_ID,
            "runnerVersion": runner_version,
            "seed": args.seed,
            "modelId": config["modelId"],
            "modelRevisionResolved": config["modelRevision"],
            "datasetId": config["datasetId"],
            "datasetSha256": dataset_sha,
            "verifierId": "exact-integer-v1",
            "optimizerSteps": config["maxSteps"],
            "trainingVerifierPassRate": training["verifierPassRate"],
            "evaluationBeforePassRate": before_rate,
            "evaluationAfterPassRate": after_rate,
            "evaluationPassRateDelta": delta,
            "retainedSamples": LEDGER.retained_count,
            "elapsedMs": elapsed_ms,
            "checkpointRetained": False,
            "budgets": {
                "maxGeneratedTokens": config["maxGeneratedTokens"],
                "maxWallClockSeconds": config["maxWallClockSeconds"],
                "maxGpuHours": config["maxGpuHours"],
            },
        },
    )
    emit({"type": "artifact", "kind": "summary", "path": "summary.json"})

    write_json(
        args.run_dir,
        "evaluation.json",
        {
            "kind": "llm-post-training-evaluation",
            "modelId": config["modelId"],
            "modelRevisionResolved": config["modelRevision"],
            "datasetId": config["datasetId"],
            "datasetSha256": dataset_sha,
            "policy": "held-out-tail-v1; before-and-after; seeded-sampling",
            "seed": args.seed,
            "before": evaluation_before,
            "after": evaluation_after,
            "verifierPassRateDelta": delta,
        },
    )
    emit({"type": "artifact", "kind": "evaluation", "path": "evaluation.json"})

    if LEDGER.samples:
        replay_samples = [
            sample
            for phase in ["evaluation-before", "evaluation-after", "training"]
            for sample in LEDGER.samples
            if sample.get("phase") == phase
        ]
        write_json(
            args.run_dir,
            "replay.json",
            {
                "kind": "llm-post-training",
                "environment": f"llm:{config['modelId']}",
                "evaluationSeed": args.seed,
                "trajectory": [
                    {
                        "step": index,
                        "observation": {
                            "phase": sample["phase"],
                            "prompt": sample["prompt"],
                            "expected": sample["expected"],
                        },
                        "action": {
                            "completion": sample["completion"],
                            "parsedAnswer": sample["parsedAnswer"],
                            "verifier": sample["verifier"],
                        },
                        "reward": sample["reward"],
                        "terminated": True,
                        "truncated": False,
                    }
                    for index, sample in enumerate(replay_samples)
                ],
            },
        )
        emit({"type": "artifact", "kind": "replay", "path": "replay.json"})

    emit({"type": "done", "status": "completed"})
    return 0


def probe() -> int:
    import importlib.metadata as metadata
    import platform

    import torch

    emit(
        {
            "type": "probe",
            "runner": RUNNER_ID,
            "protocolVersion": PROTOCOL_VERSION,
            "axolotl": metadata.version("axolotl"),
            "trl": metadata.version("trl"),
            "torch": metadata.version("torch"),
            "python": platform.python_version(),
            "cudaAvailable": torch.cuda.is_available(),
        }
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="T3RL Axolotl GRPO/RLVR worker")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config-json", type=json.loads, default={})
    args = parser.parse_args()
    args.hello_sent = False
    if args.probe:
        return probe()
    if not args.run_dir:
        parser.error("--run-dir is required unless --probe is used")
    try:
        return run(args)
    except Exception as error:  # noqa: BLE001 - report arbitrary runner failures.
        traceback.print_exc(file=sys.stderr)
        if args.hello_sent:
            emit(
                {
                    "type": "error",
                    "code": "RunnerException",
                    "detail": f"{type(error).__name__}: {str(error)[:1500]}",
                }
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
