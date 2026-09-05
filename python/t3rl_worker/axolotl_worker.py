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
import shutil
import sys
import threading
import traceback
from typing import Any

import axolotl_reward
from checkpointing import (
    CheckpointPublisher,
    checkpoint_policy,
    compatibility,
    dataset_cursor,
    evaluate_before_training,
    model_identity,
    transformers_checkpoint_callback,
)
from rlvr import (
    PROTOCOL_VERSION,
    SUPPORTED_DATASET,
    SUPPORTED_DATASETS,
    SUPPORTED_MODEL,
    SUPPORTED_MODEL_REVISION,
    VERIFIER_ID,
    EvidenceLedger,
    _boolean,
    _float,
    _int,
    _string,
    atomic_write_bytes,
    emit,
    has_observed_metric,
    load_builtin_dataset,
    load_project_dataset,
    load_project_verifier,
    make_exact_integer_reward,
    normalize_grpo_metrics,
    order_replay_samples,
    run_metrics_heartbeat,
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
    "tokenizerRevision": SUPPORTED_MODEL_REVISION,
    "quantization": "none",
    "precision": "bf16",
    "loraRank": 16,
    "loraAlpha": 32.0,
    "loraDropout": 0.05,
    "loraBias": "none",
    "loraTargetModules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "loraModulesToSave": [],
    "useRslora": False,
    "checkpointCadenceSteps": 4,
    "maxIntermediateCheckpoints": 2,
    "keepBest": False,
    "keepFinal": True,
    "gracefulCheckpointDeadlineSeconds": 30,
    "datasetId": SUPPORTED_DATASET,
    "projectDatasetPath": None,
    "projectVerifierPath": None,
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
    }.items():
        if config[key] != expected:
            raise ValueError(f"{key} must be {expected}")
    if config["projectDatasetPath"] is None and config["datasetId"] not in SUPPORTED_DATASETS:
        raise ValueError(f"datasetId must be one of {sorted(SUPPORTED_DATASETS)}")
    for name in ("projectDatasetPath", "projectVerifierPath"):
        if config[name] is not None:
            config[name] = _string(name, config[name], 2048)

    config["modelRevision"] = _string("modelRevision", config["modelRevision"], 128)
    config["tokenizerRevision"] = _string(
        "tokenizerRevision", config["tokenizerRevision"], 128
    )
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
        "loraRank": 4096,
        "checkpointCadenceSteps": 1_000_000,
        "gracefulCheckpointDeadlineSeconds": 3600,
    }.items():
        config[name] = _int(name, config[name], 1, maximum)
    for name, minimum, maximum in [
        ("learningRate", 0.0, 1.0),
        ("temperature", 0.01, 10.0),
        ("beta", 0.0, 10.0),
        ("maxGpuHours", 0.001, 24.0),
        ("loraAlpha", 0.0, 1_000_000.0),
        ("loraDropout", 0.0, 1.0),
    ]:
        config[name] = _float(name, config[name], minimum, maximum)
    config["worldSize"] = _int("worldSize", config["worldSize"], 1, 4096)
    if config["worldSize"] != 1:
        raise ValueError("the direct Axolotl adapter currently requires worldSize=1")
    config["maxIntermediateCheckpoints"] = _int(
        "maxIntermediateCheckpoints", config["maxIntermediateCheckpoints"], 0, 64
    )
    for name in [
        "gradientCheckpointing",
        "useVllm",
        "requireCuda",
        "keepBest",
        "keepFinal",
        "useRslora",
    ]:
        config[name] = _boolean(name, config[name])
    if config["useVllm"]:
        # Axolotl's GRPO documentation presents vLLM as required; its schema
        # defaults `use_vllm` to false and guards every vLLM call behind it.
        # The sidecar is a later, separately supervised increment.
        raise ValueError("useVllm is not supported by the first Axolotl adapter")
    if config["keepBest"]:
        raise ValueError(
            "keepBest requires a declared selection metric and is not enabled here"
        )
    if config["quantization"] != "none":
        raise ValueError("the current Axolotl adapter supports quantization=none")
    if config["tokenizerRevision"] != config["modelRevision"]:
        raise ValueError("the Axolotl adapter requires tokenizerRevision=modelRevision")
    if config["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if config["loraBias"] != "none":
        raise ValueError("the current Axolotl adapter requires loraBias=none")
    for name, minimum in [("loraTargetModules", 1), ("loraModulesToSave", 0)]:
        value = config[name]
        if (
            not isinstance(value, list)
            or not minimum <= len(value) <= 128
            or any(
                not isinstance(module, str) or not module.strip() or len(module) > 128
                for module in value
            )
        ):
            raise ValueError(f"{name} must contain {minimum} to 128 module names")
        config[name] = [module.strip() for module in value]
    if config["keepFinal"] and config["maxSteps"] % config["checkpointCadenceSteps"]:
        raise ValueError(
            "checkpointCadenceSteps must divide maxSteps when keepFinal is true"
        )

    if config["evaluationNumGenerations"] != config["numGenerations"]:
        # Axolotl's schema exposes only `num_generations`. TRL's separate
        # `num_generations_eval` has no surface here, so a differing value would
        # be silently evaluated at the training count and the manifest would
        # claim a sample size the run never produced.
        raise ValueError(
            "the Axolotl adapter cannot evaluate at a different generation count "
            "than it trains at; set evaluationNumGenerations = numGenerations"
        )

    effective_batch = (
        config["perDeviceTrainBatchSize"] * config["gradientAccumulationSteps"]
    )
    if effective_batch % config["numGenerations"] != 0:
        raise ValueError(
            "perDeviceTrainBatchSize × gradientAccumulationSteps must be divisible by numGenerations"
        )
    if config["evaluationBatchSize"] % config["evaluationNumGenerations"] != 0:
        raise ValueError(
            "evaluationBatchSize must be divisible by evaluationNumGenerations"
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
    resume_checkpoint: str | None = None,
    warm_start_adapter: str | None = None,
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
        "adapter": "lora",
        "lora_r": config["loraRank"],
        "lora_alpha": config["loraAlpha"],
        "lora_dropout": config["loraDropout"],
        "lora_target_modules": config["loraTargetModules"],
        "lora_modules_to_save": config["loraModulesToSave"] or None,
        "peft_use_rslora": config["useRslora"],
        "lora_model_dir": warm_start_adapter,
        "resume_from_checkpoint": resume_checkpoint,
        "load_in_4bit": False,
        "load_in_8bit": False,
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
        # The worker performs the starting evaluation after setup so an exact
        # resume can preload checkpoint weights before measuring its baseline.
        "eval_on_start": False,
        "eval_strategy": "steps",
        "eval_steps": config["maxSteps"],
        # The verifier reads `answer` and `evidencePhase` off each row, so the
        # trainer must not drop columns it does not recognize itself.
        "remove_unused_columns": False,
        "learning_rate": config["learningRate"],
        "micro_batch_size": config["perDeviceTrainBatchSize"],
        # Without this the backend picks its own evaluation batch size and the
        # manifest would declare a bound the run does not honour.
        "eval_batch_size": config["evaluationBatchSize"],
        "gradient_accumulation_steps": config["gradientAccumulationSteps"],
        "sequence_len": config["maxPromptLength"] + config["maxCompletionLength"],
        "gradient_checkpointing": config["gradientCheckpointing"],
        "logging_steps": config["loggingSteps"],
        "output_dir": os.path.join(run_dir, "axolotl-output"),
        # Axolotl caches its prepared dataset next to the config by default,
        # which would write outside the run directory the worker owns.
        "dataset_prepared_path": os.path.join(run_dir, "prepared"),
        "seed": seed,
        "save_strategy": "steps",
        "save_steps": config["checkpointCadenceSteps"],
        "save_total_limit": max(1, config["maxIntermediateCheckpoints"] + 2),
        "save_only_model": False,
        "save_safetensors": True,
        "bf16": config["precision"] == "bf16",
        "fp16": config["precision"] == "fp16",
    }


def dataset_rows(
    records: list[dict[str, str]], system_prompt: str, evidence_phase: str
) -> list[dict[str, Any]]:
    return [
        {
            "sampleId": row["sampleId"],
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
    contents = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    atomic_write_bytes(path, contents.encode())


def current_phase(requested_phase: str) -> str:
    """
    Axolotl's reward bridge reads the dataset column and live optimizer step
    rather than relying on evaluation call order. TRL does not hand reward
    functions a trainer state through this integration.
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
    import platform
    import random
    import time
    from contextlib import redirect_stdout
    from importlib import metadata

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
    if args.resume_checkpoint is not None and args.warm_start_adapter is not None:
        raise ValueError(
            "resume checkpoint and warm-start adapter are mutually exclusive"
        )

    started = time.monotonic()
    # Axolotl and its dependencies write warnings to stdout while importing, and
    # stdout is the versioned protocol stream. rlvr captured the real protocol
    # handle at import time, so emit() is unaffected by this redirect.
    with redirect_stdout(sys.stderr):
        deps = load_dependencies()

    config = resolve_config(args.config_json)
    torch = deps["torch"]
    if config["requireCuda"] and not torch.cuda.is_available():
        raise RuntimeError("this bundled GRPO experiment requires a CUDA GPU")
    if config["precision"] == "bf16" and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("precision=bf16 requires a CUDA device with bf16 support")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.run_dir, exist_ok=True)
    source_records, dataset_sha = (
        load_project_dataset(config["projectDatasetPath"])
        if config["projectDatasetPath"] is not None
        else load_builtin_dataset(config["datasetId"])
    )
    project_verifier = (
        load_project_verifier(config["projectVerifierPath"])
        if config["projectVerifierPath"] is not None
        else None
    )
    training_records, evaluation_records = split_dataset(
        source_records, config["evaluationRows"]
    )
    train_path = os.path.join(args.run_dir, "train.jsonl")
    eval_path = os.path.join(args.run_dir, "eval.jsonl")
    write_jsonl(
        train_path, dataset_rows(training_records, config["systemPrompt"], "training")
    )
    write_jsonl(
        eval_path,
        dataset_rows(evaluation_records, config["systemPrompt"], "evaluation"),
    )

    LEDGER = EvidenceLedger(config["retainSampleCount"])
    axolotl_reward.REWARD = make_exact_integer_reward(
        LEDGER, phase_resolver=current_phase, verifier=project_verifier
    )

    evidence = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "axolotl": runner_version,
        "trl": metadata.version("trl"),
        "transformers": metadata.version("transformers"),
        "torch": metadata.version("torch"),
        "accelerate": metadata.version("accelerate"),
        "peft": metadata.version("peft"),
    }
    evidence["fingerprint"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    model = model_identity(
        config,
        resolved_revision=config["modelRevision"],
        tokenizer_revision=config["tokenizerRevision"],
    )
    policy = checkpoint_policy(config)
    compatibility_evidence = compatibility(
        model=model, framework_id=RUNNER_ID, framework_version=runner_version
    )
    emit(
        {
            "type": "manifest",
            "model": model,
            "checkpointPolicy": policy,
            "values": {
                **config,
                "seed": args.seed,
                "modelRevisionResolved": config["modelRevision"],
                "tokenizerRevisionResolved": config["tokenizerRevision"],
                "datasetSha256": dataset_sha,
                "datasetRows": len(source_records),
                "trainingRows": len(training_records),
                "evaluationRows": len(evaluation_records),
                "evaluationPolicy": "held-out-tail-v1; before-and-after; seeded-sampling",
                "verifierId": VERIFIER_ID,
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

    latest_step = 0
    evaluation_passes = 0
    publisher = CheckpointPublisher(
        run_dir=args.run_dir,
        compatibility_evidence=compatibility_evidence,
        policy=policy,
        emit_artifact=emit,
    )
    publisher.install_signal_handlers()

    class MetricsCallback(deps["TrainerCallback"]):  # type: ignore[misc]
        def on_log(self, callback_args, state, control, logs=None, **_):
            nonlocal evaluation_passes, latest_step
            values = logs or {}
            latest_step = int(getattr(state, "global_step", 0))
            if any(key.startswith("eval_") for key in values):
                evaluation_passes += 1
            metrics = normalize_grpo_metrics(
                values,
                step=latest_step,
                evaluation_passes=evaluation_passes,
                config=config,
                elapsed_seconds=time.monotonic() - started,
                gpu_memory_allocated_gb=(
                    float(torch.cuda.memory_allocated() / (1024**3))
                    if torch.cuda.is_available()
                    else 0.0
                ),
            )
            if not has_observed_metric(metrics):
                return
            emit(
                {
                    "type": "metrics",
                    "step": latest_step,
                    "wallClockMs": int((time.monotonic() - started) * 1000),
                    "values": metrics,
                }
            )

    axolotl_config = build_axolotl_config(
        config,
        run_dir=args.run_dir,
        train_path=train_path,
        eval_path=eval_path,
        seed=args.seed,
        resume_checkpoint=args.resume_checkpoint,
        warm_start_adapter=args.warm_start_adapter,
    )
    write_json(args.run_dir, "axolotl-config.json", axolotl_config)
    emit({"type": "artifact", "kind": "config", "path": "axolotl-config.json"})

    stop_heartbeat = threading.Event()
    heartbeat_thread = threading.Thread(
        target=lambda: run_metrics_heartbeat(
            stop_heartbeat,
            started=started,
            step=lambda: latest_step,
            gpu_count=torch.cuda.device_count,
        ),
        name="t3rl-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
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
            checkpoint_callback = transformers_checkpoint_callback(
                callback_base=deps["TrainerCallback"],
                publisher=publisher,
                trainer_output_dir=axolotl_config["output_dir"],
                max_steps=config["maxSteps"],
                effective_batch_size=(
                    config["perDeviceTrainBatchSize"]
                    * config["gradientAccumulationSteps"]
                    * config["worldSize"]
                ),
            )
            trainer.add_callback(checkpoint_callback)
            evaluate_before_training(trainer, args.resume_checkpoint)
            trainer.train(resume_from_checkpoint=args.resume_checkpoint)
            if not publisher.shutdown_requested.is_set():
                final_step = int(getattr(trainer.state, "global_step", latest_step))
                final_cursor = dataset_cursor(
                    global_step=final_step,
                    epoch=getattr(trainer.state, "epoch", None),
                    effective_batch_size=(
                        config["perDeviceTrainBatchSize"]
                        * config["gradientAccumulationSteps"]
                        * config["worldSize"]
                    ),
                )
                publisher.publish_adapter(
                    save=lambda directory: trainer.model.save_pretrained(
                        directory, safe_serialization=True
                    ),
                    global_step=final_step,
                    tokens_seen=checkpoint_callback.tokens_seen,
                    cursor=final_cursor,
                )
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2)
        shutil.rmtree(axolotl_config["output_dir"], ignore_errors=True)
        shutil.rmtree(axolotl_config["dataset_prepared_path"], ignore_errors=True)

    if publisher.shutdown_requested.is_set():
        return 0

    elapsed_ms = int((time.monotonic() - started) * 1000)
    training = LEDGER.summarize("training")
    evaluation_before = LEDGER.summarize("evaluation-before")
    evaluation_after = LEDGER.summarize("evaluation-after")
    before_rate = evaluation_before["verifierPassRate"]
    after_rate = evaluation_after["verifierPassRate"]
    delta = (
        float(after_rate) - float(before_rate)
        if isinstance(before_rate, (int, float))
        and isinstance(after_rate, (int, float))
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
            "verifierId": VERIFIER_ID,
            "optimizerSteps": config["maxSteps"],
            "trainingVerifierPassRate": training["verifierPassRate"],
            "evaluationBeforePassRate": before_rate,
            "evaluationAfterPassRate": after_rate,
            "evaluationPassRateDelta": delta,
            "retainedSamples": LEDGER.retained_count,
            "elapsedMs": elapsed_ms,
            "checkpointRetained": config["keepFinal"],
            "adapterPublished": True,
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
        replay_samples = order_replay_samples(LEDGER.samples)
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
                            "sampleId": sample["sampleId"],
                            "generationIndex": sample["generationIndex"],
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
    import platform
    from importlib import metadata

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
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--warm-start-adapter")
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
