#!/usr/bin/env python3
"""Bounded TRL worker for a first GRPO/RLVR post-training slice.

The worker intentionally keeps the experiment small and explicit: one model,
one versioned arithmetic dataset, one deterministic verifier, independently
loadable LoRA output, and exact trainer checkpoints. Optional ML dependencies
are imported lazily so configuration and reward tests remain runnable without
a GPU environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import sys
import threading
import time
import traceback
from contextlib import redirect_stdout
from importlib import metadata
from typing import Any

from checkpointing import (
    CheckpointPublisher,
    checkpoint_policy,
    compatibility,
    dataset_cursor,
    evaluate_before_training,
    model_identity,
    peft_config,
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

RUNNER_ID = "trl"

DEFAULT_CONFIG: dict[str, Any] = {
    "algorithm": "GRPO",
    "taskType": "llm-reasoning",
    "rewardSource": "verifiable",
    "backend": "trl",
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
        "backend": "trl",
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
        raise ValueError("the direct TRL adapter currently requires worldSize=1")
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
        raise ValueError("useVllm is not supported by the first local TRL adapter")
    if config["keepBest"]:
        raise ValueError(
            "keepBest requires a declared selection metric and is not enabled here"
        )
    if config["quantization"] != "none":
        raise ValueError("the current TRL adapter supports quantization=none")
    if config["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if config["loraBias"] not in {"none", "all", "lora_only"}:
        raise ValueError("loraBias must be none, all, or lora_only")
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
    generated_token_bound = training_token_bound + evaluation_token_bound
    if generated_token_bound > config["maxGeneratedTokens"]:
        raise ValueError(
            f"configured run may generate {generated_token_bound} tokens, above maxGeneratedTokens={config['maxGeneratedTokens']}"
        )
    if config["maxWallClockSeconds"] / 3600 > config["maxGpuHours"]:
        raise ValueError(
            "maxWallClockSeconds exceeds the declared single-GPU maxGpuHours budget"
        )
    return config


# Phases a run reports on, in the order they occur. The evidence budget is
# divided between them so a long training phase cannot spend the whole budget
# before the post-training evaluation runs.


def load_dependencies() -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    return {
        "torch": torch,
        "Dataset": Dataset,
        "HfApi": HfApi,
        "AutoProcessor": AutoProcessor,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "TrainerCallback": TrainerCallback,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }


def dependency_evidence(deps: dict[str, Any]) -> dict[str, Any]:
    torch = deps["torch"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "trl": metadata.version("trl"),
        "transformers": metadata.version("transformers"),
        "datasets": metadata.version("datasets"),
        "accelerate": metadata.version("accelerate"),
        "peft": metadata.version("peft"),
        "torch": metadata.version("torch"),
        "cudaAvailable": bool(torch.cuda.is_available()),
        "cudaDeviceCount": int(torch.cuda.device_count()),
        "cudaDevices": [
            str(torch.cuda.get_device_name(index))[:128]
            for index in range(torch.cuda.device_count())
        ],
        "cudaRuntime": str(torch.version.cuda),
        "cudnnVersion": int(torch.backends.cudnn.version() or 0),
    }


def probe() -> int:
    deps = load_dependencies()
    evidence = dependency_evidence(deps)
    evidence["executable"] = os.path.realpath(sys.executable)
    evidence["fingerprint"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


def resolve_model_revision(deps: dict[str, Any], model_id: str, revision: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40,64}", revision):
        return revision
    info = deps["HfApi"]().model_info(model_id, revision=revision)
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40,64}", resolved):
        raise RuntimeError("the model registry did not return an immutable revision")
    return resolved


def run(args: argparse.Namespace) -> int:
    runner_version = metadata.version("trl")
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
    with redirect_stdout(sys.stderr):
        deps = load_dependencies()
    config = resolve_config(args.config_json)
    torch = deps["torch"]
    if config["requireCuda"] and not torch.cuda.is_available():
        raise RuntimeError("this bundled GRPO experiment requires a CUDA GPU")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    resolved_revision = resolve_model_revision(
        deps, config["modelId"], config["modelRevision"]
    )
    resolved_tokenizer_revision = resolve_model_revision(
        deps, config["modelId"], config["tokenizerRevision"]
    )
    if config["precision"] == "bf16" and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("precision=bf16 requires a CUDA device with bf16 support")
    model = model_identity(
        config,
        resolved_revision=resolved_revision,
        tokenizer_revision=resolved_tokenizer_revision,
    )
    policy = checkpoint_policy(config)
    compatibility_evidence = compatibility(
        model=model, framework_id=RUNNER_ID, framework_version=runner_version
    )
    evidence = dependency_evidence(deps)
    evidence["fingerprint"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    emit(
        {
            "type": "manifest",
            "model": model,
            "checkpointPolicy": policy,
            "values": {
                **config,
                "seed": args.seed,
                "modelRevisionResolved": resolved_revision,
                "tokenizerRevisionResolved": resolved_tokenizer_revision,
                "chatTemplateSource": "model-repository",
                "datasetSha256": dataset_sha,
                "datasetRows": len(source_records),
                "trainingRows": len(training_records),
                "evaluationRows": len(evaluation_records),
                "evaluationPolicy": "held-out-tail-v1; before-and-after; seeded-sampling",
                "verifierId": VERIFIER_ID,
                "dependencies": evidence,
                "execution": {
                    "adapter": "trl-native-v1",
                    "backend": config["backend"],
                    "launcher": config["launcher"],
                    "distributedStrategy": config["distributedStrategy"],
                    "worldSize": config["worldSize"],
                },
                "determinismLimitations": [
                    "GPU kernels and sampled token generation may not be byte-identical across hardware or dependency fingerprints."
                ],
            },
        }
    )

    started = time.monotonic()
    os.makedirs(args.run_dir, exist_ok=True)
    publisher = CheckpointPublisher(
        run_dir=args.run_dir,
        compatibility_evidence=compatibility_evidence,
        policy=policy,
        emit_artifact=emit,
    )
    publisher.install_signal_handlers()
    ledger = EvidenceLedger(config["retainSampleCount"])
    latest_step = 0
    evaluation_passes = 0
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

    TrainerCallback = deps["TrainerCallback"]

    class MetricsCallback(TrainerCallback):
        def on_log(
            self,
            _training_args: Any,
            state: Any,
            _control: Any,
            logs: dict[str, Any] | None = None,
            **_kwargs: Any,
        ) -> None:
            nonlocal evaluation_passes, latest_step
            values = logs or {}
            latest_step = int(state.global_step)
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

    trainer_output_dir = os.path.join(args.run_dir, "trainer")

    def train_model() -> Any:
        def dataset_rows(
            records: list[dict[str, str]], evidence_phase: str
        ) -> list[dict[str, Any]]:
            return [
                {
                    "sampleId": row["sampleId"],
                    "prompt": [
                        {"role": "system", "content": config["systemPrompt"]},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    "answer": row["answer"],
                    "evidencePhase": evidence_phase,
                }
                for row in records
            ]

        train_rows = dataset_rows(training_records, "training")
        eval_rows = dataset_rows(evaluation_records, "evaluation")
        processing_class = deps["AutoProcessor"].from_pretrained(
            config["modelId"],
            revision=resolved_tokenizer_revision,
            trust_remote_code=False,
            truncation_side="left",
            padding_side="left",
        )
        for index, row in enumerate([*train_rows, *eval_rows]):
            token_ids = processing_class.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True
            )
            if len(token_ids) > config["maxPromptLength"]:
                raise ValueError(
                    f"dataset row {index} has {len(token_ids)} prompt tokens, above maxPromptLength={config['maxPromptLength']}"
                )
        train_dataset = deps["Dataset"].from_list(train_rows)
        eval_dataset = deps["Dataset"].from_list(eval_rows)
        training_kwargs = {
            "output_dir": trainer_output_dir,
            "max_steps": config["maxSteps"],
            "learning_rate": config["learningRate"],
            "per_device_train_batch_size": config["perDeviceTrainBatchSize"],
            "gradient_accumulation_steps": config["gradientAccumulationSteps"],
            "num_generations": config["numGenerations"],
            "max_completion_length": config["maxCompletionLength"],
            "temperature": config["temperature"],
            "beta": config["beta"],
            "logging_steps": config["loggingSteps"],
            "logging_first_step": True,
            "eval_strategy": "no",
            "per_device_eval_batch_size": config["evaluationBatchSize"],
            "num_generations_eval": config["evaluationNumGenerations"],
            "disable_tqdm": True,
            "save_strategy": "steps",
            "save_steps": config["checkpointCadenceSteps"],
            "save_total_limit": max(1, config["maxIntermediateCheckpoints"] + 2),
            "save_only_model": False,
            "report_to": "none",
            "seed": args.seed,
            "data_seed": args.seed,
            "bf16": config["precision"] == "bf16",
            "fp16": config["precision"] == "fp16",
            "gradient_checkpointing": config["gradientCheckpointing"],
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "use_vllm": False,
        }
        dtype_name = {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
        }[config["precision"]]
        if args.warm_start_adapter is None:
            training_kwargs["model_init_kwargs"] = {
                "revision": resolved_revision,
                "trust_remote_code": False,
                "dtype": dtype_name,
            }
        training_args = deps["GRPOConfig"](**training_kwargs)
        peft_values = peft_config(config)
        lora_config = deps["LoraConfig"](
            r=peft_values["rank"],
            lora_alpha=peft_values["alpha"],
            lora_dropout=peft_values["dropout"],
            bias=peft_values["bias"],
            target_modules=peft_values["targetModules"],
            modules_to_save=peft_values["modulesToSave"] or None,
            use_rslora=peft_values["useRslora"],
            task_type=peft_values["taskType"],
        )
        trainer_model: Any = config["modelId"]
        trainer_peft_config: Any = lora_config
        if args.warm_start_adapter is not None:
            base_model = deps["AutoModelForCausalLM"].from_pretrained(
                config["modelId"],
                revision=resolved_revision,
                trust_remote_code=False,
                dtype=dtype_name,
            )
            trainer_model = deps["PeftModel"].from_pretrained(
                base_model, args.warm_start_adapter, is_trainable=True
            )
            trainer_peft_config = None
        effective_batch_size = (
            config["perDeviceTrainBatchSize"]
            * config["gradientAccumulationSteps"]
            * config["worldSize"]
        )
        checkpoint_callback = transformers_checkpoint_callback(
            callback_base=deps["TrainerCallback"],
            publisher=publisher,
            trainer_output_dir=trainer_output_dir,
            max_steps=config["maxSteps"],
            effective_batch_size=effective_batch_size,
        )
        trainer = deps["GRPOTrainer"](
            model=trainer_model,
            reward_funcs=make_exact_integer_reward(ledger, verifier=project_verifier),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=[MetricsCallback(), checkpoint_callback],
            peft_config=trainer_peft_config,
        )
        evaluate_before_training(trainer, args.resume_checkpoint)
        result = trainer.train(resume_from_checkpoint=args.resume_checkpoint)
        if publisher.shutdown_requested.is_set():
            return result, trainer, checkpoint_callback
        trainer.evaluate()
        final_step = int(getattr(trainer.state, "global_step", latest_step))
        final_cursor = dataset_cursor(
            global_step=final_step,
            epoch=getattr(trainer.state, "epoch", None),
            effective_batch_size=effective_batch_size,
        )
        publisher.publish_adapter(
            save=lambda directory: trainer.model.save_pretrained(
                directory, safe_serialization=True
            ),
            global_step=final_step,
            tokens_seen=checkpoint_callback.tokens_seen,
            cursor=final_cursor,
        )
        return result, trainer, checkpoint_callback

    try:
        # stdout is the versioned worker protocol. Transformers and Trainer
        # write human-readable progress there, so keep their output on stderr
        # while emit() continues to use the original protocol stream.
        with redirect_stdout(sys.stderr):
            result, _trainer, _checkpoint_callback = train_model()
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2)
        shutil.rmtree(trainer_output_dir, ignore_errors=True)

    if publisher.shutdown_requested.is_set():
        return 0

    elapsed_ms = int((time.monotonic() - started) * 1000)
    training = ledger.summarize("training")
    evaluation_before = ledger.summarize("evaluation-before")
    evaluation_after = ledger.summarize("evaluation-after")
    before_pass_rate = evaluation_before["verifierPassRate"]
    after_pass_rate = evaluation_after["verifierPassRate"]
    evaluation_delta = (
        float(after_pass_rate) - float(before_pass_rate)
        if isinstance(before_pass_rate, (int, float))
        and isinstance(after_pass_rate, (int, float))
        else None
    )
    summary = {
        "runner": RUNNER_ID,
        "runnerVersion": runner_version,
        "algorithm": config["algorithm"],
        "modelId": config["modelId"],
        "modelRevisionResolved": resolved_revision,
        "datasetId": config["datasetId"],
        "datasetSha256": dataset_sha,
        "verifierId": VERIFIER_ID,
        "seed": args.seed,
        "optimizerSteps": int(getattr(result, "global_step", latest_step)),
        "retainedSamples": ledger.retained_count,
        "trainingVerifierPassRate": training["verifierPassRate"],
        "evaluationBeforePassRate": before_pass_rate,
        "evaluationAfterPassRate": after_pass_rate,
        "evaluationPassRateDelta": evaluation_delta,
        "elapsedMs": elapsed_ms,
        "checkpointRetained": config["keepFinal"],
        "adapterPublished": True,
        "budgets": {
            "maxGeneratedTokens": config["maxGeneratedTokens"],
            "maxWallClockSeconds": config["maxWallClockSeconds"],
            "maxGpuHours": config["maxGpuHours"],
        },
    }
    write_json(args.run_dir, "summary.json", summary)
    emit({"type": "artifact", "kind": "summary", "path": "summary.json"})
    evaluation = {
        "kind": "llm-post-training-evaluation",
        "modelId": config["modelId"],
        "modelRevisionResolved": resolved_revision,
        "datasetId": config["datasetId"],
        "datasetSha256": dataset_sha,
        "policy": "held-out-tail-v1; before-and-after; seeded-sampling",
        "seed": args.seed,
        "before": evaluation_before,
        "after": evaluation_after,
        "verifierPassRateDelta": evaluation_delta,
    }
    write_json(args.run_dir, "evaluation.json", evaluation)
    emit({"type": "artifact", "kind": "evaluation", "path": "evaluation.json"})
    if ledger.samples:
        replay_samples = order_replay_samples(ledger.samples)
        replay = {
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
        }
        write_json(args.run_dir, "replay.json", replay)
        emit({"type": "artifact", "kind": "replay", "path": "replay.json"})
    emit({"type": "done", "status": "completed"})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="T3RL TRL GRPO/RLVR worker")
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
