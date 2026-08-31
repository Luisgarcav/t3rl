#!/usr/bin/env python3
"""Bounded TRL worker for a first GRPO/RLVR post-training slice.

The worker intentionally keeps the experiment small and explicit: one model,
one versioned arithmetic dataset, one deterministic verifier, and no retained
checkpoint. Optional ML dependencies are imported lazily so configuration and
reward tests remain runnable without a GPU environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import redirect_stdout
from importlib import metadata
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
RUNNER_ID = "trl"
SUPPORTED_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SUPPORTED_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SUPPORTED_DATASET = "arithmetic-rlvr-v1"
DATASET_PATHS = {
    SUPPORTED_DATASET: Path(__file__).parent / "datasets" / "arithmetic-rlvr-v1.json"
}
INTEGER_PATTERN = re.compile(r"(?<![\d.])[+-]?\d[\d,]*(?![\d.])")
EMIT_LOCK = threading.Lock()
PROTOCOL_STDOUT = sys.stdout

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


def emit(message: dict[str, Any]) -> None:
    """Write one complete NDJSON message even when the heartbeat is active."""
    line = json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n"
    with EMIT_LOCK:
        PROTOCOL_STDOUT.write(line)
        PROTOCOL_STDOUT.flush()


def finite_metric(value: Any) -> float | str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    return number


def _int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _float(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
    return result


def _string(name: str, value: Any, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{name} must be a non-empty string of at most {maximum} characters"
        )
    return value.strip()


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


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
        raise ValueError("the direct TRL adapter currently requires worldSize=1")
    for name in ["gradientCheckpointing", "useVllm", "requireCuda"]:
        config[name] = _boolean(name, config[name])
    if config["useVllm"]:
        raise ValueError("useVllm is not supported by the first local TRL adapter")

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


def extract_final_integer(value: str) -> str | None:
    matches = INTEGER_PATTERN.findall(value)
    if not matches:
        return None
    try:
        return str(int(matches[-1].replace(",", "")))
    except ValueError:
        return None


def completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return (
            content if isinstance(content, str) else json.dumps(value, sort_keys=True)
        )
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                return item["content"]
        return json.dumps(value, sort_keys=True)
    return str(value)


def load_builtin_dataset(dataset_id: str) -> tuple[list[dict[str, str]], str]:
    path = DATASET_PATHS.get(dataset_id)
    if path is None:
        raise ValueError(f"unsupported datasetId: {dataset_id}")
    payload = path.read_bytes()
    decoded = json.loads(payload)
    if not isinstance(decoded, list) or not decoded:
        raise ValueError(f"dataset {dataset_id} must be a non-empty JSON array")
    records: list[dict[str, str]] = []
    for index, value in enumerate(decoded):
        if not isinstance(value, dict):
            raise TypeError(f"dataset row {index} must be an object")
        prompt = _string(f"dataset row {index} prompt", value.get("prompt"), 1024)
        answer = _string(f"dataset row {index} answer", value.get("answer"), 64)
        normalized = extract_final_integer(answer)
        if normalized is None or normalized != answer:
            raise ValueError(f"dataset row {index} answer must be a canonical integer")
        records.append({"prompt": prompt, "answer": answer})
    return records, hashlib.sha256(payload).hexdigest()


def split_dataset(
    records: list[dict[str, str]], evaluation_rows: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < evaluation_rows < len(records):
        raise ValueError(
            "evaluationRows must leave at least one training and one evaluation row"
        )
    return records[:-evaluation_rows], records[-evaluation_rows:]


def make_exact_integer_reward(
    retained_samples: list[dict[str, Any]], retain_limit: int
) -> Callable[..., list[float]]:
    def exact_integer_reward(
        completions: list[Any],
        answer: list[str],
        prompts: list[Any] | None = None,
        evidencePhase: list[str] | None = None,
        trainer_state: Any = None,
        **_: Any,
    ) -> list[float]:
        if len(completions) != len(answer):
            raise ValueError("completions and answers must have the same length")
        rewards: list[float] = []
        for index, (completion, expected) in enumerate(zip(completions, answer)):
            text = completion_text(completion)
            parsed = extract_final_integer(text)
            reward = 1.0 if parsed == expected else 0.0
            rewards.append(reward)
            if len(retained_samples) < retain_limit:
                prompt = (
                    prompts[index]
                    if prompts is not None and index < len(prompts)
                    else ""
                )
                requested_phase = (
                    evidencePhase[index]
                    if evidencePhase is not None and index < len(evidencePhase)
                    else "training"
                )
                phase = (
                    "evaluation-before"
                    if requested_phase == "evaluation"
                    and int(getattr(trainer_state, "global_step", 0)) == 0
                    else "evaluation-after"
                    if requested_phase == "evaluation"
                    else "training"
                )
                retained_samples.append(
                    {
                        "phase": phase,
                        "prompt": completion_text(prompt),
                        "expected": expected,
                        "completion": text,
                        "parsedAnswer": parsed,
                        "reward": reward,
                        "verifier": {
                            "id": "exact-integer-v1",
                            "passed": reward == 1.0,
                        },
                    }
                )
        return rewards

    return exact_integer_reward


def summarize_samples(
    retained_samples: list[dict[str, Any]], phase: str
) -> dict[str, Any]:
    samples = [sample for sample in retained_samples if sample.get("phase") == phase]
    rewards = [float(sample["reward"]) for sample in samples]
    return {
        "phase": phase,
        "sampleCount": len(samples),
        "rewardMean": statistics.fmean(rewards) if rewards else None,
        "rewardStd": statistics.pstdev(rewards) if rewards else None,
        "verifierPassRate": sum(reward == 1.0 for reward in rewards) / len(rewards)
        if rewards
        else None,
        "samples": samples,
    }


def write_json(run_dir: str, relative_path: str, value: Any) -> None:
    target = os.path.join(run_dir, relative_path)
    os.makedirs(os.path.dirname(target) or run_dir, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_dependencies() -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    return {
        "torch": torch,
        "Dataset": Dataset,
        "HfApi": HfApi,
        "AutoProcessor": AutoProcessor,
        "TrainerCallback": TrainerCallback,
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


def metric_value(logs: dict[str, Any], *keys: str) -> float | str | None:
    for key in keys:
        if key in logs:
            return finite_metric(logs[key])
    return None


def has_observed_metric(metrics: dict[str, Any]) -> bool:
    """Exclude Trainer summaries that contain no normalized training or eval data."""
    return any(
        key.startswith(("train/", "eval/")) and value is not None
        for key, value in metrics.items()
    )


def run(args: argparse.Namespace) -> int:
    deps = load_dependencies()
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
    config = resolve_config(args.config_json)
    torch = deps["torch"]
    if config["requireCuda"] and not torch.cuda.is_available():
        raise RuntimeError("this bundled GRPO experiment requires a CUDA GPU")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source_records, dataset_sha = load_builtin_dataset(config["datasetId"])
    training_records, evaluation_records = split_dataset(
        source_records, config["evaluationRows"]
    )
    resolved_revision = resolve_model_revision(
        deps, config["modelId"], config["modelRevision"]
    )
    evidence = dependency_evidence(deps)
    evidence["fingerprint"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    emit(
        {
            "type": "manifest",
            "values": {
                **config,
                "seed": args.seed,
                "modelRevisionResolved": resolved_revision,
                "tokenizerRevisionResolved": resolved_revision,
                "chatTemplateSource": "model-repository",
                "datasetSha256": dataset_sha,
                "datasetRows": len(source_records),
                "trainingRows": len(training_records),
                "evaluationRows": len(evaluation_records),
                "evaluationPolicy": "held-out-tail-v1; before-and-after; seeded-sampling",
                "verifierId": "exact-integer-v1",
                "dependencies": evidence,
                "execution": {
                    "adapter": "trl-native-v1",
                    "backend": config["backend"],
                    "launcher": config["launcher"],
                    "distributedStrategy": config["distributedStrategy"],
                    "worldSize": config["worldSize"],
                },
                "checkpointRetention": "none",
                "determinismLimitations": [
                    "GPU kernels and sampled token generation may not be byte-identical across hardware or dependency fingerprints."
                ],
            },
        }
    )

    started = time.monotonic()
    os.makedirs(args.run_dir, exist_ok=True)
    retained_samples: list[dict[str, Any]] = []
    latest_step = 0
    evaluation_passes = 0
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(15):
            emit(
                {
                    "type": "metrics",
                    "step": latest_step,
                    "wallClockMs": int((time.monotonic() - started) * 1000),
                    "values": {
                        "system/heartbeat": 1.0,
                        "system/gpu_count": float(torch.cuda.device_count()),
                    },
                }
            )

    heartbeat_thread = threading.Thread(
        target=heartbeat, name="t3rl-heartbeat", daemon=True
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
            is_evaluation = any(key.startswith("eval_") for key in values)
            if is_evaluation:
                evaluation_passes += 1
                metrics = {
                    "eval/reward": metric_value(values, "eval_reward"),
                    "eval/reward_std": metric_value(values, "eval_reward_std"),
                    "eval/verifier_pass_rate": metric_value(
                        values,
                        "eval_rewards/exact_integer_reward/mean",
                        "eval_reward",
                    ),
                    "eval/completion_length": metric_value(
                        values, "eval_completions/mean_length"
                    ),
                    "eval/kl": metric_value(values, "eval_kl"),
                    "eval/entropy": metric_value(values, "eval_entropy"),
                }
            else:
                reward = metric_value(
                    values, "reward", "rewards/exact_integer_reward/mean"
                )
                metrics = {
                    "train/reward": reward,
                    "train/reward_std": metric_value(
                        values,
                        "reward_std",
                        "rewards/exact_integer_reward/std",
                    ),
                    "train/verifier_pass_rate": metric_value(
                        values,
                        "rewards/exact_integer_reward/mean",
                        "rewards/exact_integer_reward",
                        "reward",
                    ),
                    "train/kl": metric_value(values, "kl"),
                    "train/approx_kl": metric_value(values, "kl"),
                    "train/entropy": metric_value(values, "entropy"),
                    "train/completion_length": metric_value(
                        values, "completions/mean_length", "completion_length"
                    ),
                    "train/loss": metric_value(values, "policy_loss", "loss"),
                    "train/policy_loss": metric_value(values, "policy_loss", "loss"),
                    "train/learning_rate": metric_value(values, "learning_rate"),
                    "train/grad_norm": metric_value(values, "grad_norm"),
                    "system/num_tokens": metric_value(values, "num_tokens"),
                    "system/step_time_seconds": metric_value(values, "step_time"),
                    "system/gpu_memory_allocated_gb": float(
                        torch.cuda.memory_allocated() / (1024**3)
                    )
                    if torch.cuda.is_available()
                    else 0.0,
                }
            completed_training_bound = (
                latest_step
                * config["perDeviceTrainBatchSize"]
                * config["gradientAccumulationSteps"]
                * config["maxCompletionLength"]
            )
            completed_evaluation_bound = (
                evaluation_passes
                * config["evaluationRows"]
                * config["evaluationNumGenerations"]
                * config["maxCompletionLength"]
            )
            metrics["system/generated_tokens_upper_bound"] = float(
                completed_training_bound + completed_evaluation_bound
            )
            elapsed_seconds = max(time.monotonic() - started, 1e-9)
            observed_tokens = metric_value(values, "num_tokens", "eval_num_tokens")
            metrics["system/tokens_per_second"] = (
                float(observed_tokens) / elapsed_seconds
                if isinstance(observed_tokens, (int, float))
                else None
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

    def train_model() -> Any:
        def dataset_rows(
            records: list[dict[str, str]], evidence_phase: str
        ) -> list[dict[str, Any]]:
            return [
                {
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
            revision=resolved_revision,
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
        bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        training_args = deps["GRPOConfig"](
            output_dir=os.path.join(args.run_dir, "trainer"),
            max_steps=config["maxSteps"],
            learning_rate=config["learningRate"],
            per_device_train_batch_size=config["perDeviceTrainBatchSize"],
            gradient_accumulation_steps=config["gradientAccumulationSteps"],
            num_generations=config["numGenerations"],
            max_completion_length=config["maxCompletionLength"],
            temperature=config["temperature"],
            beta=config["beta"],
            logging_steps=config["loggingSteps"],
            logging_first_step=True,
            eval_strategy="no",
            per_device_eval_batch_size=config["evaluationBatchSize"],
            num_generations_eval=config["evaluationNumGenerations"],
            disable_tqdm=True,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            bf16=bf16,
            fp16=bool(torch.cuda.is_available() and not bf16),
            gradient_checkpointing=config["gradientCheckpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            use_vllm=False,
            model_init_kwargs={
                "revision": resolved_revision,
                "trust_remote_code": False,
                "dtype": "auto",
            },
        )
        trainer = deps["GRPOTrainer"](
            model=config["modelId"],
            reward_funcs=make_exact_integer_reward(
                retained_samples, config["retainSampleCount"]
            ),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=[MetricsCallback()],
        )
        trainer.evaluate()
        result = trainer.train()
        trainer.evaluate()
        return result

    try:
        # stdout is the versioned worker protocol. Transformers and Trainer
        # write human-readable progress there, so keep their output on stderr
        # while emit() continues to use the original protocol stream.
        with redirect_stdout(sys.stderr):
            result = train_model()
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    training = summarize_samples(retained_samples, "training")
    evaluation_before = summarize_samples(retained_samples, "evaluation-before")
    evaluation_after = summarize_samples(retained_samples, "evaluation-after")
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
        "verifierId": "exact-integer-v1",
        "seed": args.seed,
        "optimizerSteps": int(getattr(result, "global_step", latest_step)),
        "retainedSamples": len(retained_samples),
        "trainingVerifierPassRate": training["verifierPassRate"],
        "evaluationBeforePassRate": before_pass_rate,
        "evaluationAfterPassRate": after_pass_rate,
        "evaluationPassRateDelta": evaluation_delta,
        "elapsedMs": elapsed_ms,
        "checkpointRetained": False,
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
    if retained_samples:
        replay_samples = [
            sample
            for phase in ["evaluation-before", "evaluation-after", "training"]
            for sample in retained_samples
            if sample.get("phase") == phase
        ]
        replay = {
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
