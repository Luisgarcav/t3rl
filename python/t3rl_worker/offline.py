"""Typed, backend-neutral SFT/DPO dataset and metric adapters."""

from __future__ import annotations

import hashlib
import json
import math
from importlib import metadata
from pathlib import Path
from typing import Any

from checkpointing import sha256_json

METHOD_CAPABILITIES = {
    "trl": frozenset({"sft", "dpo", "grpo", "rloo", "ppo"}),
    "axolotl": frozenset({"sft", "dpo", "grpo"}),
}

OFFLINE_SEED_USAGE = {
    "training": "model-initialization-and-training",
    "data": "training-sampler",
    "evaluationSample": "not-used-fixed-held-out-tail",
    "generation": "not-used-teacher-forced-evaluation",
}


def offline_seed_set(training_seed: int, requested_json: str | None) -> dict[str, int]:
    """Resolve the declared seed set without changing deterministic evaluation semantics."""
    keys = set(OFFLINE_SEED_USAGE)
    seeds = {key: training_seed for key in keys} if requested_json is None else json.loads(requested_json)
    if not isinstance(seeds, dict) or set(seeds) != keys:
        raise ValueError("the offline seed set must contain training, data, evaluationSample, and generation")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds.values()):
        raise ValueError("offline seeds must be integers from 0 to 2**32 - 1")
    if seeds["training"] != training_seed:
        raise ValueError("the declared training seed must match --seed")
    return seeds


def configure_offline_data_seed(trainer: Any, data_seed: int) -> None:
    """Set both pinned Trainer and Accelerate sampler configuration before constructing a loader."""
    trainer.args.data_seed = data_seed
    trainer.accelerator.dataloader_config.data_seed = data_seed


def _text(name: str, value: Any, maximum: int = 131_072) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be non-empty text of at most {maximum} characters")
    return value


def _stable_id(method: str, index: int, row: dict[str, Any]) -> str:
    supplied = row.get("sampleId")
    if isinstance(supplied, str) and supplied:
        return supplied
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sample_{hashlib.sha256(f'{method}\0{index}\0{payload}'.encode()).hexdigest()[:24]}"


def load_offline_dataset(path_value: str, method: str, dataset_format: str) -> tuple[list[dict[str, Any]], str]:
    path = Path(path_value)
    payload = path.read_bytes()
    decoded = (
        [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        if path.suffix == ".jsonl"
        else json.loads(payload)
    )
    if not isinstance(decoded, list) or len(decoded) < 2:
        raise ValueError("offline dataset must contain at least two records")
    records: list[dict[str, Any]] = []
    for index, value in enumerate(decoded):
        if not isinstance(value, dict):
            raise TypeError(f"dataset row {index} must be an object")
        sample_id = _stable_id(method, index, value)
        if method == "sft" and dataset_format == "sft-text":
            records.append({"sampleId": sample_id, "text": _text("text", value.get("text"))})
        elif method == "sft" and dataset_format == "sft-conversation":
            prompt = _text("prompt", value.get("prompt"))
            completion = _text("completion", value.get("completion"))
            records.append({"sampleId": sample_id, "prompt": prompt, "completion": completion})
        elif method == "dpo" and dataset_format == "dpo-preference":
            prompt = _text("prompt", value.get("prompt"))
            chosen = _text("chosen", value.get("chosen"))
            rejected = _text("rejected", value.get("rejected"))
            if chosen == rejected:
                raise ValueError(f"dataset row {index} chosen and rejected must differ")
            records.append({"sampleId": sample_id, "prompt": prompt, "chosen": chosen, "rejected": rejected})
        else:
            raise ValueError(f"unsupported {method} dataset format: {dataset_format}")
    if len({record["sampleId"] for record in records}) != len(records):
        raise ValueError("offline dataset sample IDs must be unique across training and evaluation")
    return records, hashlib.sha256(payload).hexdigest()


def split_offline_dataset(records: list[dict[str, Any]], evaluation_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < evaluation_rows < len(records):
        raise ValueError("evaluationRows must leave training and evaluation records")
    return records[:-evaluation_rows], records[-evaluation_rows:]


def offline_evaluation_protocol(dataset_sha: str, evaluation_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Describe deterministic, per-record evaluation of the held-out dataset tail."""
    evaluator = {
        "sources": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("offline.py", "trl_offline_worker.py")
        },
        "settings": {
            key: config[key] for key in (
                "method", "datasetFormat", "maxSequenceLength", "modelId",
                "modelRevision", "tokenizerRevision", "precision",
            )
        },
        "dependencies": {name: metadata.version(name) for name in ("torch", "trl", "peft", "transformers")},
    }
    protocol = {
        "version": 1,
        "datasetFingerprint": dataset_sha,
        "split": "test",
        "sampleIds": [row["sampleId"] for row in evaluation_records],
        "generationSeedPolicy": "fixed-per-sample",
        "decoding": {},
        "verifierSha256": sha256_json(evaluator),
    }
    return {**protocol, "protocolSha256": sha256_json(protocol)}


def validate_offline_evaluation_protocol(requested_json: str | None, actual: dict[str, Any]) -> None:
    """Fail before training when a study asks for evaluation this worker cannot produce."""
    if requested_json is None:
        return
    requested = json.loads(requested_json)
    if requested != actual:
        raise ValueError(
            "the study evaluation protocol does not match deterministic held-out-tail evaluation "
            "(dataset, test split, sample IDs, fixed-per-sample seed policy, empty decoding, and evaluator hash)"
        )


def evaluate_offline_samples(trainer: Any, method: str, evaluation_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the trainer's processed dataset and objective to measure each held-out record."""
    processed = trainer.eval_dataset
    if len(processed) != len(evaluation_records):
        raise ValueError("per-sample evaluation requires one processed record per held-out sample")
    samples = []
    for index, row in enumerate(evaluation_records):
        metrics = trainer.evaluate(eval_dataset=processed.select([index]))
        samples.append({
            "sampleId": row["sampleId"],
            "generationSeed": None,
            "values": {
                key: value for key, value in normalize_offline_metrics(method, "eval_after", metrics).items()
                if value is not None
            },
        })
    return samples


def normalize_offline_metrics(method: str, phase: str, metrics: dict[str, Any]) -> dict[str, float | None]:
    def finite(*keys: str) -> float | None:
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                return float(value)
        return None

    if method == "sft":
        return {
            f"{phase}/loss": finite("eval_loss", "loss"),
            f"{phase}/perplexity": finite("eval_perplexity", "perplexity"),
        }
    return {
        f"{phase}/loss": finite("eval_loss", "loss"),
        f"{phase}/preference_accuracy": finite("eval_rewards/accuracies", "rewards/accuracies", "preference_accuracy"),
        f"{phase}/reward_margin": finite("eval_rewards/margins", "rewards/margins", "reward_margin"),
    }
