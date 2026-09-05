"""Typed, backend-neutral SFT/DPO dataset and metric adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

METHOD_CAPABILITIES = {
    "trl": frozenset({"sft", "dpo", "grpo", "rloo", "ppo"}),
    "axolotl": frozenset({"sft", "dpo", "grpo"}),
}


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
    return records, hashlib.sha256(payload).hexdigest()


def split_offline_dataset(records: list[dict[str, Any]], evaluation_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < evaluation_rows < len(records):
        raise ValueError("evaluationRows must leave training and evaluation records")
    return records[:-evaluation_rows], records[-evaluation_rows:]


def normalize_offline_metrics(method: str, phase: str, metrics: dict[str, Any]) -> dict[str, float | None]:
    def finite(*keys: str) -> float | None:
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
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


def axolotl_offline_config(config: dict[str, Any], train_path: str, eval_path: str, output_dir: str) -> dict[str, Any]:
    method = config["method"]
    if method not in ("sft", "dpo"):
        raise ValueError("Axolotl offline adapter supports sft or dpo")
    return {
        "base_model": config["modelId"],
        "model_revision": config["modelRevision"],
        "tokenizer_config": config["modelId"],
        "adapter": "lora",
        "lora_r": config["loraRank"],
        "lora_alpha": config["loraAlpha"],
        "lora_dropout": config["loraDropout"],
        "sequence_len": config["maxSequenceLength"],
        "micro_batch_size": config["perDeviceTrainBatchSize"],
        "gradient_accumulation_steps": config["gradientAccumulationSteps"],
        "learning_rate": config["learningRate"],
        "max_steps": config["maxSteps"],
        "output_dir": output_dir,
        "datasets": [{"path": train_path, "type": "completion" if method == "sft" else "bradley_terry"}],
        "test_datasets": [{"path": eval_path, "type": "completion" if method == "sft" else "bradley_terry"}],
        "rl": "dpo" if method == "dpo" else None,
        "dataset_prepared_path": str(Path(output_dir) / "prepared"),
    }
