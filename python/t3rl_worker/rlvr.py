#!/usr/bin/env python3
"""Backend-agnostic RLVR primitives shared by the t3RL post-training workers.

The worker protocol, the verifier, the versioned dataset, the split policy, and
the evidence ledger are identical whichever framework runs the optimization.
Keeping them here is what lets two backends be compared: a difference between
runs is then a difference in the backend, not in how each worker happened to
reimplement the measurement.

Nothing here imports a training framework, so the module stays importable from
every runner environment regardless of which framework that environment pins.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1


SUPPORTED_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


SUPPORTED_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


SUPPORTED_DATASET = "arithmetic-rlvr-v1"

# v2 raises evaluation resolution. Its rows come from the difficulty tier a
# calibration run measured at 0.427 base pass rate, so a holdout can move in
# either direction instead of sitting against the ceiling v1 sits against.
SUPPORTED_DATASET_V2 = "arithmetic-rlvr-v2"


SUPPORTED_DATASETS = frozenset({SUPPORTED_DATASET, SUPPORTED_DATASET_V2})


DATASET_PATHS = {
    SUPPORTED_DATASET: Path(__file__).parent / "datasets" / "arithmetic-rlvr-v1.json",
    SUPPORTED_DATASET_V2: Path(__file__).parent / "datasets" / "arithmetic-rlvr-v2.json",
}


VERIFIER_ID = "exact-integer-v2"


INTEGER_PATTERN = re.compile(r"(?<![\w.+-])[+-]?\d[\d,]*(?![\w.+-])")


EMIT_LOCK = threading.Lock()


PROTOCOL_STDOUT = sys.stdout


def emit(message: dict[str, Any]) -> None:
    """Write one complete NDJSON message even when the heartbeat is active."""
    line = json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n"
    with EMIT_LOCK:
        PROTOCOL_STDOUT.write(line)
        PROTOCOL_STDOUT.flush()


def run_metrics_heartbeat(
    stop: threading.Event,
    *,
    started: float,
    step: Callable[[], int],
    gpu_count: Callable[[], int],
) -> None:
    """Keep the worker watchdog alive while a backend performs blocking work."""
    while not stop.wait(15):
        emit(
            {
                "type": "metrics",
                "step": step(),
                "wallClockMs": int((time.monotonic() - started) * 1000),
                "values": {
                    "system/heartbeat": 1.0,
                    "system/gpu_count": float(gpu_count()),
                },
            }
        )


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


EVIDENCE_PHASES = ("evaluation-before", "training", "evaluation-after")


def order_replay_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order retained evidence by the phases' actual execution order."""
    return [
        sample
        for phase in EVIDENCE_PHASES
        for sample in samples
        if sample.get("phase") == phase
    ]


class EvidenceLedger:
    """
    Verifier outcomes for one run.

    Counts and reward statistics cover every completion the verifier scored.
    The retained samples are a bounded excerpt kept for the replay artifact.
    The two are deliberately separate: a retention budget bounds how much
    evidence a run stores, and must never bound what a run measures.
    """

    def __init__(self, retain_limit: int) -> None:
        self._phase_retain_limit = max(1, retain_limit // len(EVIDENCE_PHASES))
        self._samples: list[dict[str, Any]] = []
        self._totals: dict[str, dict[str, float]] = {}

    def _phase_totals(self, phase: str) -> dict[str, float]:
        return self._totals.setdefault(
            phase,
            {"count": 0.0, "passed": 0.0, "sum": 0.0, "squareSum": 0.0, "retained": 0.0},
        )

    def record(
        self,
        *,
        phase: str,
        prompt: str,
        expected: str,
        completion: str,
        parsed: str | None,
        reward: float,
    ) -> None:
        totals = self._phase_totals(phase)
        totals["count"] += 1.0
        totals["passed"] += 1.0 if reward == 1.0 else 0.0
        totals["sum"] += reward
        totals["squareSum"] += reward * reward
        if totals["retained"] >= self._phase_retain_limit:
            return
        totals["retained"] += 1.0
        self._samples.append(
            {
                "phase": phase,
                "prompt": prompt,
                "expected": expected,
                "completion": completion,
                "parsedAnswer": parsed,
                "reward": reward,
                "verifier": {"id": VERIFIER_ID, "passed": reward == 1.0},
            }
        )

    @property
    def samples(self) -> list[dict[str, Any]]:
        return self._samples

    @property
    def retained_count(self) -> int:
        return len(self._samples)

    def summarize(self, phase: str) -> dict[str, Any]:
        samples = [sample for sample in self._samples if sample["phase"] == phase]
        totals = self._totals.get(phase)
        count = 0 if totals is None else int(totals["count"])
        if totals is None or count == 0:
            return {
                "phase": phase,
                "sampleCount": 0,
                "rewardMean": None,
                "rewardStd": None,
                "verifierPassRate": None,
                "samples": samples,
            }
        mean = totals["sum"] / count
        # Population variance, clamped because the streaming form can land a
        # few ulps below zero when every reward is identical.
        variance = max(totals["squareSum"] / count - mean * mean, 0.0)
        return {
            "phase": phase,
            "sampleCount": count,
            "rewardMean": mean,
            "rewardStd": math.sqrt(variance),
            "verifierPassRate": totals["passed"] / count,
            "samples": samples,
        }


def resolve_evidence_phase(
    requested_phase: str | None, trainer_state: Any
) -> str:
    if requested_phase != "evaluation":
        return "training"
    if int(getattr(trainer_state, "global_step", 0)) == 0:
        return "evaluation-before"
    return "evaluation-after"


def make_exact_integer_reward(
    ledger: EvidenceLedger, phase_resolver: Callable[[str], str] | None = None
) -> Callable[..., list[float]]:
    """
    Builds the verifier reward function.

    `phase_resolver` exists for backends that do not hand the reward function a
    trainer state. Such a backend declares the phase it is in rather than having
    it inferred, which keeps the evidence labels correct instead of silently
    collapsing both evaluations into one phase.
    """
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
            prompt = (
                prompts[index] if prompts is not None and index < len(prompts) else ""
            )
            requested_phase = (
                evidencePhase[index]
                if evidencePhase is not None and index < len(evidencePhase)
                else "training"
            )
            ledger.record(
                phase=(
                    phase_resolver(requested_phase)
                    if phase_resolver is not None
                    else resolve_evidence_phase(requested_phase, trainer_state)
                ),
                prompt=completion_text(prompt),
                expected=expected,
                completion=text,
                parsed=parsed,
                reward=reward,
            )
        return rewards

    return exact_integer_reward


def write_json(run_dir: str, relative_path: str, value: Any) -> None:
    target = os.path.join(run_dir, relative_path)
    os.makedirs(os.path.dirname(target) or run_dir, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)


def metric_value(logs: dict[str, Any], *keys: str) -> float | str | None:
    for key in keys:
        if key in logs:
            return finite_metric(logs[key])
    return None


def normalize_grpo_metrics(
    logs: dict[str, Any],
    *,
    step: int,
    evaluation_passes: int,
    config: dict[str, Any],
    elapsed_seconds: float,
    gpu_memory_allocated_gb: float,
) -> dict[str, float | str | None]:
    """Translate the TRL-shaped metrics emitted by both GRPO backends."""
    if any(key.startswith("eval_") for key in logs):
        metrics: dict[str, float | str | None] = {
            "eval/reward": metric_value(logs, "eval_reward"),
            "eval/reward_std": metric_value(logs, "eval_reward_std"),
            "eval/verifier_pass_rate": metric_value(
                logs,
                "eval_rewards/exact_integer_reward/mean",
                "eval_rewards/axolotl_reward.exact_integer_reward/mean",
                "eval_reward",
            ),
            "eval/completion_length": metric_value(
                logs, "eval_completions/mean_length"
            ),
            "eval/kl": metric_value(logs, "eval_kl"),
            "eval/entropy": metric_value(logs, "eval_entropy"),
        }
    else:
        reward = metric_value(logs, "reward", "rewards/exact_integer_reward/mean")
        metrics = {
            "train/reward": reward,
            "train/reward_std": metric_value(
                logs,
                "reward_std",
                "rewards/exact_integer_reward/std",
            ),
            "train/verifier_pass_rate": metric_value(
                logs,
                "rewards/exact_integer_reward/mean",
                "rewards/axolotl_reward.exact_integer_reward/mean",
                "rewards/exact_integer_reward",
                "reward",
            ),
            "train/kl": metric_value(logs, "kl"),
            "train/approx_kl": metric_value(logs, "kl"),
            "train/entropy": metric_value(logs, "entropy"),
            "train/completion_length": metric_value(
                logs, "completions/mean_length", "completion_length"
            ),
            "train/loss": metric_value(logs, "policy_loss", "loss"),
            "train/policy_loss": metric_value(logs, "policy_loss", "loss"),
            "train/learning_rate": metric_value(logs, "learning_rate"),
            "train/grad_norm": metric_value(logs, "grad_norm"),
            "system/step_time_seconds": metric_value(logs, "step_time"),
        }

    observed_tokens = metric_value(logs, "num_tokens", "eval_num_tokens")
    completed_training_bound = (
        step
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
    metrics.update(
        {
            "system/num_tokens": observed_tokens,
            "system/generated_tokens_upper_bound": float(
                completed_training_bound + completed_evaluation_bound
            ),
            "system/tokens_per_second": (
                float(observed_tokens) / max(elapsed_seconds, 1e-9)
                if isinstance(observed_tokens, (int, float))
                else None
            ),
            "system/gpu_memory_allocated_gb": float(gpu_memory_allocated_gb),
        }
    )
    return metrics


def has_observed_metric(metrics: dict[str, Any]) -> bool:
    """Exclude Trainer summaries that contain no normalized training or eval data."""
    return any(
        key.startswith(("train/", "eval/")) and value is not None
        for key, value in metrics.items()
    )
