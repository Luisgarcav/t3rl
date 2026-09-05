"""Trainer-neutral checkpoint publication for protocol-v2 T3RL workers.

Framework adapters decide when their trainer has completed a save. This module
owns everything after that boundary: atomic publication, required-state
validation, compatibility evidence, cancellation intent, and PEFT export.
It deliberately imports no ML framework so the deterministic fixture and unit
tests run with the Python standard library alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
TRAINER_STATE_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "t3rl-dataset-cursor.json",
)


def _wire_canonical_value(value: Any) -> Any:
    """Normalize numbers to the representation JSON.parse/JSON.stringify preserves."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not support non-finite numbers")
        return int(value) if value.is_integer() else value
    if isinstance(value, list):
        return [_wire_canonical_value(entry) for entry in value]
    if isinstance(value, dict):
        return {key: _wire_canonical_value(entry) for key, entry in value.items()}
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _wire_canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def checkpoint_policy(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "cadenceSteps": config["checkpointCadenceSteps"],
        "maxIntermediateCheckpoints": config["maxIntermediateCheckpoints"],
        "keepBest": config["keepBest"],
        "keepFinal": config["keepFinal"],
        "gracefulDeadlineSeconds": config["gracefulCheckpointDeadlineSeconds"],
    }


def peft_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "peftType": "LORA",
        "taskType": "CAUSAL_LM",
        "rank": config["loraRank"],
        "alpha": config["loraAlpha"],
        "dropout": config["loraDropout"],
        "bias": config["loraBias"],
        "targetModules": list(config["loraTargetModules"]),
        "modulesToSave": list(config["loraModulesToSave"]),
        "useRslora": config["useRslora"],
    }


def model_identity(
    config: dict[str, Any], *, resolved_revision: str, tokenizer_revision: str
) -> dict[str, Any]:
    peft = peft_config(config)
    return {
        "baseModelId": config["modelId"],
        "baseModelRevision": resolved_revision,
        "tokenizerRevision": tokenizer_revision,
        "peftConfig": peft,
        "peftConfigSha256": sha256_json(peft),
        "quantization": config["quantization"],
        "precision": config["precision"],
        "trainableModules": list(config["loraTargetModules"])
        + list(config["loraModulesToSave"]),
    }


def compatibility(
    *, model: dict[str, Any], framework_id: str, framework_version: str
) -> dict[str, Any]:
    lock_sha = os.environ.get("T3RL_ENVIRONMENT_LOCK_SHA256", "").strip()
    fingerprint = os.environ.get("T3RL_ENVIRONMENT_FINGERPRINT", "").strip()
    if not fingerprint:
        # Direct worker invocations remain inspectable, while server-launched
        # runs always receive the lock-derived authoritative fingerprint.
        fingerprint = hashlib.sha256(
            canonical_json(
                {
                    "framework": framework_id,
                    "version": framework_version,
                    "model": model,
                }
            ).encode("utf-8")
        ).hexdigest()
    return {
        "model": model,
        "framework": {"id": framework_id, "version": framework_version},
        "environmentFingerprint": fingerprint,
        "environmentLockSha256": lock_sha or None,
    }


def dataset_cursor(
    *, global_step: int, epoch: float | None, effective_batch_size: int
) -> dict[str, Any]:
    return {
        "epoch": max(0.0, float(epoch or 0.0)),
        "batchInEpoch": max(0, int(global_step)),
        "sampleOffset": max(0, int(global_step) * int(effective_batch_size)),
    }


def evaluate_before_training(trainer: Any, resume_checkpoint: str | None) -> Any:
    """Evaluate the actual starting weights, including an exact-resume checkpoint.

    Transformers restores model weights inside ``train(resume_from_checkpoint=...)``.
    That is too late for a before-training evaluation, so the pinned adapter boundary
    explicitly loads only the model state first. ``train`` still performs the full
    resume afterwards, including optimizer, scheduler, RNG, and trainer state.
    """
    if resume_checkpoint is not None:
        load_from_checkpoint = getattr(trainer, "_load_from_checkpoint", None)
        if not callable(load_from_checkpoint):
            raise RuntimeError(
                "the selected trainer cannot preload an exact-resume checkpoint for evaluation"
            )
        load_from_checkpoint(resume_checkpoint)
    return trainer.evaluate()


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"published artifact contains a symbolic link: {path.name}"
            )
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(
    *, parent: Path, name: str, populate: Callable[[Path], None]
) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / name
    if target.exists():
        raise FileExistsError(f"artifact already published: {target}")
    staging = Path(tempfile.mkdtemp(dir=parent, prefix=f".{name}.", suffix=".tmp"))
    try:
        populate(staging)
        _fsync_tree(staging)
        os.replace(staging, target)
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def require_adapter(directory: str | Path) -> None:
    root = Path(directory)
    for name in (ADAPTER_CONFIG, ADAPTER_WEIGHTS):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"PEFT adapter is missing {name}")


def load_fixture_adapter(directory: str | Path) -> dict[str, Any]:
    """Load the standard-library fixture adapter through its public files."""
    require_adapter(directory)
    root = Path(directory)
    config = json.loads((root / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    weights = json.loads((root / ADAPTER_WEIGHTS).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(weights, dict):
        raise TypeError("fixture adapter files must contain JSON objects")
    return {"config": config, "weights": weights}


class CheckpointPublisher:
    """Atomic checkpoint/adapter publisher shared by all framework callbacks."""

    def __init__(
        self,
        *,
        run_dir: str,
        compatibility_evidence: dict[str, Any],
        policy: dict[str, Any],
        emit_artifact: Callable[[dict[str, Any]], None],
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.compatibility = compatibility_evidence
        self.policy = policy
        self.emit_artifact = emit_artifact
        self.shutdown_requested = threading.Event()
        self._published_steps: set[tuple[int, str]] = set()

    def install_signal_handlers(self) -> None:
        def request_shutdown(_signum: int, _frame: Any) -> None:
            self.shutdown_requested.set()

        signal.signal(signal.SIGTERM, request_shutdown)
        signal.signal(signal.SIGINT, request_shutdown)

    def publish_checkpoint(
        self,
        *,
        trainer_checkpoint: str,
        checkpoint_class: str,
        global_step: int,
        tokens_seen: int,
        cursor: dict[str, Any],
    ) -> str | None:
        key = (global_step, checkpoint_class)
        if key in self._published_steps:
            return None
        if checkpoint_class == "final" and not self.policy["keepFinal"]:
            return None
        source = Path(trainer_checkpoint).resolve()
        if not source.is_dir():
            raise ValueError(f"trainer checkpoint does not exist: {source}")
        name = f"checkpoint-{global_step}-{checkpoint_class}"
        gradient_scaler = (
            "captured" if (source / "scaler.pt").is_file() else "not-applicable"
        )
        if (
            self.compatibility.get("model", {}).get("precision") == "fp16"
            and gradient_scaler != "captured"
        ):
            raise ValueError("fp16 checkpoint is missing scaler.pt")
        state_files = list(TRAINER_STATE_FILES)
        if gradient_scaler == "captured":
            state_files.append("scaler.pt")

        def populate(staging: Path) -> None:
            for child in source.iterdir():
                destination = staging / child.name
                if child.is_symlink():
                    raise ValueError(
                        f"trainer checkpoint contains a symlink: {child.name}"
                    )
                if child.is_dir():
                    shutil.copytree(child, destination)
                elif child.is_file():
                    shutil.copy2(child, destination)
                else:
                    raise ValueError(f"unsupported checkpoint entry: {child.name}")
            (staging / "t3rl-dataset-cursor.json").write_text(
                canonical_json(cursor) + "\n", encoding="utf-8"
            )
            require_adapter(staging)
            for state_file in state_files:
                if not (staging / state_file).is_file():
                    raise ValueError(f"resumable checkpoint is missing {state_file}")

        published = _publish_directory(
            parent=self.run_dir / "checkpoints", name=name, populate=populate
        )
        relative = published.relative_to(self.run_dir).as_posix()
        self.emit_artifact(
            {
                "type": "artifact",
                "kind": "checkpoint",
                "path": relative,
                "evidence": {
                    "_tag": "Checkpoint",
                    "checkpointClass": checkpoint_class,
                    "compatibility": self.compatibility,
                    "globalStep": global_step,
                    "tokensSeen": max(0, int(tokens_seen)),
                    "datasetCursor": cursor,
                    "resumeState": {
                        "trainerState": True,
                        "optimizerState": True,
                        "schedulerState": True,
                        "rngState": True,
                        "datasetCursorState": True,
                        "gradientScalerState": gradient_scaler,
                        "stateFiles": state_files,
                    },
                },
            }
        )
        self._published_steps.add(key)
        return relative

    def publish_adapter(
        self,
        *,
        save: Callable[[str], None],
        global_step: int,
        tokens_seen: int,
        cursor: dict[str, Any],
    ) -> str:
        name = f"adapter-step-{global_step}"

        def populate(staging: Path) -> None:
            save(str(staging))
            require_adapter(staging)

        published = _publish_directory(
            parent=self.run_dir / "adapters", name=name, populate=populate
        )
        relative = published.relative_to(self.run_dir).as_posix()
        self.emit_artifact(
            {
                "type": "artifact",
                "kind": "adapter",
                "path": relative,
                "evidence": {
                    "_tag": "Adapter",
                    "compatibility": self.compatibility,
                    "globalStep": global_step,
                    "tokensSeen": max(0, int(tokens_seen)),
                    "datasetCursor": cursor,
                },
            }
        )
        return relative


def transformers_checkpoint_callback(
    *,
    callback_base: type,
    publisher: CheckpointPublisher,
    trainer_output_dir: str,
    max_steps: int,
    effective_batch_size: int,
) -> Any:
    """Adapt a Transformers-compatible callback API to CheckpointPublisher."""

    class T3rlCheckpointCallback(callback_base):  # type: ignore[misc, valid-type]
        tokens_seen = 0

        def _update(self, state: Any, logs: dict[str, Any] | None = None) -> None:
            observed = getattr(state, "num_input_tokens_seen", None)
            if isinstance(observed, (int, float)):
                self.tokens_seen = max(self.tokens_seen, int(observed))
            for key in ("num_tokens", "train/num_tokens", "tokens_seen"):
                value = (logs or {}).get(key)
                if isinstance(value, (int, float)):
                    self.tokens_seen = max(self.tokens_seen, int(value))

        def on_log(
            self,
            _args: Any,
            state: Any,
            _control: Any,
            logs: Any = None,
            **_kwargs: Any,
        ) -> None:
            self._update(state, logs if isinstance(logs, dict) else None)

        def on_step_end(
            self, _args: Any, state: Any, control: Any, **_kwargs: Any
        ) -> Any:
            self._update(state)
            if publisher.shutdown_requested.is_set():
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            step = int(getattr(state, "global_step", 0))
            checkpoint_class = (
                "graceful"
                if publisher.shutdown_requested.is_set()
                else "final"
                if step >= max_steps
                else "intermediate"
            )
            publisher.publish_checkpoint(
                trainer_checkpoint=os.path.join(
                    trainer_output_dir, f"checkpoint-{step}"
                ),
                checkpoint_class=checkpoint_class,
                global_step=step,
                tokens_seen=self.tokens_seen,
                cursor=dataset_cursor(
                    global_step=step,
                    epoch=getattr(state, "epoch", None),
                    effective_batch_size=effective_batch_size,
                ),
            )
            return control

    return T3rlCheckpointCallback()
