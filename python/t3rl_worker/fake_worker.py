#!/usr/bin/env python3
"""Deterministic T3RL worker that performs no learning.

Exists to exercise the run kernel end to end: spawn, protocol, artifacts,
cancellation, and every failure path, without depending on Gymnasium,
Stable-Baselines3, or a GPU. Standard library only.

Output is byte-identical for a given seed and scenario, so it can serve as a
fixture: `wallClockMs` derives from the step index rather than from the clock.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

from checkpointing import (
    CheckpointPublisher,
    checkpoint_policy,
    compatibility,
    dataset_cursor,
    load_fixture_adapter,
    model_identity,
)

PROTOCOL_VERSION = 1
CHECKPOINT_PROTOCOL_VERSION = 2
RUNNER_ID = "fake"
RUNNER_VERSION = "0.1.0"
METRIC_STEPS = 10

FIXTURE_MODEL_REVISION = "f" * 64
FIXTURE_CONFIG = {
    "modelId": "t3rl/tiny-linear-fixture",
    "modelRevision": FIXTURE_MODEL_REVISION,
    "tokenizerRevision": FIXTURE_MODEL_REVISION,
    "quantization": "none",
    "precision": "fp32",
    "loraRank": 2,
    "loraAlpha": 4.0,
    "loraDropout": 0.0,
    "loraBias": "none",
    "loraTargetModules": ["linear"],
    "loraModulesToSave": [],
    "useRslora": False,
    "checkpointCadenceSteps": 2,
    "maxIntermediateCheckpoints": 2,
    "keepBest": False,
    "keepFinal": True,
    "gracefulCheckpointDeadlineSeconds": 5,
    "maxSteps": 8,
    "stepDelayMs": 0,
}


def emit(message: dict) -> None:
    """Writes one NDJSON line and flushes. A buffered worker looks stalled."""
    sys.stdout.write(json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def metric_values(seed: int, step: int) -> dict:
    """Values from the seed and step alone, so two runs at one seed agree.

    Non-finite results are sent as markers, never as JSON null: `null` means
    "not captured", which is a different claim than "diverged".
    """
    base = (seed * 31 + step * 7) % 100
    values = {
        "train/return": base / 10.0,
        "train/episode_length": float(50 + (base % 25)),
        "train/policy_loss": round(1.0 / (step + 1), 6),
    }
    # One deliberately unrecorded metric, so consumers exercise the null path.
    values["train/approx_kl"] = None if step % 3 == 0 else round(0.01 * step, 6)
    # And one deliberate divergence near the end, to exercise the marker path.
    if step == METRIC_STEPS:
        values["train/value_loss"] = "nan"
    else:
        values["train/value_loss"] = round(2.0 / (step + 1), 6)
    return values


def emit_hello(protocol: int = PROTOCOL_VERSION) -> None:
    emit(
        {
            "type": "hello",
            "protocol": protocol,
            "runner": RUNNER_ID,
            "runnerVersion": RUNNER_VERSION,
        }
    )


def emit_manifest(args: argparse.Namespace) -> None:
    values = dict(args.config_json)
    values.update(
        {
            "scenario": args.scenario,
            "seed": args.seed,
            "steps": METRIC_STEPS,
            "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
    )
    emit(
        {
            "type": "manifest",
            "values": values,
        }
    )


def emit_metrics(seed: int, steps: int) -> None:
    for step in range(1, steps + 1):
        emit(
            {
                "type": "metrics",
                "step": step,
                # Derived, not measured: a clock reading would make two runs at
                # the same seed differ and this worker useless as a fixture.
                "wallClockMs": step * 100,
                "values": metric_values(seed, step),
            }
        )


def write_summary(run_dir: str, seed: int) -> str:
    os.makedirs(run_dir, exist_ok=True)
    relative = "summary.json"
    payload = {
        "runner": RUNNER_ID,
        "runnerVersion": RUNNER_VERSION,
        "seed": seed,
        "steps": METRIC_STEPS,
        "finalReturn": metric_values(seed, METRIC_STEPS)["train/return"],
    }
    target = os.path.join(run_dir, relative)
    descriptor, temporary = tempfile.mkstemp(
        dir=run_dir,
        prefix=f".{relative}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return relative


def sleep_forever() -> None:
    while True:
        time.sleep(3600)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fixture_adapter_config(config: dict) -> dict:
    return {
        "base_model_name_or_path": config["modelId"],
        "bias": config["loraBias"],
        "inference_mode": True,
        "lora_alpha": config["loraAlpha"],
        "lora_dropout": config["loraDropout"],
        "peft_type": "LORA",
        "r": config["loraRank"],
        "revision": config["modelRevision"],
        "target_modules": config["loraTargetModules"],
        "task_type": "CAUSAL_LM",
    }


def _write_fixture_trainer_checkpoint(
    run_dir: Path, *, step: int, weight: float, momentum: float, config: dict
) -> Path:
    root = run_dir / "trainer" / f"checkpoint-{step}"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "adapter_config.json", _fixture_adapter_config(config))
    _atomic_json(root / "adapter_model.safetensors", {"linear.lora": weight})
    _atomic_json(
        root / "trainer_state.json",
        {"epoch": step / config["maxSteps"], "global_step": step},
    )
    _atomic_json(root / "optimizer.pt", {"momentum": momentum})
    _atomic_json(root / "scheduler.pt", {"step": step})
    _atomic_json(root / "rng_state.pth", {"seed": config.get("seed", 0), "step": step})
    _atomic_json(
        root / "fixture_state.json",
        {"globalStep": step, "momentum": momentum, "weight": weight},
    )
    return root


def _load_fixture_checkpoint(path: str) -> tuple[int, float, float]:
    root = Path(path)
    payload = json.loads((root / "fixture_state.json").read_text(encoding="utf-8"))
    return (
        int(payload["globalStep"]),
        float(payload["weight"]),
        float(payload["momentum"]),
    )


def run_checkpoint_fixture(args: argparse.Namespace) -> int:
    config = {**FIXTURE_CONFIG, **args.config_json, "seed": args.seed}
    model = model_identity(
        config,
        resolved_revision=config["modelRevision"],
        tokenizer_revision=config["tokenizerRevision"],
    )
    compat = compatibility(
        model=model, framework_id=RUNNER_ID, framework_version=RUNNER_VERSION
    )
    policy = checkpoint_policy(config)
    emit_hello(CHECKPOINT_PROTOCOL_VERSION)
    emit(
        {
            "type": "manifest",
            "values": config,
            "model": model,
            "checkpointPolicy": policy,
        }
    )

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    publisher = CheckpointPublisher(
        run_dir=str(run_dir),
        compatibility_evidence=compat,
        policy=policy,
        emit_artifact=emit,
    )
    publisher.install_signal_handlers()

    step = 0
    weight = float(args.seed) / 100.0
    momentum = 0.0
    if args.resume_checkpoint:
        step, weight, momentum = _load_fixture_checkpoint(args.resume_checkpoint)
    elif args.warm_start_adapter:
        loaded = load_fixture_adapter(args.warm_start_adapter)
        weight = float(loaded["weights"]["linear.lora"])

    while step < config["maxSteps"]:
        step += 1
        gradient = ((args.seed * 17 + step * 13) % 97) / 10_000.0
        momentum = round(momentum * 0.9 + gradient, 12)
        weight = round(weight + momentum, 12)
        emit(
            {
                "type": "metrics",
                "step": step,
                "wallClockMs": step * 100,
                "values": {
                    "train/loss": round(weight * weight, 12),
                    "train/weight": weight,
                },
            }
        )
        if config["stepDelayMs"] > 0:
            time.sleep(float(config["stepDelayMs"]) / 1000.0)
        cadence = step % config["checkpointCadenceSteps"] == 0
        graceful = publisher.shutdown_requested.is_set()
        if cadence or graceful or step == config["maxSteps"]:
            trainer_checkpoint = _write_fixture_trainer_checkpoint(
                run_dir,
                step=step,
                weight=weight,
                momentum=momentum,
                config=config,
            )
            publisher.publish_checkpoint(
                trainer_checkpoint=str(trainer_checkpoint),
                checkpoint_class=(
                    "graceful"
                    if graceful
                    else "final"
                    if step == config["maxSteps"]
                    else "intermediate"
                ),
                global_step=step,
                tokens_seen=step * 16,
                cursor=dataset_cursor(
                    global_step=step,
                    epoch=step / config["maxSteps"],
                    effective_batch_size=1,
                ),
            )
        if graceful:
            shutil.rmtree(run_dir / "trainer", ignore_errors=True)
            return 0

    final_cursor = dataset_cursor(
        global_step=step,
        epoch=1.0,
        effective_batch_size=1,
    )

    def save_adapter(directory: str) -> None:
        target = Path(directory)
        _atomic_json(target / "adapter_config.json", _fixture_adapter_config(config))
        _atomic_json(target / "adapter_model.safetensors", {"linear.lora": weight})

    publisher.publish_adapter(
        save=save_adapter,
        global_step=step,
        tokens_seen=step * 16,
        cursor=final_cursor,
    )
    shutil.rmtree(run_dir / "trainer", ignore_errors=True)
    summary = {
        "runner": RUNNER_ID,
        "seed": args.seed,
        "globalStep": step,
        "weight": weight,
        "loss": round(weight * weight, 12),
    }
    _atomic_json(run_dir / "summary.json", summary)
    emit({"type": "artifact", "kind": "summary", "path": "summary.json"})
    emit({"type": "done", "status": "completed"})
    return 0


def run(args: argparse.Namespace) -> int:
    scenario = args.scenario

    if scenario == "checkpoint":
        return run_checkpoint_fixture(args)

    if scenario == "bad-protocol":
        emit_hello(protocol=99)
        sleep_forever()
        return 0

    emit_hello()
    emit_manifest(args)

    if scenario == "malformed":
        sys.stdout.write("{not json\n")
        sys.stdout.flush()
        sleep_forever()
        return 0

    if scenario == "stall":
        emit_metrics(args.seed, 1)
        sleep_forever()
        return 0

    if scenario == "ignore-cancel":
        # Refuse to die on the polite signal so the server has to escalate.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        emit_metrics(args.seed, 2)
        sleep_forever()
        return 0

    if scenario == "exit":
        emit_metrics(args.seed, 2)
        return 3

    if scenario == "fail":
        emit_metrics(args.seed, 2)
        emit(
            {"type": "error", "code": "RunnerException", "detail": "synthetic failure"}
        )
        return 1

    emit_metrics(args.seed, METRIC_STEPS)
    relative = write_summary(args.run_dir, args.seed)
    emit({"type": "artifact", "kind": "summary", "path": relative})
    emit({"type": "done", "status": "completed"})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic T3RL fake worker")
    parser.add_argument(
        "--scenario",
        default="success",
        choices=[
            "success",
            "fail",
            "malformed",
            "stall",
            "exit",
            "ignore-cancel",
            "bad-protocol",
            "checkpoint",
        ],
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config-json", default="{}", type=json.loads)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--warm-start-adapter")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
