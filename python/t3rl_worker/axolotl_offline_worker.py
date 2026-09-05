#!/usr/bin/env python3
"""Axolotl translation adapter for the public T3RL SFT/DPO schema."""

from __future__ import annotations

import argparse
import json
import subprocess
import shutil
import sys
from importlib import metadata
from pathlib import Path

import yaml

from offline import axolotl_offline_config, load_offline_dataset, normalize_offline_metrics, split_offline_dataset
from checkpointing import CheckpointPublisher, checkpoint_policy, compatibility, dataset_cursor, model_identity
from rlvr import emit, write_json
from trl_offline_worker import resolve_config
from trl_worker import resolve_model_revision


def run(args: argparse.Namespace) -> None:
    config = resolve_config(json.loads(args.config_json))
    version = metadata.version("axolotl")
    emit({"type": "hello", "protocol": 2, "runner": "axolotl", "runnerVersion": version})
    records, dataset_sha = load_offline_dataset(config["projectDatasetPath"], config["method"], config["datasetFormat"])
    train_rows, eval_rows = split_offline_dataset(records, config["evaluationRows"])
    run_dir = Path(args.run_dir)
    train_path = run_dir / "offline-train.json"
    eval_path = run_dir / "offline-eval.json"
    write_json(train_path, train_rows)
    write_json(eval_path, eval_rows)
    output_dir = run_dir / "trainer"
    translated = axolotl_offline_config(config, str(train_path), str(eval_path), str(output_dir))
    if args.resume_checkpoint:
        translated["resume_from_checkpoint"] = args.resume_checkpoint
    if args.warm_start_adapter:
        translated["lora_model_dir"] = args.warm_start_adapter
    from huggingface_hub import HfApi
    revision = resolve_model_revision({"HfApi": HfApi}, config["modelId"], config["modelRevision"])
    tokenizer_revision = resolve_model_revision({"HfApi": HfApi}, config["modelId"], config["tokenizerRevision"])
    identity = model_identity(config, resolved_revision=revision, tokenizer_revision=tokenizer_revision)
    compat = compatibility(model=identity, framework_id="axolotl", framework_version=version)
    policy = checkpoint_policy(config)
    translated_path = run_dir / "axolotl-offline.yml"
    translated_path.write_text(yaml.safe_dump({key: value for key, value in translated.items() if value is not None}, sort_keys=True))
    emit({"type": "manifest", "model": identity, "checkpointPolicy": policy, "values": {**config, "datasetSha256": dataset_sha, "datasetRows": len(records), "trainingRows": len(train_rows), "evaluationRows": len(eval_rows), "evaluationClaim": config["evaluationClaim"], "axolotlConfigSha256": __import__("hashlib").sha256(translated_path.read_bytes()).hexdigest(), "framework": {"axolotl": version, "trl": metadata.version("trl")}}})
    completed = subprocess.run([sys.executable, "-m", "axolotl.cli.train", str(translated_path)], cwd=run_dir, capture_output=True, text=True, timeout=86_400, check=False)
    log_path = run_dir / "axolotl.log"
    log_path.write_text((completed.stdout + "\n" + completed.stderr)[-65536:])
    emit({"type": "artifact", "kind": "log", "path": log_path.relative_to(run_dir).as_posix()})
    if completed.returncode != 0:
        raise RuntimeError(f"Axolotl exited with code {completed.returncode}")
    trainer_state = output_dir / "trainer_state.json"
    history = json.loads(trainer_state.read_text()).get("log_history", []) if trainer_state.exists() else []
    latest = history[-1] if history else {}
    step = int(latest.get("step", config["maxSteps"]))
    cursor = dataset_cursor(global_step=step, epoch=latest.get("epoch"), effective_batch_size=config["perDeviceTrainBatchSize"] * config["gradientAccumulationSteps"])
    publisher = CheckpointPublisher(run_dir=args.run_dir, compatibility_evidence=compat, policy=policy, emit_artifact=emit)
    def save_adapter(destination: str) -> None:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            source = output_dir / name
            if not source.is_file():
                matches = list(output_dir.rglob(name))
                if not matches:
                    raise RuntimeError(f"Axolotl output is missing {name}")
                source = matches[-1]
            shutil.copy2(source, Path(destination) / name)
    publisher.publish_adapter(save=save_adapter, global_step=step, tokens_seen=int(latest.get("num_input_tokens_seen", 0)), cursor=cursor)
    emit({"type": "metrics", "step": step, "wallClockMs": 0, "values": normalize_offline_metrics(config["method"], "eval_after", latest)})
    summary = run_dir / "summary.json"
    write_json(summary, {"method": config["method"], "evaluationClaim": config["evaluationClaim"], "datasetSha256": dataset_sha, "trainerState": latest})
    emit({"type": "artifact", "kind": "summary", "path": summary.relative_to(run_dir).as_posix()})
    emit({"type": "done", "status": "completed"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--warm-start-adapter")
    try:
        run(parser.parse_args())
        return 0
    except Exception as error:
        emit({"type": "error", "code": "RunnerException", "detail": str(error)[:2048]})
        emit({"type": "done", "status": "failed"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
