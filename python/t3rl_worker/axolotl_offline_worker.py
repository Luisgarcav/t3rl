#!/usr/bin/env python3
"""Native Axolotl SFT/DPO adapter with the common checkpoint and evidence lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import redirect_stdout
from importlib import metadata
from pathlib import Path
from typing import Any

from checkpointing import CheckpointPublisher, checkpoint_policy, compatibility, dataset_cursor, evaluate_before_training, model_identity, sha256_json, transformers_checkpoint_callback
from offline import OFFLINE_SEED_USAGE, configure_offline_data_seed, evaluate_offline_samples, load_offline_dataset, normalize_offline_metrics, offline_evaluation_protocol, offline_seed_set, split_offline_dataset, validate_offline_evaluation_protocol
from rlvr import emit, write_json
from trl_offline_worker import resolve_config
from trl_worker import dependency_evidence, resolve_model_revision


def build_config(config: dict[str, Any], run_dir: str, seed: int) -> dict[str, Any]:
    """Translate only settings supported by the pinned native Axolotl trainer."""
    dataset = {"ds_type": "json", "split": "train", "type": "completion" if config["method"] == "sft" else None}
    if config["datasetFormat"] == "sft-conversation":
        dataset["type"] = {"field_instruction": "prompt", "field_output": "completion", "format": "{instruction}", "no_input_format": "{instruction}"}
    return {
        "base_model": config["modelId"], "revision_of_model": config["modelRevision"],
        "adapter": "lora", "lora_r": config["loraRank"], "lora_alpha": config["loraAlpha"],
        "lora_dropout": config["loraDropout"], "lora_target_modules": config["loraTargetModules"],
        "lora_modules_to_save": config["loraModulesToSave"] or None, "peft_use_rslora": config["useRslora"],
        "load_in_4bit": False, "load_in_8bit": False,
        "sequence_len": config["maxSequenceLength"], "sample_packing": False,
        "eval_sample_packing": False, "pad_to_sequence_len": False,
        "shuffle_before_merging_datasets": False, "dataset_processes": 1,
        "train_on_inputs": config["datasetFormat"] == "sft-text",
        "micro_batch_size": config["perDeviceTrainBatchSize"], "eval_batch_size": 1,
        "gradient_accumulation_steps": config["gradientAccumulationSteps"],
        "gradient_checkpointing": False, "learning_rate": config["learningRate"],
        "max_steps": config["maxSteps"], "num_epochs": 1, "seed": seed,
        "logging_steps": config["loggingSteps"], "eval_on_start": False,
        "eval_strategy": "steps", "eval_steps": config["checkpointCadenceSteps"],
        "save_strategy": "steps", "save_steps": config["checkpointCadenceSteps"],
        "save_total_limit": max(1, config["maxIntermediateCheckpoints"] + 2),
        "save_only_model": False, "save_safetensors": True,
        "bf16": config["precision"] == "bf16", "fp16": config["precision"] == "fp16",
        "datasets": [{**dataset, "path": str(Path(run_dir) / "offline-train.json")}],
        "test_datasets": [{**dataset, "path": str(Path(run_dir) / "offline-eval.json")}],
        "rl": "dpo" if config["method"] == "dpo" else None,
        "output_dir": str(Path(run_dir) / "trainer"),
        "dataset_prepared_path": str(Path(run_dir) / "prepared"),
    }


def evaluation_protocol(dataset_sha: str, eval_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    protocol = offline_evaluation_protocol(dataset_sha, eval_rows, config)
    protocol.pop("protocolSha256")
    protocol["verifierSha256"] = sha256_json({"offlineEvaluator": protocol["verifierSha256"], "axolotl": metadata.version("axolotl"), "adapterSourceSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return {**protocol, "protocolSha256": sha256_json(protocol)}


def load_dependencies() -> dict[str, Any]:
    import torch
    from axolotl.cli.config import load_cfg
    from axolotl.common.datasets import load_datasets, load_preference_datasets
    from axolotl.train import setup_model_and_trainer
    from huggingface_hub import HfApi
    from transformers import TrainerCallback, set_seed
    return locals()


def run(args: argparse.Namespace) -> None:
    if args.resume_checkpoint and args.warm_start_adapter:
        raise ValueError("resume checkpoint and warm-start adapter are mutually exclusive")
    config = resolve_config(json.loads(args.config_json))
    seeds = offline_seed_set(args.seed, os.environ.get("T3RL_SEED_SET_JSON"))
    if config["tokenizerRevision"] != config["modelRevision"]:
        raise ValueError("the Axolotl adapter requires tokenizerRevision=modelRevision")
    if config["loraBias"] != "none":
        raise ValueError("the Axolotl adapter requires loraBias=none")
    version = metadata.version("axolotl")
    emit({"type": "hello", "protocol": 2, "runner": "axolotl", "runnerVersion": version})
    deps = load_dependencies()
    records, dataset_sha = load_offline_dataset(config["projectDatasetPath"], config["method"], config["datasetFormat"])
    train_rows, eval_rows = split_offline_dataset(records, config["evaluationRows"])
    revision = resolve_model_revision(deps, config["modelId"], config["modelRevision"])
    config = {**config, "modelRevision": revision, "tokenizerRevision": revision}
    protocol = evaluation_protocol(dataset_sha, eval_rows, config)
    validate_offline_evaluation_protocol(os.environ.get("T3RL_EVALUATION_PROTOCOL_JSON"), protocol)
    deps["set_seed"](args.seed)
    identity = model_identity(config, resolved_revision=revision, tokenizer_revision=revision)
    policy = checkpoint_policy(config)
    publisher = CheckpointPublisher(run_dir=args.run_dir, compatibility_evidence=compatibility(model=identity, framework_id="axolotl", framework_version=version), policy=policy, emit_artifact=emit)
    publisher.install_signal_handlers()
    translated = build_config(config, args.run_dir, args.seed)
    translated.update({"resume_from_checkpoint": args.resume_checkpoint, "lora_model_dir": args.warm_start_adapter})
    write_json(args.run_dir, "offline-train.json", train_rows)
    write_json(args.run_dir, "offline-eval.json", eval_rows)
    write_json(args.run_dir, "axolotl-config.json", translated)
    emit({"type": "manifest", "model": identity, "checkpointPolicy": policy, "values": {**config, "seed": args.seed, "seeds": seeds, "seedUsage": OFFLINE_SEED_USAGE, "datasetSha256": dataset_sha, "datasetRows": len(records), "trainingRows": len(train_rows), "evaluationRows": len(eval_rows), "evaluationProtocol": protocol, "dependencies": {**dependency_evidence(deps), "axolotl": version}}})
    emit({"type": "artifact", "kind": "config", "path": "axolotl-config.json"})
    started = time.monotonic()
    try:
        cfg = deps["load_cfg"](str(Path(args.run_dir) / "axolotl-config.json"))
        loader = deps["load_datasets"] if config["method"] == "sft" else deps["load_preference_datasets"]
        trainer, _model, _tokenizer, _peft, _processor = deps["setup_model_and_trainer"](cfg, loader(cfg=cfg))
        configure_offline_data_seed(trainer, seeds["data"])
        callback = transformers_checkpoint_callback(callback_base=deps["TrainerCallback"], publisher=publisher, trainer_output_dir=translated["output_dir"], max_steps=config["maxSteps"], effective_batch_size=config["perDeviceTrainBatchSize"] * config["gradientAccumulationSteps"])
        trainer.add_callback(callback)
        before = evaluate_before_training(trainer, args.resume_checkpoint)
        emit({"type": "metrics", "step": 0, "wallClockMs": 0, "values": normalize_offline_metrics(config["method"], "eval_before", before)})
        trainer.train(resume_from_checkpoint=args.resume_checkpoint)
        if publisher.shutdown_requested.is_set():
            return
        after = trainer.evaluate()
        samples = evaluate_offline_samples(trainer, config["method"], eval_rows)
        step = int(trainer.state.global_step)
        cursor = dataset_cursor(global_step=step, epoch=trainer.state.epoch, effective_batch_size=config["perDeviceTrainBatchSize"] * config["gradientAccumulationSteps"])
        publisher.publish_adapter(save=trainer.save_model, global_step=step, tokens_seen=callback.tokens_seen, cursor=cursor)
    finally:
        shutil.rmtree(translated["output_dir"], ignore_errors=True)
        shutil.rmtree(translated["dataset_prepared_path"], ignore_errors=True)
    emit({"type": "metrics", "step": step, "wallClockMs": int((time.monotonic() - started) * 1000), "values": normalize_offline_metrics(config["method"], "eval_after", after)})
    write_json(args.run_dir, "summary.json", {"method": config["method"], "evaluationClaim": config["evaluationClaim"], "datasetSha256": dataset_sha, "seed": args.seed, "seeds": seeds, "seedUsage": OFFLINE_SEED_USAGE, "dataSeedApplied": trainer.args.data_seed, "optimizerSteps": step, "device": str(trainer.accelerator.device), "cudaPeakMemoryBytes": int(deps["torch"].cuda.max_memory_allocated()) if trainer.accelerator.device.type == "cuda" else None, "before": before, "after": after})
    emit({"type": "artifact", "kind": "summary", "path": "summary.json"})
    write_json(args.run_dir, "study-evaluation.json", {"version": 1, "protocolSha256": protocol["protocolSha256"], "samples": samples})
    emit({"type": "artifact", "kind": "evaluation", "path": "study-evaluation.json"})
    emit({"type": "done", "status": "completed"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--warm-start-adapter")
    try:
        with redirect_stdout(sys.stderr):
            run(parser.parse_args())
        return 0
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit({"type": "error", "code": "RunnerException", "detail": str(error)[:2048]})
        emit({"type": "done", "status": "failed"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
