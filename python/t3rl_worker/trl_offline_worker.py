#!/usr/bin/env python3
"""Native TRL SFT/DPO adapter behind the common T3RL worker protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from checkpointing import (
    CheckpointPublisher,
    checkpoint_policy,
    compatibility,
    dataset_cursor,
    model_identity,
    transformers_checkpoint_callback,
)
from offline import load_offline_dataset, normalize_offline_metrics, split_offline_dataset
from rlvr import emit, write_json
from trl_worker import dependency_evidence, resolve_model_revision

RUNNER_ID = "trl"


def resolve_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("config must be an object")
    defaults = {
        "method": "sft", "datasetFormat": "sft-text", "evaluationClaim": "held-out-loss",
        "projectDatasetPath": None, "projectVerifierPath": None, "modelId": "", "modelRevision": "", "tokenizerRevision": "",
        "maxSteps": 8, "evaluationRows": 2, "maxSequenceLength": 512,
        "perDeviceTrainBatchSize": 1, "gradientAccumulationSteps": 1,
        "learningRate": 5e-6, "loggingSteps": 1, "checkpointCadenceSteps": 4,
        "maxIntermediateCheckpoints": 2, "keepBest": False, "keepFinal": True,
        "gracefulCheckpointDeadlineSeconds": 30, "precision": "fp32", "quantization": "none",
        "loraRank": 8, "loraAlpha": 16.0, "loraDropout": 0.0, "loraBias": "none",
        "loraTargetModules": ["q_proj", "v_proj"], "loraModulesToSave": [], "useRslora": False,
        "chatTemplate": {"source": "tokenizer", "sha256": None},
    }
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    config = {**defaults, **raw}
    if config["method"] not in ("sft", "dpo"):
        raise ValueError("method must be sft or dpo")
    expected = "held-out-loss" if config["method"] == "sft" else "preference-accuracy"
    if config["evaluationClaim"] != expected:
        raise ValueError(f"{config['method']} requires evaluationClaim={expected}")
    if not isinstance(config["projectDatasetPath"], str):
        raise ValueError("offline training requires a snapshotted projectDatasetPath")
    return config


def load_dependencies() -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    return locals()


def run(args: argparse.Namespace) -> None:
    config = resolve_config(json.loads(args.config_json))
    deps = load_dependencies()
    runner_version = metadata.version("trl")
    emit({"type": "hello", "protocol": 2, "runner": RUNNER_ID, "runnerVersion": runner_version})
    records, dataset_sha = load_offline_dataset(config["projectDatasetPath"], config["method"], config["datasetFormat"])
    train_rows, eval_rows = split_offline_dataset(records, config["evaluationRows"])
    revision = resolve_model_revision(deps, config["modelId"], config["modelRevision"])
    tokenizer_revision = resolve_model_revision(deps, config["modelId"], config["tokenizerRevision"])
    identity = model_identity(config, resolved_revision=revision, tokenizer_revision=tokenizer_revision)
    compat = compatibility(model=identity, framework_id=RUNNER_ID, framework_version=runner_version)
    policy = checkpoint_policy(config)
    emit({"type": "manifest", "model": identity, "checkpointPolicy": policy, "values": {**config, "datasetSha256": dataset_sha, "datasetRows": len(records), "trainingRows": len(train_rows), "evaluationRows": len(eval_rows), "evaluationClaim": config["evaluationClaim"], "chatTemplateEvidence": config["chatTemplate"], "dependencies": dependency_evidence(deps)}})

    model = deps["AutoModelForCausalLM"].from_pretrained(config["modelId"], revision=revision)
    if args.warm_start_adapter:
        model = deps["PeftModel"].from_pretrained(model, args.warm_start_adapter, is_trainable=True)
    tokenizer = deps["AutoTokenizer"].from_pretrained(config["modelId"], revision=tokenizer_revision)
    peft_config = deps["LoraConfig"](r=config["loraRank"], lora_alpha=config["loraAlpha"], lora_dropout=config["loraDropout"], bias=config["loraBias"], target_modules=config["loraTargetModules"], modules_to_save=config["loraModulesToSave"] or None, use_rslora=config["useRslora"], task_type="CAUSAL_LM")
    output_dir = str(Path(args.run_dir) / "trainer")
    common = dict(output_dir=output_dir, max_steps=config["maxSteps"], learning_rate=config["learningRate"], per_device_train_batch_size=config["perDeviceTrainBatchSize"], per_device_eval_batch_size=config["perDeviceTrainBatchSize"], gradient_accumulation_steps=config["gradientAccumulationSteps"], logging_steps=config["loggingSteps"], save_strategy="steps", save_steps=config["checkpointCadenceSteps"], eval_strategy="steps", eval_steps=config["checkpointCadenceSteps"], report_to="none", seed=args.seed)
    if config["method"] == "sft":
        training_args = deps["SFTConfig"](**common, max_length=config["maxSequenceLength"])
        trainer = deps["SFTTrainer"](model=model, args=training_args, train_dataset=deps["Dataset"].from_list(train_rows), eval_dataset=deps["Dataset"].from_list(eval_rows), processing_class=tokenizer, peft_config=None if args.warm_start_adapter else peft_config)
    else:
        training_args = deps["DPOConfig"](**common, max_length=config["maxSequenceLength"])
        trainer = deps["DPOTrainer"](model=model, args=training_args, train_dataset=deps["Dataset"].from_list(train_rows), eval_dataset=deps["Dataset"].from_list(eval_rows), processing_class=tokenizer, peft_config=None if args.warm_start_adapter else peft_config)
    publisher = CheckpointPublisher(run_dir=args.run_dir, compatibility_evidence=compat, policy=policy, emit_artifact=emit)
    publisher.install_signal_handlers()
    callback = transformers_checkpoint_callback(callback_base=deps["TrainerCallback"], publisher=publisher, trainer_output_dir=output_dir, max_steps=config["maxSteps"], effective_batch_size=config["perDeviceTrainBatchSize"] * config["gradientAccumulationSteps"])
    trainer.add_callback(callback)
    before = trainer.evaluate()
    emit({"type": "metrics", "step": 0, "wallClockMs": 0, "values": normalize_offline_metrics(config["method"], "eval_before", before)})
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    after = trainer.evaluate()
    step = int(trainer.state.global_step)
    cursor = dataset_cursor(global_step=step, epoch=trainer.state.epoch, effective_batch_size=config["perDeviceTrainBatchSize"] * config["gradientAccumulationSteps"])
    publisher.publish_adapter(save=trainer.save_model, global_step=step, tokens_seen=int(getattr(trainer.state, "num_input_tokens_seen", 0)), cursor=cursor)
    emit({"type": "metrics", "step": step, "wallClockMs": int((time.monotonic() - started) * 1000), "values": normalize_offline_metrics(config["method"], "eval_after", after)})
    summary = Path(args.run_dir) / "summary.json"
    write_json(summary, {"method": config["method"], "evaluationClaim": config["evaluationClaim"], "datasetSha256": dataset_sha, "before": before, "after": after})
    emit({"type": "artifact", "kind": "summary", "path": summary.relative_to(args.run_dir).as_posix()})
    emit({"type": "done", "status": "completed"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--warm-start-adapter")
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as error:
        emit({"type": "error", "code": "RunnerException", "detail": str(error)[:2048]})
        emit({"type": "done", "status": "failed"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
