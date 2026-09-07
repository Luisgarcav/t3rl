#!/usr/bin/env python3
"""Native TRL SFT/DPO adapter behind the common T3RL worker protocol."""

from __future__ import annotations

import argparse
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

from checkpointing import (
    CheckpointPublisher,
    checkpoint_policy,
    compatibility,
    dataset_cursor,
    evaluate_before_training,
    model_identity,
    transformers_checkpoint_callback,
)
from offline import (
    OFFLINE_SEED_USAGE,
    evaluate_offline_samples,
    load_offline_dataset,
    normalize_offline_metrics,
    offline_evaluation_protocol,
    offline_seed_set,
    split_offline_dataset,
    validate_offline_evaluation_protocol,
)
from rlvr import _boolean, _float, _int, emit, write_json
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
    for name in ("maxSteps", "evaluationRows", "maxSequenceLength", "perDeviceTrainBatchSize", "gradientAccumulationSteps", "loggingSteps", "checkpointCadenceSteps", "loraRank", "gracefulCheckpointDeadlineSeconds"):
        config[name] = _int(name, config[name], 1, 1_000_000)
    config["maxIntermediateCheckpoints"] = _int("maxIntermediateCheckpoints", config["maxIntermediateCheckpoints"], 0, 64)
    for name in ("keepBest", "keepFinal", "useRslora"):
        config[name] = _boolean(name, config[name])
    for name, maximum in (("learningRate", 1.0), ("loraDropout", 1.0), ("loraAlpha", 1_000_000.0)):
        config[name] = _float(name, config[name], 0.0, maximum)
    if config["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16, or bf16")
    if config["quantization"] != "none":
        raise ValueError("the offline TRL adapter supports quantization=none")
    if config["keepBest"]:
        raise ValueError("keepBest requires a declared selection metric and is not enabled here")
    if config["chatTemplate"] != {"source": "tokenizer", "sha256": None}:
        raise ValueError("custom chat template evidence is not supported by this offline adapter")
    return config


def load_dependencies() -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    return locals()


def run(args: argparse.Namespace) -> None:
    if args.resume_checkpoint and args.warm_start_adapter:
        raise ValueError("resume checkpoint and warm-start adapter are mutually exclusive")
    config = resolve_config(json.loads(args.config_json))
    seeds = offline_seed_set(args.seed, os.environ.get("T3RL_SEED_SET_JSON"))
    deps = load_dependencies()
    runner_version = metadata.version("trl")
    emit({"type": "hello", "protocol": 2, "runner": RUNNER_ID, "runnerVersion": runner_version})
    records, dataset_sha = load_offline_dataset(config["projectDatasetPath"], config["method"], config["datasetFormat"])
    train_rows, eval_rows = split_offline_dataset(records, config["evaluationRows"])
    revision = resolve_model_revision(deps, config["modelId"], config["modelRevision"])
    tokenizer_revision = resolve_model_revision(deps, config["modelId"], config["tokenizerRevision"])
    evaluation_protocol = offline_evaluation_protocol(dataset_sha, eval_rows, {**config, "modelRevision": revision, "tokenizerRevision": tokenizer_revision})
    validate_offline_evaluation_protocol(os.environ.get("T3RL_EVALUATION_PROTOCOL_JSON"), evaluation_protocol)
    identity = model_identity(config, resolved_revision=revision, tokenizer_revision=tokenizer_revision)
    compat = compatibility(model=identity, framework_id=RUNNER_ID, framework_version=runner_version)
    policy = checkpoint_policy(config)
    if config["precision"] in ("fp16", "bf16") and not deps["torch"].cuda.is_available():
        raise ValueError(f"precision={config['precision']} requires a CUDA device")
    if config["precision"] == "bf16" and not deps["torch"].cuda.is_bf16_supported():
        raise ValueError("precision=bf16 requires a CUDA device with bf16 support")
    emit({"type": "manifest", "model": identity, "checkpointPolicy": policy, "values": {**config, "seed": args.seed, "seeds": seeds, "seedUsage": OFFLINE_SEED_USAGE, "datasetSha256": dataset_sha, "datasetRows": len(records), "trainingRows": len(train_rows), "evaluationRows": len(eval_rows), "evaluationClaim": config["evaluationClaim"], "evaluationProtocol": evaluation_protocol, "chatTemplateEvidence": config["chatTemplate"], "dependencies": dependency_evidence(deps)}})

    deps["set_seed"](args.seed)
    dtype = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[config["precision"]]
    model = deps["AutoModelForCausalLM"].from_pretrained(config["modelId"], revision=revision, dtype=dtype, trust_remote_code=False)
    if args.warm_start_adapter:
        model = deps["PeftModel"].from_pretrained(model, args.warm_start_adapter, is_trainable=True)
    tokenizer = deps["AutoTokenizer"].from_pretrained(config["modelId"], revision=tokenizer_revision, trust_remote_code=False)
    peft_config = deps["LoraConfig"](r=config["loraRank"], lora_alpha=config["loraAlpha"], lora_dropout=config["loraDropout"], bias=config["loraBias"], target_modules=config["loraTargetModules"], modules_to_save=config["loraModulesToSave"] or None, use_rslora=config["useRslora"], task_type="CAUSAL_LM")
    output_dir = str(Path(args.run_dir) / "trainer")
    common = dict(output_dir=output_dir, max_steps=config["maxSteps"], learning_rate=config["learningRate"], per_device_train_batch_size=config["perDeviceTrainBatchSize"], per_device_eval_batch_size=1, gradient_accumulation_steps=config["gradientAccumulationSteps"], logging_steps=config["loggingSteps"], save_strategy="steps", save_steps=config["checkpointCadenceSteps"], save_total_limit=max(1, config["maxIntermediateCheckpoints"] + 2), save_only_model=False, eval_strategy="steps", eval_steps=config["checkpointCadenceSteps"], report_to="none", seed=args.seed, data_seed=seeds["data"], bf16=config["precision"] == "bf16", fp16=config["precision"] == "fp16", disable_tqdm=True)
    if config["method"] == "sft":
        training_args = deps["SFTConfig"](**common, max_length=config["maxSequenceLength"], packing=False, padding_free=False)
        trainer = deps["SFTTrainer"](model=model, args=training_args, train_dataset=deps["Dataset"].from_list(train_rows), eval_dataset=deps["Dataset"].from_list(eval_rows), processing_class=tokenizer, peft_config=None if args.warm_start_adapter else peft_config)
    else:
        training_args = deps["DPOConfig"](**common, max_length=config["maxSequenceLength"])
        trainer = deps["DPOTrainer"](model=model, args=training_args, train_dataset=deps["Dataset"].from_list(train_rows), eval_dataset=deps["Dataset"].from_list(eval_rows), processing_class=tokenizer, peft_config=None if args.warm_start_adapter else peft_config)
    publisher = CheckpointPublisher(run_dir=args.run_dir, compatibility_evidence=compat, policy=policy, emit_artifact=emit)
    publisher.install_signal_handlers()
    callback = transformers_checkpoint_callback(callback_base=deps["TrainerCallback"], publisher=publisher, trainer_output_dir=output_dir, max_steps=config["maxSteps"], effective_batch_size=config["perDeviceTrainBatchSize"] * config["gradientAccumulationSteps"])
    trainer.add_callback(callback)
    started = time.monotonic()
    try:
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
        shutil.rmtree(output_dir, ignore_errors=True)
    emit({"type": "metrics", "step": step, "wallClockMs": int((time.monotonic() - started) * 1000), "values": normalize_offline_metrics(config["method"], "eval_after", after)})
    write_json(args.run_dir, "summary.json", {"method": config["method"], "evaluationClaim": config["evaluationClaim"], "datasetSha256": dataset_sha, "seed": args.seed, "seeds": seeds, "seedUsage": OFFLINE_SEED_USAGE, "dataSeedApplied": trainer.args.data_seed, "optimizerSteps": step, "device": str(trainer.accelerator.device), "cudaPeakMemoryBytes": int(deps["torch"].cuda.max_memory_allocated()) if trainer.accelerator.device.type == "cuda" else None, "before": before, "after": after})
    emit({"type": "artifact", "kind": "summary", "path": "summary.json"})
    write_json(args.run_dir, "study-evaluation.json", {"version": 1, "protocolSha256": evaluation_protocol["protocolSha256"], "samples": samples})
    emit({"type": "artifact", "kind": "evaluation", "path": "study-evaluation.json"})
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
        with redirect_stdout(sys.stderr):
            run(args)
        return 0
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit({"type": "error", "code": "RunnerException", "detail": str(error)[:2048]})
        emit({"type": "done", "status": "failed"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
