"""Bounded real-framework validation with a local tiny model and no model downloads.

Run with the locked TRL interpreter, e.g. ``python offline_validation.py
--device cuda --output-dir .t3/offline-validation --seeds 7 19 41``. This verifies
the worker protocol, independent PEFT loading, graceful interruption, exact
trainer resume within an explicit tolerance, held-out evidence, and an optional
zero-learning-rate ablation. It does not assert model quality on a toy dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
from importlib import metadata
from pathlib import Path
from typing import Any

from checkpointing import TRAINER_STATE_FILES, sha256_json
from offline import load_offline_dataset, offline_evaluation_protocol, split_offline_dataset


def create_local_model(directory: Path) -> str:
    """Create a deterministic Llama model and tokenizer small enough for CPU tests."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    directory.mkdir(parents=True)
    vocabulary = ["[PAD]", "[UNK]", "<s>", "</s>", "What", "is", "Answer", "+", "?", ":"] + [str(n) for n in range(32)]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    processing = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", bos_token="<s>", eos_token="</s>")
    processing.save_pretrained(directory)
    torch.manual_seed(123)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128, bos_token_id=2, eos_token_id=3, pad_token_id=0, attention_dropout=0.0))
    model.save_pretrained(directory, safe_serialization=True)
    return sha256_json({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir()) if path.is_file()})


def run_worker(directory: Path, config: dict[str, Any], seed: int, environment: dict[str, str], *, resume: Path | None = None, interrupt: bool = False, backend: str = "trl") -> list[dict[str, Any]]:
    """Capture the spawned worker's NDJSON and request SIGTERM at a save receipt."""
    directory.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).with_name(f"{backend}_offline_worker.py")), "--run-dir", str(directory), "--seed", str(seed), "--config-json", json.dumps(config)]
    if resume is not None:
        command.extend(["--resume-checkpoint", str(resume)])
    events = []
    with (directory / "stderr.log").open("w") as stderr, (directory / "protocol.ndjson").open("w") as protocol:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, text=True, env=environment)
        deadline = threading.Timer(180, process.kill)
        deadline.start()
        sent_signal = False
        try:
            if process.stdout is None:
                raise AssertionError("worker stdout was not captured")
            for line in process.stdout:
                protocol.write(line)
                protocol.flush()
                event = json.loads(line)
                events.append(event)
                if interrupt and not sent_signal and event.get("kind") == "checkpoint":
                    process.send_signal(signal.SIGTERM)
                    sent_signal = True
            status = process.wait(timeout=10)
        finally:
            deadline.cancel()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            if process.stdout is not None:
                process.stdout.close()
    if status != 0 or any(event.get("type") == "error" for event in events):
        raise AssertionError(f"worker failed ({status}); see {directory / 'stderr.log'}: {events[-2:]}")
    if not events or events[0].get("type") != "hello":
        raise AssertionError("worker did not begin with protocol hello")
    if interrupt:
        if not sent_signal or not any(event.get("evidence", {}).get("checkpointClass") == "graceful" for event in events):
            raise AssertionError("SIGTERM did not produce a graceful checkpoint")
        if any(event.get("type") == "done" and event.get("status") == "completed" for event in events):
            raise AssertionError("interrupted training was reported completed")
    elif events[-1] != {"type": "done", "status": "completed"}:
        raise AssertionError("worker did not complete")
    return events


def artifact_path(directory: Path, events: list[dict[str, Any]], kind: str, checkpoint_class: str | None = None) -> Path:
    matches = [event for event in events if event.get("kind") == kind and (checkpoint_class is None or event.get("evidence", {}).get("checkpointClass") == checkpoint_class)]
    if len(matches) != 1:
        raise AssertionError(f"expected one {kind}/{checkpoint_class}, found {len(matches)}")
    return directory / matches[0]["path"]


def validate_method(root: Path, method: str, model_dir: Path, revision: str, seed: int, environment: dict[str, str], ablation: bool, backend: str, data_seed_offset: int) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = root / f"{method}-seed-{seed}"
    directory.mkdir()
    dataset = directory / "dataset.json"
    rows = []
    for index in range(10):
        prompt = f"What is {index} + 1 ? Answer : "
        rows.append({"sampleId": f"arithmetic-{index}", **({"text": f"{prompt}{index + 1} </s>"} if method == "sft" else {"prompt": prompt, "chosen": str(index + 1), "rejected": str(index + 2)})})
    dataset.write_text(json.dumps(rows), encoding="utf-8")
    config = {"method": method, "datasetFormat": "sft-text" if method == "sft" else "dpo-preference", "evaluationClaim": "held-out-loss" if method == "sft" else "preference-accuracy", "projectDatasetPath": str(dataset), "modelId": str(model_dir), "modelRevision": revision, "tokenizerRevision": revision, "maxSteps": 6, "evaluationRows": 2, "maxSequenceLength": 48, "learningRate": 0.005, "checkpointCadenceSteps": 2, "precision": "fp32", "loraRank": 4, "loraAlpha": 8.0}
    records, dataset_sha = load_offline_dataset(str(dataset), method, config["datasetFormat"])
    train_rows, eval_rows = split_offline_dataset(records, 2)
    make_protocol = offline_evaluation_protocol
    if backend == "axolotl":
        from axolotl_offline_worker import evaluation_protocol
        make_protocol = evaluation_protocol
    protocol = make_protocol(dataset_sha, eval_rows, config)
    seeds = {"training": seed, "data": seed + data_seed_offset, "evaluationSample": seed, "generation": seed}
    env = {**environment, "T3RL_EVALUATION_PROTOCOL_JSON": json.dumps(protocol), "T3RL_SEED_SET_JSON": json.dumps(seeds)}
    full = directory / "continuous"
    full_events = run_worker(full, config, seed, env, backend=backend)
    stopped = directory / "interrupted"
    stopped_events = run_worker(stopped, config, seed, env, interrupt=True, backend=backend)
    checkpoint = artifact_path(stopped, stopped_events, "checkpoint", "graceful")
    for name in TRAINER_STATE_FILES:
        if not (checkpoint / name).is_file():
            raise AssertionError(f"missing resumable state: {name}")
    stopped_state = json.loads((checkpoint / "trainer_state.json").read_text())
    if not 0 < stopped_state["global_step"] < config["maxSteps"]:
        raise AssertionError("interruption must occur before the final optimizer step")
    resumed = directory / "resumed"
    resumed_events = run_worker(resumed, config, seed, env, resume=checkpoint, backend=backend)
    full_adapter = artifact_path(full, full_events, "adapter")
    resumed_adapter = artifact_path(resumed, resumed_events, "adapter")
    weights = load_file(str(full_adapter / "adapter_model.safetensors"))
    resumed_weights = load_file(str(resumed_adapter / "adapter_model.safetensors"))
    if weights.keys() != resumed_weights.keys():
        raise AssertionError("resume changed adapter parameter names")
    if not any(bool(value.abs().max() > 0) for name, value in weights.items() if "lora_B" in name):
        raise AssertionError("training did not update any LoRA B weights")
    maximum_error = 0.0
    for name, value in weights.items():
        torch.testing.assert_close(value, resumed_weights[name], atol=1e-6, rtol=1e-5)
        maximum_error = max(maximum_error, float((value - resumed_weights[name]).abs().max()))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_dir), full_adapter).eval()
    with torch.inference_mode():
        logits = model(**tokenizer("What is 8 + 1 ? Answer :", return_tensors="pt")).logits
    if not torch.isfinite(logits).all():
        raise AssertionError("independently loaded PEFT adapter produced invalid logits")
    summary = json.loads((full / "summary.json").read_text())
    resumed_summary = json.loads((resumed / "summary.json").read_text())
    if summary["seeds"] != seeds or summary["dataSeedApplied"] != seeds["data"] or resumed_summary["dataSeedApplied"] != seeds["data"]:
        raise AssertionError("the worker did not apply the declared independent seed set")
    evaluation = json.loads((full / "study-evaluation.json").read_text())
    if evaluation["protocolSha256"] != protocol["protocolSha256"]:
        raise AssertionError("evaluation protocol hash mismatch")
    if [sample["sampleId"] for sample in evaluation["samples"]] != protocol["sampleIds"]:
        raise AssertionError("held-out sample coverage mismatch")
    metric = "eval_after/loss" if method == "sft" else "eval_after/preference_accuracy"
    aggregate_key = "eval_loss" if method == "sft" else "eval_rewards/accuracies"
    sample_mean = sum(sample["values"][metric] for sample in evaluation["samples"]) / len(eval_rows)
    if not math.isclose(sample_mean, summary["after"][aggregate_key], abs_tol=1e-6, rel_tol=1e-5):
        raise AssertionError("sample evidence does not reproduce aggregate evaluation")
    if not math.isclose(summary["after"]["eval_loss"], resumed_summary["after"]["eval_loss"], abs_tol=1e-6, rel_tol=1e-5):
        raise AssertionError("resumed held-out loss exceeds tolerance")
    result = {"method": method, "seed": seed, "seeds": seeds, "seedUsage": summary["seedUsage"], "dataSeedApplied": summary["dataSeedApplied"], "config": config, "evaluationProtocol": protocol, "trainingSampleIds": [row["sampleId"] for row in train_rows], "continuous": str(full.relative_to(root)), "resumed": str(resumed.relative_to(root)), "interruptedAtStep": stopped_state["global_step"], "adapterReloaded": True, "resumeMaximumParameterError": maximum_error, "resumeTolerance": {"absolute": 1e-6, "relative": 1e-5}, "optimizerSteps": summary["optimizerSteps"], "device": summary["device"], "cudaPeakMemoryBytes": summary["cudaPeakMemoryBytes"], "before": summary["before"], "after": summary["after"], "samples": evaluation["samples"]}
    if ablation:
        ablated = directory / "zero-learning-rate"
        run_worker(ablated, {**config, "learningRate": 0.0}, seed, env, backend=backend)
        ablated_summary = json.loads((ablated / "summary.json").read_text())
        if not math.isclose(ablated_summary["before"]["eval_loss"], ablated_summary["after"]["eval_loss"], abs_tol=1e-6):
            raise AssertionError("zero-learning-rate ablation changed evaluation loss")
        result["ablation"] = {"learningRate": 0.0, "before": ablated_summary["before"], "after": ablated_summary["after"]}
    return result


def validate(output_dir: Path, device: str, seeds: list[int], *, ablation: bool = False, backend: str = "trl", data_seed_offset: int = 0) -> dict[str, Any]:
    import torch
    from trl_worker import dependency_evidence

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested, but CUDA is unavailable")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "source-snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for name in ("offline.py", "trl_offline_worker.py", "axolotl_offline_worker.py", "checkpointing.py", "trl_worker.py", "rlvr.py", "offline_validation.py"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, snapshot / name)
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (snapshot / "source-hashes.json").write_text(json.dumps(source_hashes, indent=2))
    model_dir = output_dir / "model"
    revision = create_local_model(model_dir)
    dependencies = dependency_evidence({"torch": torch})
    if backend == "axolotl":
        dependencies["axolotl"] = metadata.version("axolotl")
    lock = Path(__file__).parents[1] / "environments" / backend / "uv.lock"
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    lock_text = lock.read_text()
    for dependency in ("torch", "trl", "peft", "transformers", "datasets", "accelerate", *(["axolotl"] if backend == "axolotl" else [])):
        locked = re.search(r'\[\[package\]\]\nname = "' + dependency + r'"\nversion = "([^"]+)"', lock_text)
        if locked is None or dependencies[dependency] != locked.group(1):
            raise RuntimeError(f"{dependency} does not match the checked-in {backend} lock ({lock}); use its frozen environment")
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "AXOLOTL_DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "T3RL_ENVIRONMENT_LOCK_SHA256": lock_sha, "T3RL_ENVIRONMENT_FINGERPRINT": sha256_json({"dependencies": dependencies, "lockSha256": lock_sha})}
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    for method in ("sft", "dpo"):
        for seed in seeds:
            result = validate_method(output_dir, method, model_dir, revision, seed, environment, ablation, backend, data_seed_offset)
            if not result["device"].startswith(device):
                raise AssertionError(f"requested {device}, trainer used {result['device']}")
            if device == "cuda" and not result["cudaPeakMemoryBytes"] > 0:
                raise AssertionError("CUDA validation did not record GPU allocations")
            results.append(result)
            print(f"validated {method} seed={seed} device={device}", flush=True)
    report = {"version": 1, "status": "passed", "backend": backend, "device": device, "dependencies": dependencies, "sourceSha256": source_hashes, "dataSeedOffset": data_seed_offset, "environmentLockSha256": lock_sha, "modelRevision": revision, "scope": "local tiny-model real trainer/PEFT lifecycle; does not establish model quality or cross-hardware determinism", "results": results}
    (output_dir / "validation-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--backend", choices=("trl", "axolotl"), default="trl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--data-seed-offset", type=int, default=0)
    args = parser.parse_args()
    validate(args.output_dir, args.device, args.seeds, ablation=args.ablation, backend=args.backend, data_seed_offset=args.data_seed_offset)
