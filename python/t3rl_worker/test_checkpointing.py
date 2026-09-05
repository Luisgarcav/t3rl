import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from checkpointing import (
    CheckpointPublisher,
    checkpoint_policy,
    evaluate_before_training,
    load_fixture_adapter,
    model_identity,
    sha256_json,
    transformers_checkpoint_callback,
)


class CheckpointingUnitTest(unittest.TestCase):
    def test_resume_checkpoint_is_loaded_before_the_starting_evaluation(self) -> None:
        class Trainer:
            def __init__(self) -> None:
                self.events = []

            def _load_from_checkpoint(self, path: str) -> None:
                self.events.append(("load", path))

            def evaluate(self) -> str:
                self.events.append(("evaluate", None))
                return "evidence"

        trainer = Trainer()
        result = evaluate_before_training(trainer, "/run/inputs/checkpoint")

        self.assertEqual(result, "evidence")
        self.assertEqual(
            trainer.events,
            [("load", "/run/inputs/checkpoint"), ("evaluate", None)],
        )

    def test_fresh_run_evaluates_without_loading_a_checkpoint(self) -> None:
        class Trainer:
            def __init__(self) -> None:
                self.events = []

            def evaluate(self) -> None:
                self.events.append("evaluate")

        trainer = Trainer()
        evaluate_before_training(trainer, None)

        self.assertEqual(trainer.events, ["evaluate"])

    def test_model_identity_hashes_the_canonical_peft_configuration(self) -> None:
        config = {
            "modelId": "test/model",
            "quantization": "none",
            "precision": "fp32",
            "loraRank": 2,
            "loraAlpha": 4,
            "loraDropout": 0.0,
            "loraBias": "none",
            "loraTargetModules": ["linear"],
            "loraModulesToSave": [],
            "useRslora": False,
        }
        identity = model_identity(
            config,
            resolved_revision="a" * 40,
            tokenizer_revision="b" * 40,
        )
        self.assertEqual(
            identity["peftConfigSha256"], sha256_json(identity["peftConfig"])
        )
        self.assertEqual(sha256_json({"alpha": 4.0}), sha256_json({"alpha": 4}))
        self.assertEqual(
            sha256_json(
                {
                    "peftType": "LORA",
                    "taskType": "CAUSAL_LM",
                    "rank": 16,
                    "alpha": 32.0,
                    "dropout": 0.05,
                    "bias": "none",
                    "targetModules": ["q_proj", "k_proj"],
                    "modulesToSave": [],
                    "useRslora": False,
                }
            ),
            "9d2c06d166a8968b564d0e72ca88fe023774e2e7f92293f185b94f814ba1ab8f",
        )

    def test_adapter_publication_is_atomic_and_independently_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir:
            emitted = []
            publisher = CheckpointPublisher(
                run_dir=run_dir,
                compatibility_evidence={"fixture": True},
                policy={"keepFinal": True},
                emit_artifact=emitted.append,
            )

            def save(directory: str) -> None:
                root = Path(directory)
                (root / "adapter_config.json").write_text(
                    json.dumps({"base_model_name_or_path": "test/model"}),
                    encoding="utf-8",
                )
                (root / "adapter_model.safetensors").write_text(
                    json.dumps({"linear.lora": 1.25}), encoding="utf-8"
                )

            relative = publisher.publish_adapter(
                save=save,
                global_step=4,
                tokens_seen=64,
                cursor={"epoch": 0.5, "batchInEpoch": 4, "sampleOffset": 4},
            )
            loaded = load_fixture_adapter(Path(run_dir) / relative)
            self.assertEqual(loaded["weights"]["linear.lora"], 1.25)
            self.assertEqual(emitted[0]["evidence"]["_tag"], "Adapter")
            self.assertEqual(list((Path(run_dir) / "adapters").glob(".*.tmp")), [])

    def test_checkpoint_policy_is_explicit(self) -> None:
        self.assertEqual(
            checkpoint_policy(
                {
                    "checkpointCadenceSteps": 4,
                    "maxIntermediateCheckpoints": 2,
                    "keepBest": False,
                    "keepFinal": True,
                    "gracefulCheckpointDeadlineSeconds": 30,
                }
            ),
            {
                "cadenceSteps": 4,
                "maxIntermediateCheckpoints": 2,
                "keepBest": False,
                "keepFinal": True,
                "gracefulDeadlineSeconds": 30,
            },
        )

    def test_incomplete_trainer_checkpoint_is_never_announced(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir:
            root = Path(run_dir)
            source = root / "trainer" / "checkpoint-4"
            source.mkdir(parents=True)
            (source / "adapter_config.json").write_text("{}", encoding="utf-8")
            (source / "adapter_model.safetensors").write_bytes(b"weights")
            (source / "trainer_state.json").write_text("{}", encoding="utf-8")
            emitted = []
            publisher = CheckpointPublisher(
                run_dir=run_dir,
                compatibility_evidence={"model": {"precision": "fp32"}},
                policy={"keepFinal": True},
                emit_artifact=emitted.append,
            )

            with self.assertRaisesRegex(ValueError, "optimizer.pt"):
                publisher.publish_checkpoint(
                    trainer_checkpoint=str(source),
                    checkpoint_class="intermediate",
                    global_step=4,
                    tokens_seen=64,
                    cursor={"epoch": 0.5, "batchInEpoch": 4, "sampleOffset": 4},
                )

            self.assertEqual(emitted, [])
            self.assertFalse(
                (root / "checkpoints" / "checkpoint-4-intermediate").exists()
            )
            self.assertEqual(list((root / "checkpoints").glob(".*.tmp")), [])

    def test_transformers_callback_turns_sigterm_intent_into_a_graceful_save(
        self,
    ) -> None:
        class Publisher:
            shutdown_requested = threading.Event()

            def __init__(self) -> None:
                self.calls = []

            def publish_checkpoint(self, **values) -> None:
                self.calls.append(values)

        publisher = Publisher()
        publisher.shutdown_requested.set()
        callback = transformers_checkpoint_callback(
            callback_base=object,
            publisher=publisher,  # type: ignore[arg-type]
            trainer_output_dir="/tmp/trainer",
            max_steps=8,
            effective_batch_size=2,
        )
        state = SimpleNamespace(global_step=4, epoch=0.5, num_input_tokens_seen=64)
        control = SimpleNamespace(should_save=False, should_training_stop=False)

        callback.on_step_end(None, state, control)
        callback.on_save(None, state, control)

        self.assertIs(control.should_save, True)
        self.assertIs(control.should_training_stop, True)
        self.assertEqual(publisher.calls[0]["checkpoint_class"], "graceful")
        self.assertEqual(publisher.calls[0]["global_step"], 4)
        self.assertEqual(publisher.calls[0]["tokens_seen"], 64)


if __name__ == "__main__":
    unittest.main()
