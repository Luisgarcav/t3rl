import contextlib
import importlib.metadata
import io
import sys
import unittest
from unittest import mock

import axolotl_worker


class AxolotlWorkerUnitTest(unittest.TestCase):
    def test_conflicting_continuation_options_emit_protocol_error(self) -> None:
        events = []
        argv = [
            "axolotl_worker.py",
            "--run-dir",
            "/tmp/run",
            "--resume-checkpoint",
            "/tmp/checkpoint",
            "--warm-start-adapter",
            "/tmp/adapter",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(axolotl_worker, "emit", events.append),
            mock.patch.object(importlib.metadata, "version", return_value="test"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = axolotl_worker.main()

        self.assertEqual(result, 1)
        self.assertEqual([event["type"] for event in events], ["hello", "error"])
        self.assertIn("mutually exclusive", events[1]["detail"])

    def resolved(self, **overrides):
        return axolotl_worker.resolve_config(overrides)

    def built(self, **overrides):
        return axolotl_worker.build_axolotl_config(
            self.resolved(**overrides),
            run_dir="/tmp/run",
            train_path="/tmp/run/train.jsonl",
            eval_path="/tmp/run/eval.jsonl",
        )

    def test_resolve_config_pins_the_axolotl_backend(self) -> None:
        self.assertEqual(self.resolved()["backend"], "axolotl")
        with self.assertRaises(ValueError):
            self.resolved(backend="trl")
        with self.assertRaises(ValueError):
            self.resolved(notAKey=1)

    def test_config_never_enables_vllm(self) -> None:
        self.assertIs(self.built()["trl"]["use_vllm"], False)
        self.assertNotIn("vllm", self.built())
        with self.assertRaises(ValueError):
            self.resolved(useVllm=True)

    def test_config_pins_the_model_revision(self) -> None:
        built = self.built()
        self.assertEqual(built["base_model"], axolotl_worker.SUPPORTED_MODEL)
        self.assertEqual(
            built["revision_of_model"], axolotl_worker.SUPPORTED_MODEL_REVISION
        )
        self.assertEqual(built["rl"], "grpo")

    def test_config_carries_the_bounded_hyperparameters(self) -> None:
        built = self.built(
            maxSteps=3,
            checkpointCadenceSteps=1,
            numGenerations=4,
            evaluationNumGenerations=4,
            evaluationBatchSize=4,
            perDeviceTrainBatchSize=4,
            temperature=0.7,
            beta=0.02,
            maxGeneratedTokens=65536,
        )
        self.assertEqual(built["max_steps"], 3)
        self.assertEqual(built["trl"]["num_generations"], 4)
        self.assertEqual(built["trl"]["temperature"], 0.7)
        self.assertEqual(built["trl"]["beta"], 0.02)
        self.assertEqual(built["micro_batch_size"], 4)
        self.assertEqual(built["eval_batch_size"], 4)
        self.assertEqual(built["gradient_accumulation_steps"], 1)
        self.assertEqual(built["adapter"], "lora")
        self.assertEqual(built["lora_r"], 16)
        self.assertEqual(built["save_strategy"], "steps")
        self.assertIs(built["save_only_model"], False)

    def test_config_distinguishes_resume_from_adapter_warm_start(self) -> None:
        config = self.resolved()
        resumed = axolotl_worker.build_axolotl_config(
            config,
            run_dir="/tmp/run",
            train_path="/tmp/run/train.jsonl",
            eval_path="/tmp/run/eval.jsonl",
            resume_checkpoint="/tmp/run/inputs/checkpoint",
        )
        warmed = axolotl_worker.build_axolotl_config(
            config,
            run_dir="/tmp/run",
            train_path="/tmp/run/train.jsonl",
            eval_path="/tmp/run/eval.jsonl",
            warm_start_adapter="/tmp/run/inputs/adapter",
        )
        self.assertEqual(
            resumed["resume_from_checkpoint"], "/tmp/run/inputs/checkpoint"
        )
        self.assertIsNone(resumed["lora_model_dir"])
        self.assertEqual(warmed["lora_model_dir"], "/tmp/run/inputs/adapter")
        self.assertIsNone(warmed["resume_from_checkpoint"])

    def test_config_points_at_the_written_splits(self) -> None:
        built = self.built()
        self.assertEqual(built["datasets"][0]["path"], "/tmp/run/train.jsonl")
        self.assertEqual(built["test_datasets"][0]["path"], "/tmp/run/eval.jsonl")

    def test_config_makes_the_trainer_run_both_evaluations(self) -> None:
        built = self.built(maxSteps=8)
        self.assertIs(built["eval_on_start"], False)
        self.assertEqual(built["eval_strategy"], "steps")
        self.assertEqual(built["eval_steps"], 8)
        self.assertIs(built["remove_unused_columns"], False)

    def test_config_keeps_every_written_path_under_the_run_directory(self) -> None:
        built = self.built()
        for key in ("output_dir", "dataset_prepared_path"):
            self.assertTrue(
                built[key].startswith("/tmp/run"), f"{key} escaped the run directory"
            )

    def test_dataset_rows_label_their_evidence_phase(self) -> None:
        rows = axolotl_worker.dataset_rows(
            [
                {
                    "sampleId": "sample_fixed",
                    "prompt": "What is 1 + 1?",
                    "answer": "2",
                }
            ],
            "system",
            "evaluation",
        )
        self.assertEqual(rows[0]["sampleId"], "sample_fixed")
        self.assertEqual(rows[0]["evidencePhase"], "evaluation")
        self.assertEqual(rows[0]["answer"], "2")

    def test_evaluation_batch_must_divide_into_generation_groups(self) -> None:
        with self.assertRaises(ValueError):
            self.resolved(evaluationBatchSize=3, evaluationNumGenerations=2)

    def test_metrics_use_the_shared_ui_names_and_system_instrumentation(self) -> None:
        metrics = axolotl_worker.normalize_grpo_metrics(
            {"eval_reward": 0.75, "eval_num_tokens": 120},
            step=4,
            evaluation_passes=1,
            config=self.resolved(),
            elapsed_seconds=2.0,
            gpu_memory_allocated_gb=3.5,
        )
        self.assertEqual(metrics["eval/reward"], 0.75)
        self.assertEqual(metrics["eval/verifier_pass_rate"], 0.75)
        self.assertNotIn("eval/eval_reward", metrics)
        self.assertEqual(metrics["system/tokens_per_second"], 60.0)
        self.assertEqual(metrics["system/gpu_memory_allocated_gb"], 3.5)

    def test_resolve_config_accepts_the_higher_resolution_dataset(self) -> None:
        config = axolotl_worker.resolve_config(
            {
                "datasetId": "arithmetic-rlvr-v2",
                "evaluationRows": 64,
                "numGenerations": 4,
                "perDeviceTrainBatchSize": 4,
                "evaluationNumGenerations": 4,
                "evaluationBatchSize": 4,
                "maxGeneratedTokens": 65536,
            }
        )
        self.assertEqual(config["datasetId"], "arithmetic-rlvr-v2")
        self.assertEqual(config["evaluationRows"], 64)
        with self.assertRaises(ValueError):
            axolotl_worker.resolve_config({"datasetId": "arithmetic-rlvr-v9"})

    def test_resolve_config_refuses_an_evaluation_it_cannot_honour(self) -> None:
        # Axolotl exposes only `num_generations`; TRL's separate eval count has
        # no config surface, so a differing value must be refused rather than
        # silently evaluated at the training count.
        with self.assertRaises(ValueError):
            self.resolved(numGenerations=2, evaluationNumGenerations=4)
        config = self.resolved(
            numGenerations=4,
            perDeviceTrainBatchSize=4,
            evaluationNumGenerations=4,
            evaluationBatchSize=4,
            maxGeneratedTokens=65536,
        )
        self.assertEqual(config["evaluationNumGenerations"], 4)


if __name__ == "__main__":
    unittest.main()
