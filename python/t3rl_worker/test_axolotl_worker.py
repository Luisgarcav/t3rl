import unittest

import axolotl_worker


class AxolotlWorkerUnitTest(unittest.TestCase):
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
            numGenerations=4,
            perDeviceTrainBatchSize=4,
            temperature=0.7,
            beta=0.02,
        )
        self.assertEqual(built["max_steps"], 3)
        self.assertEqual(built["trl"]["num_generations"], 4)
        self.assertEqual(built["trl"]["temperature"], 0.7)
        self.assertEqual(built["trl"]["beta"], 0.02)
        self.assertEqual(built["micro_batch_size"], 4)
        self.assertEqual(built["gradient_accumulation_steps"], 1)

    def test_config_points_at_the_written_splits(self) -> None:
        built = self.built()
        self.assertEqual(built["datasets"][0]["path"], "/tmp/run/train.jsonl")
        self.assertEqual(built["test_datasets"][0]["path"], "/tmp/run/eval.jsonl")


    def test_config_makes_the_trainer_run_both_evaluations(self) -> None:
        built = self.built(maxSteps=8)
        self.assertIs(built["eval_on_start"], True)
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
            [{"prompt": "What is 1 + 1?", "answer": "2"}], "system", "evaluation"
        )
        self.assertEqual(rows[0]["evidencePhase"], "evaluation")
        self.assertEqual(rows[0]["answer"], "2")


if __name__ == "__main__":
    unittest.main()
