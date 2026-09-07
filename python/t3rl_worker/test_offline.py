import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import offline
import trl_offline_worker


class OfflineAdapterTest(unittest.TestCase):
    def test_independent_data_seed_reaches_trainer_and_accelerate(self) -> None:
        declared = {"training": 7, "data": 19, "evaluationSample": 23, "generation": 29}
        seeds = offline.offline_seed_set(7, json.dumps(declared))
        trainer = SimpleNamespace(args=SimpleNamespace(data_seed=7), accelerator=SimpleNamespace(dataloader_config=SimpleNamespace(data_seed=7)))
        offline.configure_offline_data_seed(trainer, seeds["data"])
        self.assertEqual(trainer.args.data_seed, 19)
        self.assertEqual(trainer.accelerator.dataloader_config.data_seed, 19)
        self.assertEqual(seeds, declared)
        with self.assertRaisesRegex(ValueError, "must match"):
            offline.offline_seed_set(8, json.dumps(declared))

    def test_sft_and_dpo_datasets_have_stable_typed_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft = root / "sft.jsonl"
            dpo = root / "dpo.json"
            sft.write_text('{"text":"one"}\n{"prompt":"ignored","text":"two"}\n')
            dpo.write_text(json.dumps([
                {"prompt": "p1", "chosen": "yes", "rejected": "no"},
                {"prompt": "p2", "chosen": "up", "rejected": "down"},
            ]))
            sft_rows, sft_hash = offline.load_offline_dataset(str(sft), "sft", "sft-text")
            dpo_rows, dpo_hash = offline.load_offline_dataset(str(dpo), "dpo", "dpo-preference")
            self.assertEqual([row["text"] for row in sft_rows], ["one", "two"])
            self.assertEqual(dpo_rows[0]["chosen"], "yes")
            self.assertEqual(len({row["sampleId"] for row in sft_rows + dpo_rows}), 4)
            self.assertEqual((len(sft_hash), len(dpo_hash)), (64, 64))

    def test_normalized_metrics_preserve_missing_values(self) -> None:
        self.assertEqual(
            offline.normalize_offline_metrics("dpo", "eval", {"eval_rewards/accuracies": 0.75}),
            {"eval/loss": None, "eval/preference_accuracy": 0.75, "eval/reward_margin": None},
        )
        self.assertEqual(
            offline.normalize_offline_metrics("sft", "eval", {"eval_loss": float("nan"), "perplexity": float("inf")}),
            {"eval/loss": None, "eval/perplexity": None},
        )

    def test_duplicate_sample_ids_cannot_cross_train_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.json"
            dataset.write_text(json.dumps([
                {"sampleId": "same", "text": "train"},
                {"sampleId": "same", "text": "test"},
            ]))
            with self.assertRaisesRegex(ValueError, "must be unique"):
                offline.load_offline_dataset(str(dataset), "sft", "sft-text")

    def test_study_protocol_must_describe_the_actual_evaluation(self) -> None:
        config = trl_offline_worker.resolve_config({"projectDatasetPath": "/data.json"})
        with mock.patch.object(offline.metadata, "version", return_value="test"):
            actual = offline.offline_evaluation_protocol("a" * 64, [{"sampleId": "held-out-1"}], config)
            truncated = offline.offline_evaluation_protocol("a" * 64, [{"sampleId": "held-out-1"}], {**config, "maxSequenceLength": 16})
        self.assertNotEqual(actual["verifierSha256"], truncated["verifierSha256"])
        offline.validate_offline_evaluation_protocol(json.dumps(actual), actual)
        for key, value in (
            ("datasetFingerprint", "b" * 64),
            ("sampleIds", ["training-1"]),
            ("split", "train"),
            ("generationSeedPolicy", "per-run"),
            ("decoding", {"temperature": 1}),
            ("verifierSha256", "b" * 64),
            ("protocolSha256", "b" * 64),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not match"):
                offline.validate_offline_evaluation_protocol(json.dumps({**actual, key: value}), actual)

    def test_offline_config_rejects_unsupported_or_misreported_training(self) -> None:
        for override in (
            {"maxSteps": 0}, {"learningRate": float("nan")}, {"precision": "tf32"},
            {"quantization": "4bit"}, {"keepBest": True},
        ):
            with self.subTest(override=override), self.assertRaises((TypeError, ValueError)):
                trl_offline_worker.resolve_config({"projectDatasetPath": "/data.json", **override})

    def test_dpo_rejects_unpaired_preferences_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "bad.json"
            dataset.write_text(json.dumps([
                {"prompt": "p1", "chosen": "same", "rejected": "same"},
                {"prompt": "p2", "chosen": "yes", "rejected": "no"},
            ]))
            with self.assertRaisesRegex(ValueError, "must differ"):
                offline.load_offline_dataset(str(dataset), "dpo", "dpo-preference")

if __name__ == "__main__":
    unittest.main()
