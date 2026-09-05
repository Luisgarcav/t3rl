import json
import tempfile
import unittest
from pathlib import Path

import offline


class OfflineAdapterTest(unittest.TestCase):
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

    def test_dpo_rejects_unpaired_preferences_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "bad.json"
            dataset.write_text(json.dumps([
                {"prompt": "p1", "chosen": "same", "rejected": "same"},
                {"prompt": "p2", "chosen": "yes", "rejected": "no"},
            ]))
            with self.assertRaisesRegex(ValueError, "must differ"):
                offline.load_offline_dataset(str(dataset), "dpo", "dpo-preference")

    def test_axolotl_translation_uses_one_public_method(self) -> None:
        config = {"method": "dpo", "modelId": "m", "modelRevision": "r", "loraRank": 8, "loraAlpha": 16, "loraDropout": 0.0, "maxSequenceLength": 128, "perDeviceTrainBatchSize": 1, "gradientAccumulationSteps": 1, "learningRate": 1e-5, "maxSteps": 2}
        translated = offline.axolotl_offline_config(config, "train.jsonl", "eval.jsonl", "out")
        self.assertEqual(translated["rl"], "dpo")
        self.assertEqual(translated["datasets"][0]["type"], "bradley_terry")


if __name__ == "__main__":
    unittest.main()
