import argparse
import unittest

import axolotl_offline_worker
import trl_offline_worker


class AxolotlOfflineWorkerTest(unittest.TestCase):
    def test_translation_preserves_seed_revision_precision_and_save_policy(self) -> None:
        config = trl_offline_worker.resolve_config({
            "projectDatasetPath": "/data.json", "modelId": "local-model",
            "modelRevision": "a" * 40, "tokenizerRevision": "a" * 40,
            "precision": "fp32", "loraTargetModules": ["q_proj"],
        })
        translated = axolotl_offline_worker.build_config(config, "/run", 19)
        self.assertEqual(translated["seed"], 19)
        self.assertEqual(translated["revision_of_model"], config["modelRevision"])
        self.assertFalse(translated["bf16"])
        self.assertFalse(translated["fp16"])
        self.assertFalse(translated["save_only_model"])
        self.assertEqual(translated["lora_target_modules"], ["q_proj"])
        self.assertNotEqual(translated["datasets"][0]["path"], translated["test_datasets"][0]["path"])
        self.assertFalse(translated["shuffle_before_merging_datasets"])

    def test_dpo_keeps_normalized_preference_records_without_an_extra_transform(self) -> None:
        config = trl_offline_worker.resolve_config({
            "projectDatasetPath": "/data.json", "method": "dpo",
            "datasetFormat": "dpo-preference", "evaluationClaim": "preference-accuracy",
        })
        translated = axolotl_offline_worker.build_config(config, "/run", 7)
        self.assertEqual(translated["rl"], "dpo")
        self.assertIsNone(translated["datasets"][0]["type"])

    def test_resume_and_warm_start_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            axolotl_offline_worker.run(argparse.Namespace(resume_checkpoint="/checkpoint", warm_start_adapter="/adapter"))


if __name__ == "__main__":
    unittest.main()
