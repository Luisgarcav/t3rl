import io
import json
import math
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import trl_worker


class TrlWorkerUnitTest(unittest.TestCase):
    def test_emit_keeps_protocol_separate_from_library_stdout(self) -> None:
        protocol = io.StringIO()
        library_output = io.StringIO()
        with (
            mock.patch.object(trl_worker, "PROTOCOL_STDOUT", protocol),
            redirect_stdout(library_output),
        ):
            print("trainer progress")
            trl_worker.emit({"type": "done", "status": "completed"})

        self.assertEqual(library_output.getvalue(), "trainer progress\n")
        self.assertEqual(
            json.loads(protocol.getvalue()),
            {"status": "completed", "type": "done"},
        )

    def test_summary_only_callback_does_not_replace_observed_metrics(self) -> None:
        self.assertFalse(
            trl_worker.has_observed_metric(
                {
                    "train/reward": None,
                    "train/kl": None,
                    "system/generated_tokens_upper_bound": 256.0,
                }
            )
        )
        self.assertTrue(
            trl_worker.has_observed_metric(
                {
                    "train/reward": 0.0,
                    "train/kl": 0.0,
                    "system/generated_tokens_upper_bound": 256.0,
                }
            )
        )
        self.assertTrue(
            trl_worker.has_observed_metric({"eval/reward": 0.5, "train/reward": None})
        )

    def test_resolve_config_applies_bounded_defaults(self) -> None:
        config = trl_worker.resolve_config({})
        self.assertEqual(config["algorithm"], "GRPO")
        self.assertEqual(config["backend"], "trl")
        self.assertEqual(config["launcher"], "direct")
        self.assertEqual(config["distributedStrategy"], "single-process")
        self.assertEqual(config["worldSize"], 1)
        self.assertEqual(config["maxSteps"], 8)
        self.assertEqual(config["evaluationRows"], 4)
        self.assertEqual(config["modelId"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(
            config["modelRevision"], "7ae557604adf67be50417f59c2c2f167def9a775"
        )
        self.assertEqual(config["datasetId"], "arithmetic-rlvr-v1")

    def test_resolve_config_rejects_unknown_or_unbounded_values(self) -> None:
        invalid = [
            {"telepathy": True},
            {"algorithm": "PPO"},
            {"rewardSource": "human-preference"},
            {"modelId": "some/other-model"},
            {"useVllm": True},
            {"worldSize": 2},
            {"perDeviceTrainBatchSize": 3, "numGenerations": 2},
            {"maxGeneratedTokens": 127},
            {"maxWallClockSeconds": 1800, "maxGpuHours": 0.49},
        ]
        for config in invalid:
            with (
                self.subTest(config=config),
                self.assertRaises((TypeError, ValueError)),
            ):
                trl_worker.resolve_config(config)

    def test_exact_integer_verifier_uses_the_final_integer(self) -> None:
        self.assertEqual(
            trl_worker.extract_final_integer("Work: 5 + 7. Final: 12"), "12"
        )
        self.assertEqual(trl_worker.extract_final_integer("-1,024"), "-1024")
        self.assertIsNone(trl_worker.extract_final_integer("no integer"))

        samples = []
        reward = trl_worker.make_exact_integer_reward(samples, 2)
        values = reward(
            completions=["The answer is 12", "I think 8"],
            answer=["12", "9"],
            prompts=["7 + 5", "18 - 9"],
        )
        self.assertEqual(values, [1.0, 0.0])
        self.assertTrue(samples[0]["verifier"]["passed"])
        self.assertFalse(samples[1]["verifier"]["passed"])
        self.assertEqual(samples[0]["phase"], "training")
        with self.assertRaises(ValueError):
            reward(completions=["12"], answer=[])

    def test_verifier_labels_before_and_after_evaluation_evidence(self) -> None:
        samples = []
        reward = trl_worker.make_exact_integer_reward(samples, 4)
        reward(
            completions=["12"],
            answer=["12"],
            evidencePhase=["evaluation"],
            trainer_state=mock.Mock(global_step=0),
        )
        reward(
            completions=["11"],
            answer=["12"],
            evidencePhase=["evaluation"],
            trainer_state=mock.Mock(global_step=8),
        )

        before = trl_worker.summarize_samples(samples, "evaluation-before")
        after = trl_worker.summarize_samples(samples, "evaluation-after")
        self.assertEqual(before["verifierPassRate"], 1.0)
        self.assertEqual(after["verifierPassRate"], 0.0)
        self.assertEqual(before["rewardStd"], 0.0)

    def test_immutable_model_revision_does_not_require_registry_resolution(
        self,
    ) -> None:
        revision = "7ae557604adf67be50417f59c2c2f167def9a775"
        self.assertEqual(
            trl_worker.resolve_model_revision({}, trl_worker.SUPPORTED_MODEL, revision),
            revision,
        )

    def test_builtin_dataset_is_versioned_and_valid(self) -> None:
        records, digest = trl_worker.load_builtin_dataset("arithmetic-rlvr-v1")
        self.assertEqual(len(records), 16)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(records[0], {"prompt": "What is 7 + 5?", "answer": "12"})
        training, evaluation = trl_worker.split_dataset(records, 4)
        self.assertEqual(len(training), 12)
        self.assertEqual(len(evaluation), 4)
        self.assertEqual(evaluation[0]["prompt"], "What is 34 + 68?")
        with self.assertRaises(ValueError):
            trl_worker.split_dataset(records, len(records))

    def test_catalog_trl_experiments_resolve_strictly(self) -> None:
        catalog = Path(__file__).parent / "experiments"
        definitions = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in catalog.glob("*.json")
        ]
        trl_definitions = [item for item in definitions if item["runnerId"] == "trl"]
        self.assertGreaterEqual(len(trl_definitions), 1)
        for definition in trl_definitions:
            with self.subTest(experiment=definition["experimentId"]):
                resolved = trl_worker.resolve_config(definition["config"])
                self.assertEqual(resolved["algorithm"], "GRPO")

    def test_non_finite_metrics_use_explicit_markers(self) -> None:
        self.assertEqual(trl_worker.finite_metric(math.nan), "nan")
        self.assertEqual(trl_worker.finite_metric(math.inf), "+inf")
        self.assertEqual(trl_worker.finite_metric(-math.inf), "-inf")
        self.assertIsNone(trl_worker.finite_metric(None))


if __name__ == "__main__":
    unittest.main()
