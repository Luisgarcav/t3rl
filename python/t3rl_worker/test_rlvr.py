import io
import json
import math
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import rlvr


class RlvrUnitTest(unittest.TestCase):
    def test_emit_keeps_protocol_separate_from_library_stdout(self) -> None:
        protocol = io.StringIO()
        library_output = io.StringIO()
        with (
            mock.patch.object(rlvr, "PROTOCOL_STDOUT", protocol),
            redirect_stdout(library_output),
        ):
            print("trainer progress")
            rlvr.emit({"type": "done", "status": "completed"})

        self.assertEqual(library_output.getvalue(), "trainer progress\n")
        self.assertEqual(
            json.loads(protocol.getvalue()),
            {"status": "completed", "type": "done"},
        )

    def test_summary_only_callback_does_not_replace_observed_metrics(self) -> None:
        self.assertFalse(
            rlvr.has_observed_metric(
                {
                    "train/reward": None,
                    "train/kl": None,
                    "system/generated_tokens_upper_bound": 256.0,
                }
            )
        )
        self.assertTrue(
            rlvr.has_observed_metric(
                {
                    "train/reward": 0.0,
                    "train/kl": 0.0,
                    "system/generated_tokens_upper_bound": 256.0,
                }
            )
        )
        self.assertTrue(
            rlvr.has_observed_metric({"eval/reward": 0.5, "train/reward": None})
        )

    def test_exact_integer_verifier_uses_the_final_integer(self) -> None:
        self.assertEqual(
            rlvr.extract_final_integer("Work: 5 + 7. Final: 12"), "12"
        )
        self.assertEqual(rlvr.extract_final_integer("-1,024"), "-1024")
        self.assertIsNone(rlvr.extract_final_integer("no integer"))

        ledger = rlvr.EvidenceLedger(6)
        reward = rlvr.make_exact_integer_reward(ledger)
        values = reward(
            completions=["The answer is 12", "I think 8"],
            answer=["12", "9"],
            prompts=["7 + 5", "18 - 9"],
        )
        self.assertEqual(values, [1.0, 0.0])
        samples = ledger.samples
        self.assertTrue(samples[0]["verifier"]["passed"])
        self.assertFalse(samples[1]["verifier"]["passed"])
        self.assertEqual(samples[0]["phase"], "training")
        with self.assertRaises(ValueError):
            reward(completions=["12"], answer=[])

    def test_verifier_labels_before_and_after_evaluation_evidence(self) -> None:
        ledger = rlvr.EvidenceLedger(6)
        reward = rlvr.make_exact_integer_reward(ledger)
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

        before = ledger.summarize("evaluation-before")
        after = ledger.summarize("evaluation-after")
        self.assertEqual(before["verifierPassRate"], 1.0)
        self.assertEqual(after["verifierPassRate"], 0.0)
        self.assertEqual(before["rewardStd"], 0.0)

    def test_evidence_budget_does_not_truncate_the_measurement(self) -> None:
        ledger = rlvr.EvidenceLedger(3)
        reward = rlvr.make_exact_integer_reward(ledger)

        reward(completions=["1"] * 40, answer=["1"] * 40)

        training = ledger.summarize("training")
        self.assertEqual(training["sampleCount"], 40)
        self.assertEqual(training["verifierPassRate"], 1.0)
        self.assertEqual(len(training["samples"]), 1)

    def test_training_evidence_cannot_starve_the_after_evaluation(self) -> None:
        ledger = rlvr.EvidenceLedger(9)
        reward = rlvr.make_exact_integer_reward(ledger)

        reward(
            completions=["12"],
            answer=["12"],
            evidencePhase=["evaluation"],
            trainer_state=mock.Mock(global_step=0),
        )
        reward(completions=["1"] * 50, answer=["1"] * 50)
        reward(
            completions=["11"],
            answer=["12"],
            evidencePhase=["evaluation"],
            trainer_state=mock.Mock(global_step=8),
        )

        after = ledger.summarize("evaluation-after")
        self.assertEqual(after["sampleCount"], 1)
        self.assertEqual(after["verifierPassRate"], 0.0)
        self.assertEqual(len(after["samples"]), 1)

    def test_reward_statistics_cover_every_completion(self) -> None:
        ledger = rlvr.EvidenceLedger(3)
        reward = rlvr.make_exact_integer_reward(ledger)

        reward(completions=["1", "2", "1", "2"], answer=["1", "1", "1", "1"])

        training = ledger.summarize("training")
        self.assertEqual(training["sampleCount"], 4)
        self.assertEqual(training["rewardMean"], 0.5)
        self.assertEqual(training["rewardStd"], 0.5)
        self.assertEqual(training["verifierPassRate"], 0.5)

    def test_builtin_dataset_is_versioned_and_valid(self) -> None:
        records, digest = rlvr.load_builtin_dataset("arithmetic-rlvr-v1")
        self.assertEqual(len(records), 16)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(records[0], {"prompt": "What is 7 + 5?", "answer": "12"})
        training, evaluation = rlvr.split_dataset(records, 4)
        self.assertEqual(len(training), 12)
        self.assertEqual(len(evaluation), 4)
        self.assertEqual(evaluation[0]["prompt"], "What is 34 + 68?")
        with self.assertRaises(ValueError):
            rlvr.split_dataset(records, len(records))

    def test_non_finite_metrics_use_explicit_markers(self) -> None:
        self.assertEqual(rlvr.finite_metric(math.nan), "nan")
        self.assertEqual(rlvr.finite_metric(math.inf), "+inf")
        self.assertEqual(rlvr.finite_metric(-math.inf), "-inf")
        self.assertIsNone(rlvr.finite_metric(None))

    def test_reward_accepts_an_explicit_phase_from_the_backend(self) -> None:
        ledger = rlvr.EvidenceLedger(9)
        step = {"value": 0}
        reward = rlvr.make_exact_integer_reward(
            ledger,
            phase_resolver=lambda requested: "training"
            if requested != "evaluation"
            else ("evaluation-before" if step["value"] == 0 else "evaluation-after"),
        )

        reward(completions=["12"], answer=["12"], evidencePhase=["evaluation"])
        reward(completions=["7"], answer=["7"], evidencePhase=["training"])
        step["value"] = 8
        reward(completions=["11"], answer=["12"], evidencePhase=["evaluation"])

        self.assertEqual(ledger.summarize("evaluation-before")["verifierPassRate"], 1.0)
        self.assertEqual(ledger.summarize("training")["verifierPassRate"], 1.0)
        self.assertEqual(ledger.summarize("evaluation-after")["verifierPassRate"], 0.0)


if __name__ == "__main__":
    unittest.main()
