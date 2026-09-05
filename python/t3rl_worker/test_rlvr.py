import io
import json
import math
import tempfile
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
        self.assertEqual(rlvr.extract_final_integer("Work: 5 + 7. Final: 12"), "12")
        self.assertEqual(rlvr.extract_final_integer("-1,024"), "-1024")
        self.assertEqual(rlvr.extract_final_integer("The answer is 42."), "42")
        self.assertEqual(rlvr.extract_final_integer("The answer is 1,234."), "1234")
        self.assertIsNone(rlvr.extract_final_integer("no integer"))
        self.assertIsNone(rlvr.extract_final_integer("4e42"))
        self.assertIsNone(rlvr.extract_final_integer("answer42"))
        self.assertIsNone(rlvr.extract_final_integer("12,34"))
        self.assertIsNone(rlvr.extract_final_integer("42.5"))
        self.assertIsNone(rlvr.extract_final_integer("84/7"))
        self.assertIsNone(rlvr.extract_final_integer("41-42"))

        ledger = rlvr.EvidenceLedger(6)
        reward = rlvr.make_exact_integer_reward(ledger)
        values = reward(
            completions=["The answer is 12", "I think 8", "answer12"],
            answer=["12", "9", "12"],
            prompts=["7 + 5", "18 - 9", "7 + 5"],
        )
        self.assertEqual(values, [1.0, 0.0, 0.0])
        samples = ledger.samples
        self.assertEqual(samples[0]["verifier"]["id"], "exact-integer-v2")
        self.assertRegex(samples[0]["sampleId"], r"^sample_[0-9a-f]{24}$")
        self.assertEqual(samples[0]["generationIndex"], 0)
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
        self.assertEqual(records[0]["prompt"], "What is 7 + 5?")
        self.assertEqual(records[0]["answer"], "12")
        self.assertRegex(records[0]["sampleId"], r"^sample_[0-9a-f]{24}$")
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

    def test_json_artifacts_publish_through_an_atomic_rename(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir:
            destination = Path(run_dir) / "summary.json"
            real_replace = rlvr.os.replace
            observed_temporary: list[Path] = []

            def replace(temporary: str, target: str | Path) -> None:
                temporary_path = Path(temporary)
                self.assertTrue(temporary_path.is_file())
                self.assertEqual(Path(target), destination)
                self.assertFalse(destination.exists())
                observed_temporary.append(temporary_path)
                real_replace(temporary, target)

            with mock.patch.object(rlvr.os, "replace", side_effect=replace):
                rlvr.write_json(run_dir, "summary.json", {"value": 7})

            self.assertEqual(json.loads(destination.read_text()), {"value": 7})
            self.assertEqual(len(observed_temporary), 1)
            self.assertEqual(list(Path(run_dir).glob("*.tmp")), [])

    def test_heartbeat_emits_liveness_metrics_until_stopped(self) -> None:
        stop = mock.Mock()
        stop.wait.side_effect = [False, True]
        with (
            mock.patch.object(rlvr, "emit") as emit,
            mock.patch.object(rlvr.time, "monotonic", return_value=12.0),
        ):
            rlvr.run_metrics_heartbeat(
                stop,
                started=2.0,
                step=lambda: 7,
                gpu_count=lambda: 2,
            )

        self.assertEqual(
            emit.call_args_list,
            [
                mock.call({"type": "heartbeat", "step": 7, "wallClockMs": 10_000}),
                mock.call(
                    {
                        "type": "resource",
                        "step": 7,
                        "wallClockMs": 10_000,
                        "values": {"system/gpu_count": 2.0},
                    }
                ),
            ],
        )
        self.assertEqual(stop.wait.call_args_list, [mock.call(15), mock.call(15)])

    def test_grpo_metrics_are_normalized_for_the_ui(self) -> None:
        metrics = rlvr.normalize_grpo_metrics(
            {
                "eval_reward": 0.75,
                "eval_rewards/exact_integer_reward/mean": 0.5,
                "eval_num_tokens": 120,
            },
            step=4,
            evaluation_passes=2,
            config={
                "perDeviceTrainBatchSize": 2,
                "gradientAccumulationSteps": 1,
                "maxCompletionLength": 8,
                "evaluationRows": 3,
                "evaluationNumGenerations": 2,
            },
            elapsed_seconds=2.0,
            gpu_memory_allocated_gb=3.5,
        )

        self.assertEqual(metrics["eval/reward"], 0.75)
        self.assertEqual(metrics["eval/verifier_pass_rate"], 0.5)
        self.assertNotIn("eval/eval_reward", metrics)
        self.assertEqual(metrics["system/num_tokens"], 120.0)
        self.assertEqual(metrics["system/tokens_per_second"], 60.0)
        self.assertEqual(metrics["system/gpu_memory_allocated_gb"], 3.5)
        self.assertEqual(metrics["system/generated_tokens_upper_bound"], 160.0)

    def test_replay_samples_follow_execution_order(self) -> None:
        samples = [
            {"phase": "training", "sampleId": "sample_b", "value": 3},
            {"phase": "evaluation-after", "sampleId": "sample_a", "value": 4},
            {"phase": "evaluation-before", "sampleId": "sample_a", "value": 1},
            {"phase": "training", "sampleId": "sample_a", "value": 2},
        ]
        ordered = rlvr.order_replay_samples(samples)
        self.assertEqual([sample["value"] for sample in ordered], [1, 2, 3, 4])

    def test_explicit_sample_ids_pair_generations_stably(self) -> None:
        ledger = rlvr.EvidenceLedger(6)
        reward = rlvr.make_exact_integer_reward(ledger)
        reward(
            completions=["12"],
            answer=["12"],
            sampleId=["sample_fixed"],
        )
        reward(
            completions=["11"],
            answer=["12"],
            sampleId=["sample_fixed"],
        )
        self.assertEqual(
            [
                (sample["sampleId"], sample["generationIndex"])
                for sample in ledger.samples
            ],
            [("sample_fixed", 0), ("sample_fixed", 1)],
        )

    def test_reward_accepts_an_explicit_phase_from_the_backend(self) -> None:
        ledger = rlvr.EvidenceLedger(9)
        step = {"value": 0}
        reward = rlvr.make_exact_integer_reward(
            ledger,
            phase_resolver=lambda requested: (
                "training"
                if requested != "evaluation"
                else ("evaluation-before" if step["value"] == 0 else "evaluation-after")
            ),
        )

        reward(completions=["12"], answer=["12"], evidencePhase=["evaluation"])
        reward(completions=["7"], answer=["7"], evidencePhase=["training"])
        step["value"] = 8
        reward(completions=["11"], answer=["12"], evidencePhase=["evaluation"])

        self.assertEqual(ledger.summarize("evaluation-before")["verifierPassRate"], 1.0)
        self.assertEqual(ledger.summarize("training")["verifierPassRate"], 1.0)
        self.assertEqual(ledger.summarize("evaluation-after")["verifierPassRate"], 0.0)

    def test_v2_dataset_leaves_a_holdout_large_enough_to_measure(self) -> None:
        records, sha = rlvr.load_builtin_dataset("arithmetic-rlvr-v2")
        self.assertEqual(len(records), 320)
        self.assertEqual(len(sha), 64)
        self.assertEqual(len({row["prompt"] for row in records}), 320)

        training, holdout = rlvr.split_dataset(records, 64)
        self.assertEqual((len(training), len(holdout)), (256, 64))
        self.assertTrue(
            {record["prompt"] for record in training}.isdisjoint(
                record["prompt"] for record in holdout
            )
        )

    def test_project_dataset_and_verifier_load_only_from_snapshot_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            verifier = root / "verifier.py"
            dataset.write_text(
                '{"sampleId":"stable-1","prompt":"2+2","answer":"4"}\n'
                '{"sampleId":"stable-2","prompt":"3+3","answer":"6"}\n'
            )
            verifier.write_text(
                "def verify(completion, expected):\n"
                "    return completion.strip() == f'answer={expected}'\n"
            )

            records, digest = rlvr.load_project_dataset(str(dataset))
            verify = rlvr.load_project_verifier(str(verifier))
            reward = rlvr.make_exact_integer_reward(
                rlvr.EvidenceLedger(9), verifier=verify
            )

            self.assertEqual([row["sampleId"] for row in records], ["stable-1", "stable-2"])
            self.assertEqual(len(digest), 64)
            self.assertEqual(reward(completions=["answer=4"], answer=["4"]), [1.0])


if __name__ == "__main__":
    unittest.main()
