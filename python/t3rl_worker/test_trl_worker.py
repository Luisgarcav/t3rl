import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import trl_worker


class TrlWorkerUnitTest(unittest.TestCase):
    def test_conflicting_continuation_options_emit_protocol_error(self) -> None:
        events = []
        argv = [
            "trl_worker.py",
            "--run-dir",
            "/tmp/run",
            "--resume-checkpoint",
            "/tmp/checkpoint",
            "--warm-start-adapter",
            "/tmp/adapter",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(trl_worker, "emit", events.append),
            mock.patch.object(trl_worker.metadata, "version", return_value="test"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = trl_worker.main()

        self.assertEqual(result, 1)
        self.assertEqual([event["type"] for event in events], ["hello", "error"])
        self.assertIn("mutually exclusive", events[1]["detail"])

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
        self.assertEqual(config["loraRank"], 16)
        self.assertEqual(config["checkpointCadenceSteps"], 4)
        self.assertIs(config["keepFinal"], True)

    def test_resolve_config_rejects_unknown_or_unbounded_values(self) -> None:
        invalid = [
            {"telepathy": True},
            {"algorithm": "PPO"},
            {"rewardSource": "human-preference"},
            {"modelId": "some/other-model"},
            {"useVllm": True},
            {"worldSize": 2},
            {"perDeviceTrainBatchSize": 3, "numGenerations": 2},
            {"evaluationBatchSize": 3, "evaluationNumGenerations": 2},
            {"maxGeneratedTokens": 127},
            {"maxWallClockSeconds": 1800, "maxGpuHours": 0.49},
        ]
        for config in invalid:
            with (
                self.subTest(config=config),
                self.assertRaises((TypeError, ValueError)),
            ):
                trl_worker.resolve_config(config)

    def test_immutable_model_revision_does_not_require_registry_resolution(
        self,
    ) -> None:
        revision = "7ae557604adf67be50417f59c2c2f167def9a775"
        self.assertEqual(
            trl_worker.resolve_model_revision({}, trl_worker.SUPPORTED_MODEL, revision),
            revision,
        )

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
                self.assertEqual(definition["protocolVersion"], 2)

    def test_resolve_config_accepts_the_higher_resolution_dataset(self) -> None:
        config = trl_worker.resolve_config(
            {
                "datasetId": "arithmetic-rlvr-v2",
                "evaluationRows": 64,
                "evaluationNumGenerations": 4,
                "evaluationBatchSize": 8,
                "maxGeneratedTokens": 65536,
            }
        )
        self.assertEqual(config["datasetId"], "arithmetic-rlvr-v2")
        self.assertEqual(config["evaluationRows"], 64)
        with self.assertRaises(ValueError):
            trl_worker.resolve_config({"datasetId": "arithmetic-rlvr-v9"})


if __name__ == "__main__":
    unittest.main()
