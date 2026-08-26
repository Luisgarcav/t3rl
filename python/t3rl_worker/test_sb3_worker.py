import json
import math
import unittest
from pathlib import Path

import sb3_worker


class Sb3WorkerUnitTest(unittest.TestCase):
    def test_resolve_config_applies_defaults_and_explicit_override(self) -> None:
        config = sb3_worker.resolve_config(
            {"totalTimesteps": 512, "evaluationEpisodes": 2}
        )
        self.assertEqual(config["environment"], "CartPole-v1")
        self.assertEqual(config["totalTimesteps"], 512)
        self.assertEqual(config["evaluationEpisodes"], 2)

    def test_resolve_config_rejects_unknown_or_incompatible_values(self) -> None:
        with self.assertRaises(ValueError):
            sb3_worker.resolve_config({"telepathy": True})
        with self.assertRaises(ValueError):
            sb3_worker.resolve_config({"environment": "LunarLander-v3"})
        with self.assertRaises(ValueError):
            sb3_worker.resolve_config({"nSteps": 8, "batchSize": 64})
        with self.assertRaises(ValueError):
            sb3_worker.resolve_config(
                {"algorithm": "DQN", "environment": "Pendulum-v1"}
            )
        with self.assertRaises(ValueError):
            sb3_worker.resolve_config(
                {"algorithm": "SAC", "environment": "CartPole-v1"}
            )
        with self.assertRaises(ValueError):
            sb3_worker.resolve_config({"algorithm": "DQN", "nSteps": 8})
        with self.assertRaises(ValueError):
            sb3_worker.resolve_config({"algorithm": ["PPO"]})
        with self.assertRaises(TypeError):
            sb3_worker.resolve_config({"environment": ["CartPole-v1"]})

    def test_resolve_config_covers_integrated_algorithm_families(self) -> None:
        expected = {
            "PPO": "CartPole-v1",
            "A2C": "CartPole-v1",
            "DQN": "CartPole-v1",
            "SAC": "Pendulum-v1",
            "TD3": "Pendulum-v1",
            "DDPG": "Pendulum-v1",
        }
        for algorithm, environment in expected.items():
            with self.subTest(algorithm=algorithm):
                config = sb3_worker.resolve_config({"algorithm": algorithm})
                self.assertEqual(config["environment"], environment)

    def test_catalog_sb3_experiments_resolve_strictly(self) -> None:
        catalog = Path(__file__).parent / "experiments"
        for path in catalog.glob("*.json"):
            definition = json.loads(path.read_text(encoding="utf-8"))
            if definition["runnerId"] != "stable-baselines3":
                continue
            with self.subTest(experiment=definition["experimentId"]):
                resolved = sb3_worker.resolve_config(definition["config"])
                self.assertEqual(
                    resolved["algorithm"], definition["config"]["algorithm"]
                )

    def test_non_finite_metrics_use_explicit_markers(self) -> None:
        self.assertEqual(sb3_worker.finite_metric(math.nan), "nan")
        self.assertEqual(sb3_worker.finite_metric(math.inf), "+inf")
        self.assertEqual(sb3_worker.finite_metric(-math.inf), "-inf")
        self.assertIsNone(sb3_worker.finite_metric(None))
        self.assertEqual(sb3_worker.finite_metric(1.25), 1.25)


if __name__ == "__main__":
    unittest.main()
