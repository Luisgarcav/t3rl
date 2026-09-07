import os
import tempfile
import unittest
from pathlib import Path

from offline_validation import validate


@unittest.skipUnless(os.environ.get("T3RL_RUN_GPU_SMOKE") == "1", "set T3RL_RUN_GPU_SMOKE=1 on a configured CUDA host")
class OfflineGpuSmokeTest(unittest.TestCase):
    def test_real_sft_dpo_checkpoint_resume_and_held_out_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="t3rl-offline-gpu-") as directory:
            report = validate(Path(directory), "cuda", [7], data_seed_offset=12)
            self.assertEqual(report["status"], "passed")
            self.assertEqual({result["method"] for result in report["results"]}, {"sft", "dpo"})


@unittest.skipUnless(os.environ.get("T3RL_RUN_FRAMEWORK_SMOKE") == "1", "set T3RL_RUN_FRAMEWORK_SMOKE=1 with the locked TRL environment")
class OfflineCpuSmokeTest(unittest.TestCase):
    def test_real_sft_dpo_checkpoint_resume_and_held_out_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="t3rl-offline-cpu-") as directory:
            report = validate(Path(directory), "cpu", [7], data_seed_offset=12)
            self.assertEqual(report["status"], "passed")


@unittest.skipUnless(os.environ.get("T3RL_RUN_AXOLOTL_GPU_SMOKE") == "1", "set T3RL_RUN_AXOLOTL_GPU_SMOKE=1 with the locked Axolotl environment")
class AxolotlGpuSmokeTest(unittest.TestCase):
    def test_real_sft_dpo_checkpoint_resume_and_held_out_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="t3rl-axolotl-gpu-") as directory:
            report = validate(Path(directory), "cuda", [7], backend="axolotl", data_seed_offset=12)
            self.assertEqual(report["status"], "passed")


if __name__ == "__main__":
    unittest.main()
