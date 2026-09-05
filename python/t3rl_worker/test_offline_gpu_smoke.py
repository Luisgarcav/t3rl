import os
import unittest


@unittest.skipUnless(os.environ.get("T3RL_RUN_GPU_SMOKE") == "1", "set T3RL_RUN_GPU_SMOKE=1 on a configured CUDA host")
class OfflineGpuSmokeTest(unittest.TestCase):
    def test_cuda_gate_is_explicit(self) -> None:
        import torch

        self.assertTrue(torch.cuda.is_available())


if __name__ == "__main__":
    unittest.main()
