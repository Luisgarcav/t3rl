import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fake_worker
from checkpointing import load_fixture_adapter


class FakeWorkerUnitTest(unittest.TestCase):
    def test_summary_is_published_through_a_same_directory_atomic_rename(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir:
            destination = Path(run_dir) / "summary.json"
            real_replace = fake_worker.os.replace
            observed_temporary: list[Path] = []

            def replace(temporary: str, target: str) -> None:
                temporary_path = Path(temporary)
                self.assertEqual(temporary_path.parent, destination.parent)
                self.assertTrue(temporary_path.is_file())
                self.assertFalse(destination.exists())
                observed_temporary.append(temporary_path)
                real_replace(temporary, target)

            with mock.patch.object(fake_worker.os, "replace", side_effect=replace):
                relative = fake_worker.write_summary(run_dir, 7)

            self.assertEqual(relative, "summary.json")
            self.assertEqual(len(observed_temporary), 1)
            self.assertEqual(
                json.loads(destination.read_text()),
                {
                    "finalReturn": fake_worker.metric_values(
                        7, fake_worker.METRIC_STEPS
                    )["train/return"],
                    "runner": "fake",
                    "runnerVersion": "0.1.0",
                    "seed": 7,
                    "steps": fake_worker.METRIC_STEPS,
                },
            )
            self.assertEqual(list(destination.parent.glob(".*.tmp")), [])

    def test_uninterrupted_and_resumed_fixture_reach_identical_final_evidence(
        self,
    ) -> None:
        worker = Path(fake_worker.__file__).resolve()

        def run_worker(run_dir: Path, *extra: str) -> list[dict]:
            environment = {
                **os.environ,
                "T3RL_ENVIRONMENT_FINGERPRINT": "fixture-environment",
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(worker),
                    "--scenario",
                    "checkpoint",
                    "--run-dir",
                    str(run_dir),
                    "--seed",
                    "7",
                    *extra,
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            return [json.loads(line) for line in result.stdout.splitlines()]

        def tree_digest(root: Path) -> str:
            digest = hashlib.sha256()
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
            return digest.hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            child = root / "child"
            parent_messages = run_worker(parent)
            checkpoint = next(
                message
                for message in parent_messages
                if message.get("kind") == "checkpoint"
                and message["evidence"]["globalStep"] == 4
            )
            parent_before = tree_digest(parent)
            child_messages = run_worker(
                child,
                "--resume-checkpoint",
                str(parent / checkpoint["path"]),
            )
            self.assertEqual(tree_digest(parent), parent_before)

            parent_adapter = next(
                message
                for message in parent_messages
                if message.get("kind") == "adapter"
            )
            child_adapter = next(
                message
                for message in child_messages
                if message.get("kind") == "adapter"
            )
            parent_loaded = load_fixture_adapter(parent / parent_adapter["path"])
            child_loaded = load_fixture_adapter(child / child_adapter["path"])
            self.assertEqual(parent_loaded, child_loaded)

            parent_final = next(
                message
                for message in reversed(parent_messages)
                if message.get("type") == "metrics"
            )
            child_final = next(
                message
                for message in reversed(child_messages)
                if message.get("type") == "metrics"
            )
            self.assertEqual(parent_final["step"], 8)
            self.assertEqual(parent_final["values"], child_final["values"])


if __name__ == "__main__":
    unittest.main()
