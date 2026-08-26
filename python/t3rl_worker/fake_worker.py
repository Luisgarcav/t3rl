#!/usr/bin/env python3
"""Deterministic T3RL worker that performs no learning.

Exists to exercise the run kernel end to end: spawn, protocol, artifacts,
cancellation, and every failure path, without depending on Gymnasium,
Stable-Baselines3, or a GPU. Standard library only.

Output is byte-identical for a given seed and scenario, so it can serve as a
fixture: `wallClockMs` derives from the step index rather than from the clock.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

PROTOCOL_VERSION = 1
RUNNER_ID = "fake"
RUNNER_VERSION = "0.1.0"
METRIC_STEPS = 10


def emit(message: dict) -> None:
    """Writes one NDJSON line and flushes. A buffered worker looks stalled."""
    sys.stdout.write(json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def metric_values(seed: int, step: int) -> dict:
    """Values from the seed and step alone, so two runs at one seed agree.

    Non-finite results are sent as markers, never as JSON null: `null` means
    "not captured", which is a different claim than "diverged".
    """
    base = (seed * 31 + step * 7) % 100
    values = {
        "train/return": base / 10.0,
        "train/episode_length": float(50 + (base % 25)),
        "train/policy_loss": round(1.0 / (step + 1), 6),
    }
    # One deliberately unrecorded metric, so consumers exercise the null path.
    values["train/approx_kl"] = None if step % 3 == 0 else round(0.01 * step, 6)
    # And one deliberate divergence near the end, to exercise the marker path.
    if step == METRIC_STEPS:
        values["train/value_loss"] = "nan"
    else:
        values["train/value_loss"] = round(2.0 / (step + 1), 6)
    return values


def emit_hello(protocol: int = PROTOCOL_VERSION) -> None:
    emit(
        {
            "type": "hello",
            "protocol": protocol,
            "runner": RUNNER_ID,
            "runnerVersion": RUNNER_VERSION,
        }
    )


def emit_manifest(args: argparse.Namespace) -> None:
    values = dict(args.config_json)
    values.update(
        {
            "scenario": args.scenario,
            "seed": args.seed,
            "steps": METRIC_STEPS,
            "pythonVersion": "%d.%d" % sys.version_info[:2],
        }
    )
    emit(
        {
            "type": "manifest",
            "values": values,
        }
    )


def emit_metrics(seed: int, steps: int) -> None:
    for step in range(1, steps + 1):
        emit(
            {
                "type": "metrics",
                "step": step,
                # Derived, not measured: a clock reading would make two runs at
                # the same seed differ and this worker useless as a fixture.
                "wallClockMs": step * 100,
                "values": metric_values(seed, step),
            }
        )


def write_summary(run_dir: str, seed: int) -> str:
    os.makedirs(run_dir, exist_ok=True)
    relative = "summary.json"
    payload = {
        "runner": RUNNER_ID,
        "runnerVersion": RUNNER_VERSION,
        "seed": seed,
        "steps": METRIC_STEPS,
        "finalReturn": metric_values(seed, METRIC_STEPS)["train/return"],
    }
    with open(os.path.join(run_dir, relative), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    return relative


def sleep_forever() -> None:
    while True:
        time.sleep(3600)


def run(args: argparse.Namespace) -> int:
    scenario = args.scenario

    if scenario == "bad-protocol":
        emit_hello(protocol=99)
        sleep_forever()
        return 0

    emit_hello()
    emit_manifest(args)

    if scenario == "malformed":
        sys.stdout.write("{not json\n")
        sys.stdout.flush()
        sleep_forever()
        return 0

    if scenario == "stall":
        emit_metrics(args.seed, 1)
        sleep_forever()
        return 0

    if scenario == "ignore-cancel":
        # Refuse to die on the polite signal so the server has to escalate.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        emit_metrics(args.seed, 2)
        sleep_forever()
        return 0

    if scenario == "exit":
        emit_metrics(args.seed, 2)
        return 3

    if scenario == "fail":
        emit_metrics(args.seed, 2)
        emit({"type": "error", "code": "RunnerException", "detail": "synthetic failure"})
        return 1

    emit_metrics(args.seed, METRIC_STEPS)
    relative = write_summary(args.run_dir, args.seed)
    emit({"type": "artifact", "kind": "summary", "path": relative})
    emit({"type": "done", "status": "completed"})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic T3RL fake worker")
    parser.add_argument(
        "--scenario",
        default="success",
        choices=[
            "success",
            "fail",
            "malformed",
            "stall",
            "exit",
            "ignore-cancel",
            "bad-protocol",
        ],
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config-json", default="{}", type=json.loads)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
