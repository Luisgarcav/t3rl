#!/usr/bin/env python3
"""Reward bridge Axolotl resolves by import path.

Axolotl loads reward functions with `importlib` from a dotted path. A worker
executed as a script lives in `__main__`, so a reward function defined there
would be imported a second time into a distinct module object with empty
globals. Keeping the bridge in its own module means the worker and Axolotl
reach the same object.
"""

from __future__ import annotations

from typing import Any

# Installed by the worker before training starts.
REWARD: Any = None


def exact_integer_reward(*args: Any, **kwargs: Any) -> list[float]:
    if REWARD is None:
        raise RuntimeError("reward function used before the run installed it")
    return REWARD(*args, **kwargs)
