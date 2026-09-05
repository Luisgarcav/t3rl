#!/usr/bin/env python3
"""Stable-Baselines3 worker for bounded discrete and continuous-control runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import tempfile
import time
import traceback
from typing import Any

PROTOCOL_VERSION = 1
RUNNER_ID = "stable-baselines3"
MAX_EPISODE_SUMMARIES = 2048

COMMON_DEFAULT_CONFIG: dict[str, Any] = {
    "policy": "MlpPolicy",
    "device": "cpu",
    "totalTimesteps": 25_000,
    "learningRate": 0.0003,
    "gamma": 0.99,
    "evaluationEpisodes": 10,
    "evaluationSeedOffset": 100_000,
    "maxEvaluationSteps": 500,
}

ALGORITHM_DEFAULTS: dict[str, dict[str, Any]] = {
    "PPO": {
        "environment": "CartPole-v1",
        "nSteps": 256,
        "batchSize": 64,
        "nEpochs": 10,
        "gaeLambda": 0.95,
        "clipRange": 0.2,
        "entropyCoefficient": 0.0,
        "valueCoefficient": 0.5,
        "maxGradNorm": 0.5,
    },
    "A2C": {
        "environment": "CartPole-v1",
        "nSteps": 5,
        "gaeLambda": 1.0,
        "entropyCoefficient": 0.0,
        "valueCoefficient": 0.5,
        "maxGradNorm": 0.5,
    },
    "DQN": {
        "environment": "CartPole-v1",
        "bufferSize": 100_000,
        "learningStarts": 1_000,
        "batchSize": 64,
        "tau": 1.0,
        "trainFrequency": 4,
        "gradientSteps": 1,
        "targetUpdateInterval": 1_000,
        "explorationFraction": 0.1,
        "explorationInitialEpsilon": 1.0,
        "explorationFinalEpsilon": 0.05,
        "maxGradNorm": 10.0,
    },
    "SAC": {
        "environment": "Pendulum-v1",
        "bufferSize": 100_000,
        "learningStarts": 1_000,
        "batchSize": 256,
        "tau": 0.005,
        "trainFrequency": 1,
        "gradientSteps": 1,
        "entropyCoefficient": 0.2,
    },
    "TD3": {
        "environment": "Pendulum-v1",
        "bufferSize": 100_000,
        "learningStarts": 1_000,
        "batchSize": 256,
        "tau": 0.005,
        "trainFrequency": 1,
        "gradientSteps": 1,
        "policyDelay": 2,
        "targetPolicyNoise": 0.2,
        "targetNoiseClip": 0.5,
        "actionNoiseSigma": 0.1,
    },
    "DDPG": {
        "environment": "Pendulum-v1",
        "bufferSize": 100_000,
        "learningStarts": 1_000,
        "batchSize": 256,
        "tau": 0.005,
        "trainFrequency": 1,
        "gradientSteps": 1,
        "actionNoiseSigma": 0.1,
    },
}

SUPPORTED_ENVIRONMENTS = {
    "CartPole-v1": "discrete",
    "MountainCar-v0": "discrete",
    "Acrobot-v1": "discrete",
    "Pendulum-v1": "continuous",
    "MountainCarContinuous-v0": "continuous",
}

ALGORITHM_ACTION_SPACES = {
    "PPO": {"discrete", "continuous"},
    "A2C": {"discrete", "continuous"},
    "DQN": {"discrete"},
    "SAC": {"continuous"},
    "TD3": {"continuous"},
    "DDPG": {"continuous"},
}


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def finite_metric(value: Any) -> float | str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    return number


def _int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _float(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
    return result


def resolve_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("config must be a JSON object")
    algorithm = raw.get("algorithm", "PPO")
    if not isinstance(algorithm, str) or algorithm not in ALGORITHM_DEFAULTS:
        raise ValueError(
            f"algorithm must be one of {', '.join(sorted(ALGORITHM_DEFAULTS))}"
        )
    defaults = {
        **COMMON_DEFAULT_CONFIG,
        "algorithm": algorithm,
        **ALGORITHM_DEFAULTS[algorithm],
    }
    allowed_keys = set(defaults)
    unknown = sorted(set(raw) - allowed_keys)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    config = {**defaults, **raw}
    for key, expected in {"policy": "MlpPolicy", "device": "cpu"}.items():
        if config[key] != expected:
            raise ValueError(f"{key} must be {expected}")

    environment = config["environment"]
    if not isinstance(environment, str):
        raise TypeError("environment must be a string")
    action_space = SUPPORTED_ENVIRONMENTS.get(environment)
    if action_space is None:
        raise ValueError(
            f"environment must be one of {', '.join(sorted(SUPPORTED_ENVIRONMENTS))}"
        )
    if action_space not in ALGORITHM_ACTION_SPACES[algorithm]:
        raise ValueError(
            f"{algorithm} does not support the {action_space} action space"
        )

    config["totalTimesteps"] = _int(
        "totalTimesteps", config["totalTimesteps"], 1, 10_000_000
    )
    config["evaluationEpisodes"] = _int(
        "evaluationEpisodes", config["evaluationEpisodes"], 1, 100
    )
    config["evaluationSeedOffset"] = _int(
        "evaluationSeedOffset", config["evaluationSeedOffset"], 1, 2_000_000_000
    )
    config["maxEvaluationSteps"] = _int(
        "maxEvaluationSteps", config["maxEvaluationSteps"], 1, 10_000
    )
    config["learningRate"] = _float("learningRate", config["learningRate"], 0.0, 1.0)
    config["gamma"] = _float("gamma", config["gamma"], 0.0, 1.0)
    for name in ["nSteps", "batchSize", "nEpochs", "bufferSize", "learningStarts"]:
        if name in config:
            config[name] = _int(name, config[name], 1, 10_000_000)
    for name in [
        "trainFrequency",
        "gradientSteps",
        "targetUpdateInterval",
        "policyDelay",
    ]:
        if name in config:
            config[name] = _int(name, config[name], 1, 1_000_000)
    for name in [
        "gaeLambda",
        "clipRange",
        "entropyCoefficient",
        "valueCoefficient",
        "tau",
        "explorationFraction",
        "explorationInitialEpsilon",
        "explorationFinalEpsilon",
        "targetPolicyNoise",
        "targetNoiseClip",
        "actionNoiseSigma",
    ]:
        if name in config:
            config[name] = _float(name, config[name], 0.0, 100.0)
    if "maxGradNorm" in config:
        config["maxGradNorm"] = _float(
            "maxGradNorm", config["maxGradNorm"], 0.0, 10_000.0
        )
    if algorithm == "PPO" and config["batchSize"] > config["nSteps"]:
        raise ValueError(
            "batchSize cannot exceed nSteps for the single-environment PPO runner"
        )
    return config


def write_json(run_dir: str, relative_path: str, value: Any) -> None:
    target = os.path.join(run_dir, relative_path)
    os.makedirs(os.path.dirname(target) or run_dir, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=os.path.dirname(target) or run_dir,
        prefix=f".{os.path.basename(target)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_dependencies() -> dict[str, Any]:
    import gymnasium
    import numpy
    import stable_baselines3
    import torch
    from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv

    return {
        "gymnasium": gymnasium,
        "numpy": numpy,
        "stable_baselines3": stable_baselines3,
        "torch": torch,
        "PPO": PPO,
        "A2C": A2C,
        "DQN": DQN,
        "SAC": SAC,
        "TD3": TD3,
        "DDPG": DDPG,
        "BaseCallback": BaseCallback,
        "Monitor": Monitor,
        "DummyVecEnv": DummyVecEnv,
        "NormalActionNoise": NormalActionNoise,
    }


def build_model(
    deps: dict[str, Any], config: dict[str, Any], training_env: Any, seed: int
) -> Any:
    common = {
        "policy": config["policy"],
        "env": training_env,
        "seed": seed,
        "device": config["device"],
        "learning_rate": config["learningRate"],
        "gamma": config["gamma"],
        "verbose": 0,
    }
    algorithm = config["algorithm"]
    if algorithm == "PPO":
        return deps[algorithm](
            **common,
            n_steps=config["nSteps"],
            batch_size=config["batchSize"],
            n_epochs=config["nEpochs"],
            gae_lambda=config["gaeLambda"],
            clip_range=config["clipRange"],
            ent_coef=config["entropyCoefficient"],
            vf_coef=config["valueCoefficient"],
            max_grad_norm=config["maxGradNorm"],
        )
    if algorithm == "A2C":
        return deps[algorithm](
            **common,
            n_steps=config["nSteps"],
            gae_lambda=config["gaeLambda"],
            ent_coef=config["entropyCoefficient"],
            vf_coef=config["valueCoefficient"],
            max_grad_norm=config["maxGradNorm"],
        )
    replay = {
        **common,
        "buffer_size": config["bufferSize"],
        "learning_starts": config["learningStarts"],
        "batch_size": config["batchSize"],
        "tau": config["tau"],
        "train_freq": config["trainFrequency"],
        "gradient_steps": config["gradientSteps"],
    }
    if algorithm == "DQN":
        return deps[algorithm](
            **replay,
            target_update_interval=config["targetUpdateInterval"],
            exploration_fraction=config["explorationFraction"],
            exploration_initial_eps=config["explorationInitialEpsilon"],
            exploration_final_eps=config["explorationFinalEpsilon"],
            max_grad_norm=config["maxGradNorm"],
        )
    if algorithm == "SAC":
        return deps[algorithm](
            **replay,
            ent_coef=config["entropyCoefficient"],
        )
    if algorithm == "TD3":
        action_count = training_env.action_space.shape[-1]
        action_noise = deps["NormalActionNoise"](
            mean=deps["numpy"].zeros(action_count),
            sigma=config["actionNoiseSigma"] * deps["numpy"].ones(action_count),
        )
        return deps[algorithm](
            **replay,
            action_noise=action_noise,
            policy_delay=config["policyDelay"],
            target_policy_noise=config["targetPolicyNoise"],
            target_noise_clip=config["targetNoiseClip"],
        )
    if algorithm == "DDPG":
        action_count = training_env.action_space.shape[-1]
        action_noise = deps["NormalActionNoise"](
            mean=deps["numpy"].zeros(action_count),
            sigma=config["actionNoiseSigma"] * deps["numpy"].ones(action_count),
        )
        return deps[algorithm](**replay, action_noise=action_noise)
    raise ValueError(f"unsupported algorithm: {algorithm}")


def json_action(numpy: Any, action: Any) -> Any:
    value = numpy.asarray(action)
    return value.item() if value.ndim == 0 or value.size == 1 else value.tolist()


def dependency_evidence(deps: dict[str, Any]) -> dict[str, str]:
    return {
        "executable": os.path.realpath(sys.executable),
        "python": platform.python_version(),
        "pythonRuntime": sys.version,
        "platform": platform.platform(),
        "stableBaselines3": deps["stable_baselines3"].__version__,
        "gymnasium": deps["gymnasium"].__version__,
        "numpy": deps["numpy"].__version__,
        "torch": deps["torch"].__version__,
    }


def probe() -> int:
    deps = load_dependencies()
    gymnasium = deps["gymnasium"]
    env = gymnasium.make("CartPole-v1")
    try:
        observation, _ = env.reset(seed=0)
        env.action_space.seed(0)
        env.step(env.action_space.sample())
        assert observation is not None
    finally:
        env.close()
    evidence = dependency_evidence(deps)
    evidence["fingerprint"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


def run(args: argparse.Namespace) -> int:
    deps = load_dependencies()
    gymnasium = deps["gymnasium"]
    numpy = deps["numpy"]
    torch = deps["torch"]
    runner_version = deps["stable_baselines3"].__version__

    emit(
        {
            "type": "hello",
            "protocol": PROTOCOL_VERSION,
            "runner": RUNNER_ID,
            "runnerVersion": runner_version,
        }
    )
    args.hello_sent = True
    config = resolve_config(args.config_json)

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    evidence = dependency_evidence(deps)
    emit(
        {
            "type": "manifest",
            "values": {
                **config,
                "seed": args.seed,
                "evaluationSeed": args.seed + config["evaluationSeedOffset"],
                "dependencies": evidence,
                "determinismLimitations": [
                    "Stable-Baselines3 and PyTorch determinism is scoped to the recorded fingerprint and CPU backend."
                ],
            },
        }
    )

    os.makedirs(args.run_dir, exist_ok=True)
    episodes: list[dict[str, float | int]] = []
    started = time.monotonic()

    Monitor = deps["Monitor"]
    DummyVecEnv = deps["DummyVecEnv"]
    BaseCallback = deps["BaseCallback"]

    def make_training_env() -> Any:
        env = gymnasium.make(config["environment"])
        env.action_space.seed(args.seed)
        env.reset(seed=args.seed)
        return Monitor(env)

    class MetricsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.last_emit = started - 1.0

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("episode") if isinstance(info, dict) else None
                if isinstance(episode, dict):
                    episodes.append(
                        {
                            "return": float(episode.get("r", 0.0)),
                            "length": int(episode.get("l", 0)),
                            "step": int(self.num_timesteps),
                        }
                    )
                    del episodes[:-MAX_EPISODE_SUMMARIES]
            self.emit_metrics(False)
            return True

        def emit_metrics(self, force: bool) -> None:
            current = time.monotonic()
            if not force and current - self.last_emit < 0.5:
                return
            values = self.model.logger.name_to_value
            entropy_loss = finite_metric(values.get("train/entropy_loss"))
            entropy = -entropy_loss if isinstance(entropy_loss, float) else entropy_loss
            emit(
                {
                    "type": "metrics",
                    "step": int(self.num_timesteps),
                    "wallClockMs": int((current - started) * 1000),
                    "values": {
                        "train/return": finite_metric(
                            values.get("rollout/ep_rew_mean")
                        ),
                        "train/episode_length": finite_metric(
                            values.get("rollout/ep_len_mean")
                        ),
                        "train/loss": finite_metric(values.get("train/loss")),
                        "train/policy_loss": finite_metric(
                            values.get("train/policy_gradient_loss")
                        ),
                        "train/actor_loss": finite_metric(
                            values.get("train/actor_loss")
                        ),
                        "train/value_loss": finite_metric(
                            values.get("train/value_loss")
                        ),
                        "train/critic_loss": finite_metric(
                            values.get("train/critic_loss")
                        ),
                        "train/qf1_loss": finite_metric(values.get("train/qf1_loss")),
                        "train/qf2_loss": finite_metric(values.get("train/qf2_loss")),
                        "train/entropy": entropy,
                        "train/entropy_coefficient": finite_metric(
                            values.get("train/ent_coef")
                        ),
                        "train/approx_kl": finite_metric(values.get("train/approx_kl")),
                        "train/exploration_rate": finite_metric(
                            values.get("rollout/exploration_rate")
                        ),
                    },
                }
            )
            self.last_emit = current

    training_env = DummyVecEnv([make_training_env])
    evaluation_env = None
    try:
        model = build_model(deps, config, training_env, args.seed)
        callback = MetricsCallback()
        model.learn(
            total_timesteps=config["totalTimesteps"],
            callback=callback,
            progress_bar=False,
        )
        callback.emit_metrics(True)

        model_base = os.path.join(args.run_dir, "model")
        temporary_model_base = os.path.join(
            args.run_dir, f".model.{os.getpid()}.{time.monotonic_ns()}"
        )
        try:
            model.save(temporary_model_base)
            os.replace(f"{temporary_model_base}.zip", f"{model_base}.zip")
        finally:
            try:
                os.unlink(f"{temporary_model_base}.zip")
            except FileNotFoundError:
                pass

        evaluation_returns: list[float] = []
        evaluation_lengths: list[int] = []
        replay: list[dict[str, Any]] = []
        evaluation_env = gymnasium.make(config["environment"])
        evaluation_env.action_space.seed(args.seed + config["evaluationSeedOffset"])
        for episode_index in range(config["evaluationEpisodes"]):
            evaluation_seed = args.seed + config["evaluationSeedOffset"] + episode_index
            observation, _ = evaluation_env.reset(seed=evaluation_seed)
            episode_return = 0.0
            episode_replay: list[dict[str, Any]] = []
            for step in range(config["maxEvaluationSteps"]):
                action, _ = model.predict(observation, deterministic=True)
                next_observation, reward, terminated, truncated, _ = (
                    evaluation_env.step(action)
                )
                if episode_index == 0:
                    episode_replay.append(
                        {
                            "step": step,
                            "observation": numpy.asarray(observation).tolist(),
                            "action": json_action(numpy, action),
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                episode_return += float(reward)
                observation = next_observation
                if terminated or truncated:
                    break
            evaluation_returns.append(episode_return)
            evaluation_lengths.append(step + 1)
            if episode_index == 0:
                replay = episode_replay

        evaluation = {
            "episodes": config["evaluationEpisodes"],
            "deterministic": True,
            "seedBase": args.seed + config["evaluationSeedOffset"],
            "returns": evaluation_returns,
            "lengths": evaluation_lengths,
            "returnMean": statistics.fmean(evaluation_returns),
            "returnStd": statistics.pstdev(evaluation_returns),
            "lengthMean": statistics.fmean(evaluation_lengths),
        }
        summary = {
            "runner": RUNNER_ID,
            "runnerVersion": runner_version,
            "algorithm": config["algorithm"],
            "seed": args.seed,
            "trainingEpisodes": episodes,
            "trainingEpisodeCountRetained": len(episodes),
            "evaluation": evaluation,
            "elapsedMs": int((time.monotonic() - started) * 1000),
        }
        write_json(args.run_dir, "summary.json", summary)
        write_json(args.run_dir, "evaluation.json", evaluation)
        write_json(
            args.run_dir,
            "replay.json",
            {
                "environment": config["environment"],
                "evaluationSeed": args.seed + config["evaluationSeedOffset"],
                "trajectory": replay,
            },
        )
        emit(
            {
                "type": "metrics",
                "step": int(model.num_timesteps),
                "wallClockMs": int((time.monotonic() - started) * 1000),
                "values": {
                    "eval/return": finite_metric(evaluation["returnMean"]),
                    "eval/episode_length": finite_metric(evaluation["lengthMean"]),
                },
            }
        )
        for kind, relative_path in [
            ("summary", "summary.json"),
            ("model", "model.zip"),
            ("evaluation", "evaluation.json"),
            ("replay", "replay.json"),
        ]:
            emit({"type": "artifact", "kind": kind, "path": relative_path})
        emit({"type": "done", "status": "completed"})
        return 0
    finally:
        training_env.close()
        if evaluation_env is not None:
            evaluation_env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="T3RL Stable-Baselines3 worker")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config-json", type=json.loads, default={})
    args = parser.parse_args()
    args.hello_sent = False
    if args.probe:
        return probe()
    if not args.run_dir:
        parser.error("--run-dir is required unless --probe is used")
    try:
        return run(args)
    except Exception as error:  # noqa: BLE001 - process boundary reports arbitrary worker failures.
        traceback.print_exc(file=sys.stderr)
        # A dependency failure happens before hello and is represented by the
        # process exit. All post-hello scientific failures use the typed error.
        if args.hello_sent:
            emit(
                {
                    "type": "error",
                    "code": "RunnerException",
                    "detail": f"{type(error).__name__}: {str(error)[:1500]}",
                }
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
