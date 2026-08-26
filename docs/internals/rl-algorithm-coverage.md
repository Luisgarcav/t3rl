# RL algorithm coverage

## Decision

T3RL models research coverage independently from execution coverage. The research workbench can
plan, compare, and audit a method as soon as its scientific invariants are known. It calls a method
executable only when the connected server reports a matching runner and a version-controlled
experiment definition exists.

RLVR, RLHF, and RLAIF are not algorithm IDs. They describe where the optimization signal comes
from. PPO, GRPO, RLOO, REINFORCE++, and DAPO describe how the policy is optimized. Keeping these axes
separate allows combinations such as GRPO with verifiable rewards or PPO with a learned reward model
without multiplying nearly identical catalog entries.

```text
algorithm + task topology + reward source
                    |
                    v
       family-specific evidence checklist
                    |
                    v
       review-gated Autoresearch proposal
                    |
          explicit user approval
                    |
                    v
 experiment definition + available runner
```

## Research catalog

| Family                         | Representative entries                       | Evidence emphasized                                                    |
| ------------------------------ | -------------------------------------------- | ---------------------------------------------------------------------- |
| Tabular                        | Q-learning, SARSA, Monte Carlo control       | Coverage, Bellman/TD error, convergence                                |
| Deep value-based               | DQN, Double DQN, Rainbow                     | Q bias, replay, target-network drift                                   |
| Policy gradient / actor-critic | REINFORCE, A2C, PPO, TRPO, DDPG, TD3         | Objective, advantage, value error, KL                                  |
| Entropy-regularized            | SAC                                          | Critics, entropy temperature, return                                   |
| Model-based                    | Dyna-Q, MBPO, DreamerV3, MuZero              | Model error, imagined data, planning quality                           |
| Offline                        | BCQ, CQL, IQL, TD3+BC                        | Dataset support, OOD actions, Q extrapolation                          |
| Imitation                      | Behavior cloning, DAgger, GAIL               | Demonstration coverage, covariate shift                                |
| Multi-agent                    | IPPO, MAPPO, QMIX, MADDPG                    | Per-agent/team outcomes, non-stationarity, cross-play                  |
| Bandit / evolutionary          | UCB, Thompson sampling, Evolution Strategies | Regret, uncertainty, population diversity                              |
| LLM policy optimization        | PPO for LLMs, GRPO, RLOO, REINFORCE++, DAPO  | Reward, verifier rate, KL, entropy, length, pass rate                  |
| Preference optimization        | DPO                                          | Win rate, reward margin, length bias; explicitly adjacent to online RL |
| Custom                         | Project-defined method                       | Declared invariants, baseline, failure cases                           |

The catalog is representative rather than an exhaustive enum of paper variants. Stable validated
IDs and a custom entry keep project files forward-compatible while avoiding a contract release for
every new optimizer.

## Execution boundary

The integrated CPU Stable-Baselines3 worker supports PPO and A2C across its validated discrete and
continuous environments, DQN for discrete control, and SAC, TD3, and DDPG for continuous control.
The bundled catalog includes CartPole and Pendulum definitions. Arbitrary scalar metrics, resolved
configuration, model/evaluation/replay artifacts, lifecycle, cancellation, and source evidence use
a runner-neutral protocol.

Adding broad execution should happen at runner adapters, in this order:

1. Add more version-controlled environments and recurrent or goal-conditioned Stable-Baselines3
   variants only when their evaluation semantics are explicit.
2. Add an offline adapter with immutable dataset identity, splits, behavior-policy evidence, and
   offline-evaluation artifacts.
3. Add a multi-agent adapter that preserves agent roles, centralized-training inputs, opponent or
   self-play checkpoints, and per-agent metrics.
4. Add model-based adapters with explicit real versus imagined steps and world-model artifacts.
5. Add an LLM policy-optimization adapter with model/tokenizer/chat-template revisions, prompt and
   dataset snapshots, rollout-engine identity, reference policy, token accounting, verifier
   identity, sandbox evidence, and GPU-hour budgets.

Each adapter must emit the same bounded worker protocol. Framework-specific configuration and
dependency probes stay at that boundary; orchestration, storage, and the UI remain algorithm-neutral.

## Revisit as the system grows

- Replace client-declared native coverage with server-advertised algorithm/task capabilities.
- Add token, sample, verifier-call, and GPU-hour budgets alongside runs and wall-clock time.
- Persist hypothesis-to-outcome records instead of relying only on prepared composer text.
- Add task-specific aggregation units: seeds for control, dataset splits for offline RL, opponents
  for multi-agent RL, and prompt/completion groups for LLM RL.
