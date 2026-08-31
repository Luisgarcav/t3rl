import type { RlResolvedManifest } from "@t3tools/contracts";

export type RlAlgorithmSpecSource = "resolved-manifest" | "experiment-id" | "generic";

export interface RlAlgorithmStage {
  readonly id: string;
  readonly label: string;
  readonly description: string;
  readonly input: string;
  readonly output: string;
  readonly evidenceMetricKeys: ReadonlyArray<string>;
}

export interface RlAlgorithmEdge {
  readonly from: string;
  readonly to: string;
  readonly kind: "next" | "loop";
}

export interface RlAlgorithmVisualizationSpec {
  readonly version: 1;
  readonly kind: "algorithm-flow";
  readonly algorithm: string;
  readonly family: string;
  readonly summary: string;
  readonly source: RlAlgorithmSpecSource;
  readonly stages: ReadonlyArray<RlAlgorithmStage>;
  readonly edges: ReadonlyArray<RlAlgorithmEdge>;
}

const stage = (
  id: string,
  label: string,
  description: string,
  input: string,
  output: string,
  evidenceMetricKeys: ReadonlyArray<string> = [],
): RlAlgorithmStage => ({ id, label, description, input, output, evidenceMetricKeys });

interface AlgorithmTemplate {
  readonly algorithm: string;
  readonly family: string;
  readonly summary: string;
  readonly stages: ReadonlyArray<RlAlgorithmStage>;
}

const ALGORITHM_TEMPLATES: Readonly<Record<string, AlgorithmTemplate>> = {
  GRPO: {
    algorithm: "GRPO",
    family: "Online RL for language models",
    summary:
      "Evaluate a held-out baseline, sample and verify completion groups, update the policy under a KL constraint, then evaluate again.",
    stages: [
      stage(
        "prompts",
        "Select prompts",
        "A bounded, versioned dataset supplies prompts and reference answers.",
        "Prompt dataset and training seed",
        "Prompt minibatch",
      ),
      stage(
        "sample",
        "Sample completion groups",
        "The current policy generates multiple completions for each prompt.",
        "Prompt minibatch and current language model",
        "Completion groups and token log probabilities",
        ["train/completion_length"],
      ),
      stage(
        "verify",
        "Verify outcomes",
        "A deterministic exact-answer verifier converts each completion into a scalar reward.",
        "Completions and reference answers",
        "Per-completion rewards",
        ["train/reward", "train/verifier_pass_rate"],
      ),
      stage(
        "advantage",
        "Estimate group advantage",
        "Rewards are centered within each completion group to produce relative advantages.",
        "Grouped completion rewards",
        "Relative advantages",
      ),
      stage(
        "update",
        "Update with KL control",
        "GRPO improves rewarded completions while measuring drift from the reference policy.",
        "Completions, log probabilities and relative advantages",
        "Updated language-model parameters",
        ["train/loss", "train/kl", "train/entropy", "train/learning_rate"],
      ),
      stage(
        "evaluate",
        "Evaluate held-out prompts",
        "The same versioned holdout is scored before and after training without entering the optimizer dataset.",
        "Updated policy, held-out prompts and exact-answer verifier",
        "Before/after reward and verifier-pass evidence",
        ["eval/reward", "eval/verifier_pass_rate"],
      ),
      stage(
        "inspect",
        "Inspect evidence",
        "Retained prompt, completion and verifier records make reward behavior auditable.",
        "Scored completion samples",
        "Metrics, evaluation report, summary and completion replay",
        ["system/generated_tokens_upper_bound", "system/tokens_per_second"],
      ),
    ],
  },
  A2C: {
    algorithm: "A2C",
    family: "On-policy actor–critic",
    summary: "Collect a synchronous rollout, estimate advantages, then update actor and critic.",
    stages: [
      stage(
        "act",
        "Act in environment",
        "The actor samples actions while the critic estimates state values.",
        "Observation and current actor–critic parameters",
        "Actions, rewards and value estimates",
        ["train/return", "train/episode_length"],
      ),
      stage(
        "rollout",
        "Collect rollout",
        "Synchronous environments produce a bounded on-policy batch.",
        "Transitions from the current policy",
        "Rollout batch",
      ),
      stage(
        "advantage",
        "Estimate advantage",
        "Bootstrapped returns turn rewards and values into actor and critic targets.",
        "Rewards, terminal flags and value estimates",
        "Returns and advantages",
      ),
      stage(
        "update",
        "Update actor–critic",
        "One optimization pass improves the policy and fits the value function.",
        "Rollout observations, actions, returns and advantages",
        "Updated actor–critic parameters",
        ["train/policy_loss", "train/value_loss", "train/entropy"],
      ),
      stage(
        "evaluate",
        "Evaluate policy",
        "A separately seeded deterministic evaluation measures generalization.",
        "Updated policy",
        "Evaluation return and episode length",
        ["eval/return", "eval/episode_length"],
      ),
    ],
  },
  PPO: {
    algorithm: "PPO",
    family: "On-policy actor–critic",
    summary: "Collect rollouts, estimate advantages and optimize a clipped policy objective.",
    stages: [
      stage(
        "act",
        "Act in environment",
        "The current stochastic policy samples actions and the critic predicts state values.",
        "Observation and current policy",
        "Action, reward, log probability and value estimate",
        ["train/return", "train/episode_length"],
      ),
      stage(
        "rollout",
        "Fill rollout buffer",
        "Only experience from the current policy enters this bounded update batch.",
        "On-policy transitions",
        "Rollout buffer",
      ),
      stage(
        "advantage",
        "Compute GAE",
        "Generalized advantage estimation balances bootstrap bias and variance.",
        "Rewards, values and terminal flags",
        "Returns and normalized advantages",
      ),
      stage(
        "update",
        "Clipped update",
        "Repeated minibatch epochs optimize policy, value and entropy terms while clipping large policy changes.",
        "Actions, old log probabilities, returns and advantages",
        "Updated policy and value parameters",
        ["train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl"],
      ),
      stage(
        "evaluate",
        "Evaluate policy",
        "A separately seeded deterministic evaluation measures the updated policy.",
        "Updated policy",
        "Evaluation return and episode length",
        ["eval/return", "eval/episode_length"],
      ),
    ],
  },
  DQN: {
    algorithm: "DQN",
    family: "Off-policy value learning",
    summary: "Learn action values from replay while a lagged target network stabilizes TD targets.",
    stages: [
      stage(
        "act",
        "Explore and act",
        "An epsilon-greedy policy mixes random exploration with the online Q-network.",
        "Observation, online Q-network and exploration rate",
        "Discrete action and transition",
        ["train/return", "train/episode_length"],
      ),
      stage(
        "replay",
        "Store in replay",
        "Transitions accumulate so updates can reuse and decorrelate experience.",
        "Observation, action, reward, next observation and terminal flag",
        "Replay-buffer entry",
      ),
      stage(
        "target",
        "Build TD target",
        "The target network estimates the bootstrapped value of the next state.",
        "Sampled replay minibatch and target Q-network",
        "Temporal-difference targets",
      ),
      stage(
        "update",
        "Update Q-network",
        "Gradient descent reduces error between online predictions and TD targets.",
        "Current Q-values and TD targets",
        "Updated online Q-network",
        ["train/value_loss"],
      ),
      stage(
        "sync",
        "Sync target network",
        "A periodic or soft copy keeps target estimates slower than the online network.",
        "Online and target parameters",
        "Refreshed target network",
      ),
      stage(
        "evaluate",
        "Evaluate policy",
        "Greedy actions on separate seeds measure learned behavior without exploration noise.",
        "Online Q-network",
        "Evaluation return and episode length",
        ["eval/return", "eval/episode_length"],
      ),
    ],
  },
  DDPG: {
    algorithm: "DDPG",
    family: "Off-policy deterministic actor–critic",
    summary: "Train a deterministic actor against one critic using replay and target networks.",
    stages: [
      stage(
        "act",
        "Act with noise",
        "Exploration noise perturbs the deterministic actor.",
        "Observation and actor",
        "Continuous action and transition",
        ["train/return", "train/episode_length"],
      ),
      stage(
        "replay",
        "Sample replay",
        "Stored transitions provide reusable off-policy minibatches.",
        "Replay buffer",
        "Transition minibatch",
      ),
      stage(
        "target",
        "Build critic target",
        "Target actor and critic bootstrap the next-state value.",
        "Next observations and target networks",
        "TD targets",
      ),
      stage(
        "critic",
        "Update critic",
        "The critic fits the bootstrapped return.",
        "Transitions and TD targets",
        "Updated critic",
        ["train/value_loss"],
      ),
      stage(
        "actor",
        "Update actor",
        "The actor follows the critic gradient toward higher-value actions.",
        "Observations and critic gradients",
        "Updated actor",
        ["train/policy_loss"],
      ),
      stage(
        "targets",
        "Move target networks",
        "Soft updates slowly track actor and critic parameters.",
        "Online and target parameters",
        "Updated target networks",
      ),
      stage(
        "evaluate",
        "Evaluate policy",
        "Noise-free actions on separate seeds measure the policy.",
        "Deterministic actor",
        "Evaluation metrics",
        ["eval/return", "eval/episode_length"],
      ),
    ],
  },
  TD3: {
    algorithm: "TD3",
    family: "Off-policy deterministic actor–critic",
    summary:
      "Use twin critics, smoothed targets and delayed actor updates to reduce value overestimation.",
    stages: [
      stage(
        "act",
        "Act with noise",
        "Exploration noise perturbs the deterministic actor.",
        "Observation and actor",
        "Continuous action and transition",
        ["train/return", "train/episode_length"],
      ),
      stage(
        "replay",
        "Sample replay",
        "Stored transitions provide reusable off-policy minibatches.",
        "Replay buffer",
        "Transition minibatch",
      ),
      stage(
        "target",
        "Smoothed twin target",
        "Noisy target actions and the lower target-critic value form a conservative TD target.",
        "Next observations and target networks",
        "TD targets",
      ),
      stage(
        "critics",
        "Update twin critics",
        "Both critics fit the same bootstrapped target.",
        "Transitions and TD targets",
        "Updated critics",
        ["train/value_loss"],
      ),
      stage(
        "actor",
        "Delayed actor update",
        "The actor updates less frequently through the first critic.",
        "Observations and critic gradient",
        "Updated actor",
        ["train/policy_loss"],
      ),
      stage(
        "targets",
        "Move target networks",
        "Soft updates slowly track actor and critic parameters.",
        "Online and target parameters",
        "Updated target networks",
      ),
      stage(
        "evaluate",
        "Evaluate policy",
        "Noise-free actions on separate seeds measure the policy.",
        "Deterministic actor",
        "Evaluation metrics",
        ["eval/return", "eval/episode_length"],
      ),
    ],
  },
  SAC: {
    algorithm: "SAC",
    family: "Off-policy entropy-regularized actor–critic",
    summary:
      "Optimize twin critics and a stochastic actor while rewarding both return and entropy.",
    stages: [
      stage(
        "act",
        "Sample action",
        "The stochastic actor explores through its learned action distribution.",
        "Observation and actor distribution",
        "Continuous action and transition",
        ["train/return", "train/episode_length", "train/entropy"],
      ),
      stage(
        "replay",
        "Sample replay",
        "Stored transitions provide reusable off-policy minibatches.",
        "Replay buffer",
        "Transition minibatch",
      ),
      stage(
        "target",
        "Entropy-aware target",
        "Target critics combine next-state value with the policy entropy bonus.",
        "Next observations, actor and target critics",
        "Soft TD targets",
      ),
      stage(
        "critics",
        "Update twin critics",
        "Both critics fit the entropy-regularized target.",
        "Transitions and soft TD targets",
        "Updated critics",
        ["train/value_loss"],
      ),
      stage(
        "actor",
        "Update actor",
        "The policy balances critic value against entropy according to temperature.",
        "Observations and critic estimates",
        "Updated stochastic actor",
        ["train/policy_loss", "train/entropy"],
      ),
      stage(
        "targets",
        "Move target critics",
        "Soft updates slowly track the online critics.",
        "Online and target critic parameters",
        "Updated target critics",
      ),
      stage(
        "evaluate",
        "Evaluate policy",
        "Deterministic actions on separate seeds measure learned control.",
        "Updated actor",
        "Evaluation metrics",
        ["eval/return", "eval/episode_length"],
      ),
    ],
  },
};

const GENERIC_TEMPLATE: AlgorithmTemplate = {
  algorithm: "Custom algorithm",
  family: "Project-defined",
  summary:
    "A generic learning loop is shown because this run does not declare a recognized algorithm.",
  stages: [
    stage(
      "experience",
      "Gather experience",
      "Collect evidence from the task or dataset.",
      "Current parameters and task inputs",
      "Bounded training evidence",
      ["train/return", "train/episode_length"],
    ),
    stage(
      "objective",
      "Compute objective",
      "Turn evidence into an explicit learning signal.",
      "Training evidence and current predictions",
      "Losses, returns or advantages",
    ),
    stage(
      "update",
      "Update parameters",
      "Apply the project-defined optimization rule.",
      "Objective and parameters",
      "Updated parameters",
      ["train/policy_loss", "train/value_loss"],
    ),
    stage(
      "evaluate",
      "Evaluate",
      "Measure the updated system on separate evidence.",
      "Updated parameters",
      "Evaluation metrics",
      ["eval/return", "eval/episode_length"],
    ),
  ],
};

function configAlgorithm(manifest: RlResolvedManifest | null): string | null {
  const value = manifest?.effectiveConfig.algorithm;
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
}

function algorithmFromExperimentId(experimentId: string): string | null {
  const normalized = experimentId.toUpperCase();
  return (
    Object.keys(ALGORITHM_TEMPLATES).find((algorithm) =>
      new RegExp(`(?:^|[-_])${algorithm}(?:$|[-_])`).test(normalized),
    ) ?? null
  );
}

function connectStages(stages: ReadonlyArray<RlAlgorithmStage>): ReadonlyArray<RlAlgorithmEdge> {
  if (stages.length < 2) return [];
  const next = stages.slice(0, -1).map(
    (entry, index): RlAlgorithmEdge => ({
      from: entry.id,
      to: stages[index + 1]!.id,
      kind: "next",
    }),
  );
  return next.concat({ from: stages.at(-1)!.id, to: stages[0]!.id, kind: "loop" });
}

export function resolveRlAlgorithmVisualization(
  manifest: RlResolvedManifest | null,
  experimentId: string,
): RlAlgorithmVisualizationSpec {
  const declared = configAlgorithm(manifest);
  const inferred = algorithmFromExperimentId(experimentId);
  const normalized = declared?.toUpperCase() ?? inferred;
  const template = (normalized === null ? undefined : ALGORITHM_TEMPLATES[normalized]) ?? {
    ...GENERIC_TEMPLATE,
    ...(declared === null ? {} : { algorithm: declared }),
  };
  const source: RlAlgorithmSpecSource =
    declared !== null ? "resolved-manifest" : inferred !== null ? "experiment-id" : "generic";
  return {
    version: 1,
    kind: "algorithm-flow",
    algorithm: template.algorithm,
    family: template.family,
    summary: template.summary,
    source,
    stages: template.stages,
    edges: connectStages(template.stages),
  };
}

export function clampRlAlgorithmStageIndex(index: number, stageCount: number): number {
  if (stageCount <= 0) return 0;
  return Math.min(stageCount - 1, Math.max(0, Math.trunc(index)));
}
