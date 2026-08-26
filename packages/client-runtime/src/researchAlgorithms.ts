import {
  ResearchAlgorithmId,
  type ResearchRewardSource,
  type ResearchTaskType,
} from "@t3tools/contracts";

export const RESEARCH_ALGORITHM_FAMILIES = [
  {
    id: "tabular",
    label: "Tabular & dynamic programming",
    evidence: ["state-action coverage", "Bellman or TD error", "policy/value convergence"],
  },
  {
    id: "value-based",
    label: "Deep value-based",
    evidence: ["Q-value scale and bias", "TD error", "replay coverage", "target-network drift"],
  },
  {
    id: "policy-gradient",
    label: "Policy gradient",
    evidence: ["policy objective", "advantage distribution", "entropy", "gradient stability"],
  },
  {
    id: "actor-critic",
    label: "Actor-critic",
    evidence: ["policy and value losses", "explained variance", "entropy", "policy KL"],
  },
  {
    id: "entropy-regularized",
    label: "Entropy-regularized actor-critic",
    evidence: ["actor and critic losses", "Q estimates", "entropy temperature", "return"],
  },
  {
    id: "model-based",
    label: "Model-based RL",
    evidence: ["model prediction error", "real/imagined data ratio", "planning quality", "return"],
  },
  {
    id: "offline",
    label: "Offline RL",
    evidence: [
      "dataset coverage",
      "out-of-distribution actions",
      "Q extrapolation",
      "offline evaluation",
    ],
  },
  {
    id: "imitation",
    label: "Imitation learning",
    evidence: ["demonstration coverage", "behavior divergence", "covariate shift", "task success"],
  },
  {
    id: "multi-agent",
    label: "Multi-agent RL",
    evidence: [
      "per-agent return",
      "team return",
      "policy non-stationarity",
      "coordination failures",
    ],
  },
  {
    id: "bandit",
    label: "Bandits",
    evidence: ["cumulative regret", "action coverage", "uncertainty calibration", "propensity"],
  },
  {
    id: "evolutionary",
    label: "Evolutionary & population-based",
    evidence: ["population fitness", "diversity", "selection pressure", "sample efficiency"],
  },
  {
    id: "llm-policy-optimization",
    label: "LLM policy optimization",
    evidence: [
      "reward and verifier rates",
      "policy KL and entropy",
      "response length",
      "held-out pass rate",
    ],
  },
  {
    id: "preference-optimization",
    label: "Preference optimization (adjacent)",
    evidence: ["held-out preference win rate", "reward margin", "length bias", "reference drift"],
  },
  {
    id: "custom",
    label: "Custom / project-defined",
    evidence: ["declared objective", "algorithm-specific invariants", "baseline", "failure cases"],
  },
] as const;

export type ResearchAlgorithmFamilyId = (typeof RESEARCH_ALGORITHM_FAMILIES)[number]["id"];
export type ResearchLearningMode =
  | "model-free"
  | "on-policy"
  | "off-policy"
  | "offline"
  | "model-based"
  | "imitation"
  | "multi-agent"
  | "bandit"
  | "evolutionary"
  | "preference";
export type ResearchActionSpace = "discrete" | "continuous" | "hybrid" | "joint" | "text";

export interface ResearchAlgorithmDefinition {
  readonly id: ResearchAlgorithmId;
  readonly name: string;
  readonly family: ResearchAlgorithmFamilyId;
  readonly description: string;
  readonly modes: ReadonlyArray<ResearchLearningMode>;
  readonly actionSpaces: ReadonlyArray<ResearchActionSpace>;
  /** Informational only. A real run still requires a catalog experiment and capability check. */
  readonly suggestedAdapters: ReadonlyArray<string>;
  readonly nativeExecution?: {
    readonly runnerId: string;
    readonly taskTypes: ReadonlyArray<ResearchTaskType>;
  };
}

const algorithm = (
  id: string,
  name: string,
  family: ResearchAlgorithmFamilyId,
  description: string,
  modes: ReadonlyArray<ResearchLearningMode>,
  actionSpaces: ReadonlyArray<ResearchActionSpace>,
  suggestedAdapters: ReadonlyArray<string>,
  nativeExecution?: ResearchAlgorithmDefinition["nativeExecution"],
): ResearchAlgorithmDefinition => ({
  id: ResearchAlgorithmId.make(id),
  name,
  family,
  description,
  modes:
    modes.includes("model-based") ||
    modes.includes("bandit") ||
    modes.includes("evolutionary") ||
    modes.includes("preference")
      ? modes
      : ["model-free", ...modes],
  actionSpaces,
  suggestedAdapters,
  ...(nativeExecution === undefined ? {} : { nativeExecution }),
});

/**
 * Research coverage is intentionally wider than execution coverage. Entries describe how to plan,
 * compare, and audit a method; runner support is advertised separately and never inferred.
 */
export const RESEARCH_ALGORITHM_CATALOG: ReadonlyArray<ResearchAlgorithmDefinition> = [
  algorithm(
    "q-learning",
    "Q-learning",
    "tabular",
    "Off-policy tabular value learning.",
    ["off-policy"],
    ["discrete"],
    ["custom-python"],
  ),
  algorithm(
    "sarsa",
    "SARSA",
    "tabular",
    "On-policy tabular TD control.",
    ["on-policy"],
    ["discrete"],
    ["custom-python"],
  ),
  algorithm(
    "monte-carlo-control",
    "Monte Carlo control",
    "tabular",
    "Episode-return policy evaluation and improvement.",
    ["on-policy"],
    ["discrete"],
    ["custom-python"],
  ),
  algorithm(
    "dqn",
    "DQN",
    "value-based",
    "Deep Q-learning with replay and a target network.",
    ["off-policy"],
    ["discrete"],
    ["stable-baselines3", "cleanrl", "rllib"],
    { runnerId: "stable-baselines3", taskTypes: ["discrete-control"] },
  ),
  algorithm(
    "double-dqn",
    "Double DQN",
    "value-based",
    "DQN variant that reduces maximization bias.",
    ["off-policy"],
    ["discrete"],
    ["cleanrl", "rllib"],
  ),
  algorithm(
    "rainbow-dqn",
    "Rainbow DQN",
    "value-based",
    "Combined distributional and replay improvements for DQN.",
    ["off-policy"],
    ["discrete"],
    ["rllib", "custom-python"],
  ),
  algorithm(
    "reinforce",
    "REINFORCE",
    "policy-gradient",
    "Monte Carlo policy-gradient optimization.",
    ["on-policy"],
    ["discrete", "continuous"],
    ["cleanrl", "torchrl"],
  ),
  algorithm(
    "a2c",
    "A2C",
    "actor-critic",
    "Synchronous advantage actor-critic.",
    ["on-policy"],
    ["discrete", "continuous"],
    ["stable-baselines3", "rllib"],
    {
      runnerId: "stable-baselines3",
      taskTypes: ["discrete-control", "continuous-control"],
    },
  ),
  algorithm(
    "ppo",
    "PPO",
    "actor-critic",
    "Clipped on-policy actor-critic.",
    ["on-policy"],
    ["discrete", "continuous"],
    ["stable-baselines3", "cleanrl", "rllib"],
    {
      runnerId: "stable-baselines3",
      taskTypes: ["discrete-control", "continuous-control"],
    },
  ),
  algorithm(
    "trpo",
    "TRPO",
    "actor-critic",
    "Trust-region constrained policy optimization.",
    ["on-policy"],
    ["discrete", "continuous"],
    ["rllib", "custom-python"],
  ),
  algorithm(
    "ddpg",
    "DDPG",
    "actor-critic",
    "Deterministic off-policy actor-critic.",
    ["off-policy"],
    ["continuous"],
    ["stable-baselines3", "rllib"],
    { runnerId: "stable-baselines3", taskTypes: ["continuous-control"] },
  ),
  algorithm(
    "td3",
    "TD3",
    "actor-critic",
    "Twin-critic deterministic policy gradients.",
    ["off-policy"],
    ["continuous"],
    ["stable-baselines3", "cleanrl", "rllib"],
    { runnerId: "stable-baselines3", taskTypes: ["continuous-control"] },
  ),
  algorithm(
    "sac",
    "SAC",
    "entropy-regularized",
    "Maximum-entropy off-policy actor-critic.",
    ["off-policy"],
    ["continuous", "discrete"],
    ["stable-baselines3", "cleanrl", "rllib"],
    { runnerId: "stable-baselines3", taskTypes: ["continuous-control"] },
  ),
  algorithm(
    "dyna-q",
    "Dyna-Q",
    "model-based",
    "Tabular learning interleaved with model-generated updates.",
    ["off-policy", "model-based"],
    ["discrete"],
    ["custom-python"],
  ),
  algorithm(
    "mbpo",
    "MBPO",
    "model-based",
    "Short model rollouts augment off-policy training.",
    ["off-policy", "model-based"],
    ["continuous"],
    ["custom-python"],
  ),
  algorithm(
    "dreamer-v3",
    "DreamerV3",
    "model-based",
    "Latent world-model learning with imagined trajectories.",
    ["model-based"],
    ["discrete", "continuous"],
    ["dreamerv3", "custom-python"],
  ),
  algorithm(
    "muzero",
    "MuZero",
    "model-based",
    "Learned dynamics combined with tree-search planning.",
    ["model-based"],
    ["discrete"],
    ["mctx", "custom-python"],
  ),
  algorithm(
    "bcq",
    "BCQ",
    "offline",
    "Batch-constrained off-policy value learning.",
    ["offline", "off-policy"],
    ["discrete", "continuous"],
    ["d3rlpy", "custom-python"],
  ),
  algorithm(
    "cql",
    "CQL",
    "offline",
    "Conservative value learning for offline datasets.",
    ["offline", "off-policy"],
    ["discrete", "continuous"],
    ["d3rlpy", "rllib"],
  ),
  algorithm(
    "iql",
    "IQL",
    "offline",
    "Implicit value learning without explicit behavior constraints.",
    ["offline", "off-policy"],
    ["discrete", "continuous"],
    ["d3rlpy", "custom-python"],
  ),
  algorithm(
    "td3-bc",
    "TD3+BC",
    "offline",
    "TD3 regularized toward dataset actions.",
    ["offline", "off-policy"],
    ["continuous"],
    ["d3rlpy", "custom-python"],
  ),
  algorithm(
    "behavior-cloning",
    "Behavior cloning",
    "imitation",
    "Supervised policy learning from demonstrations.",
    ["imitation", "offline"],
    ["discrete", "continuous", "text"],
    ["imitation", "custom-python"],
  ),
  algorithm(
    "dagger",
    "DAgger",
    "imitation",
    "Interactive imitation with expert relabeling.",
    ["imitation", "on-policy"],
    ["discrete", "continuous"],
    ["imitation", "custom-python"],
  ),
  algorithm(
    "gail",
    "GAIL",
    "imitation",
    "Adversarial occupancy-measure imitation.",
    ["imitation", "on-policy"],
    ["discrete", "continuous"],
    ["imitation"],
  ),
  algorithm(
    "ippo",
    "IPPO",
    "multi-agent",
    "Independent PPO policies in a shared environment.",
    ["on-policy", "multi-agent"],
    ["joint"],
    ["pettingzoo", "rllib"],
  ),
  algorithm(
    "mappo",
    "MAPPO",
    "multi-agent",
    "Centralized-training PPO for cooperative agents.",
    ["on-policy", "multi-agent"],
    ["joint"],
    ["rllib", "custom-python"],
  ),
  algorithm(
    "qmix",
    "QMIX",
    "multi-agent",
    "Monotonic value decomposition for cooperative agents.",
    ["off-policy", "multi-agent"],
    ["joint", "discrete"],
    ["pymarl", "rllib"],
  ),
  algorithm(
    "maddpg",
    "MADDPG",
    "multi-agent",
    "Centralized critics with decentralized deterministic actors.",
    ["off-policy", "multi-agent"],
    ["joint", "continuous"],
    ["rllib", "custom-python"],
  ),
  algorithm(
    "ucb",
    "Upper Confidence Bound",
    "bandit",
    "Optimism-based action selection for bandits.",
    ["bandit"],
    ["discrete"],
    ["custom-python"],
  ),
  algorithm(
    "thompson-sampling",
    "Thompson sampling",
    "bandit",
    "Posterior-sampling exploration for bandits.",
    ["bandit"],
    ["discrete"],
    ["custom-python"],
  ),
  algorithm(
    "evolution-strategies",
    "Evolution Strategies",
    "evolutionary",
    "Gradient-free population search over policy parameters.",
    ["evolutionary"],
    ["discrete", "continuous"],
    ["evotorch", "custom-python"],
  ),
  algorithm(
    "ppo-llm",
    "PPO for LLMs",
    "llm-policy-optimization",
    "Token-policy PPO against learned, rule-based, or verifiable rewards.",
    ["on-policy"],
    ["text"],
    ["trl", "openrlhf", "verl"],
  ),
  algorithm(
    "grpo",
    "GRPO",
    "llm-policy-optimization",
    "Group-relative policy optimization commonly used with sampled LLM completions.",
    ["on-policy"],
    ["text"],
    ["trl", "open-r1", "verl"],
  ),
  algorithm(
    "rloo",
    "RLOO",
    "llm-policy-optimization",
    "REINFORCE leave-one-out baseline for LLM policy optimization.",
    ["on-policy"],
    ["text"],
    ["trl", "custom-python"],
  ),
  algorithm(
    "reinforce-plus-plus",
    "REINFORCE++",
    "llm-policy-optimization",
    "Stabilized REINFORCE-style LLM policy optimization.",
    ["on-policy"],
    ["text"],
    ["verl", "custom-python"],
  ),
  algorithm(
    "dapo",
    "DAPO",
    "llm-policy-optimization",
    "Decoupled clipping and sampling refinements for LLM reasoning RL.",
    ["on-policy"],
    ["text"],
    ["verl", "custom-python"],
  ),
  algorithm(
    "dpo",
    "DPO",
    "preference-optimization",
    "Direct preference optimization; adjacent to RL but without online rollouts.",
    ["preference", "offline"],
    ["text"],
    ["trl", "openrlhf"],
  ),
  algorithm(
    "custom",
    "Custom / project-defined",
    "custom",
    "A project-defined algorithm with explicit invariants and evidence.",
    ["on-policy", "off-policy"],
    ["discrete", "continuous", "hybrid", "joint", "text"],
    ["custom-python"],
  ),
];

const CUSTOM_ALGORITHM = RESEARCH_ALGORITHM_CATALOG.at(-1)!;

export function researchAlgorithmById(id: ResearchAlgorithmId): ResearchAlgorithmDefinition {
  return RESEARCH_ALGORITHM_CATALOG.find((entry) => entry.id === id) ?? CUSTOM_ALGORITHM;
}

export function researchAlgorithmFamily(familyId: ResearchAlgorithmFamilyId) {
  return RESEARCH_ALGORITHM_FAMILIES.find((entry) => entry.id === familyId)!;
}

export function researchExecutionStatus(
  definition: ResearchAlgorithmDefinition,
  taskType: ResearchTaskType,
): { readonly native: boolean; readonly detail: string } {
  const execution = definition.nativeExecution;
  if (execution !== undefined && execution.taskTypes.includes(taskType)) {
    return {
      native: true,
      detail: `An integrated ${execution.runnerId} runner path exists for this task class; the connected server and matching catalog definition must still pass capability checks.`,
    };
  }
  const adapters = definition.suggestedAdapters.join(", ");
  return {
    native: false,
    detail: `Research planning and evidence review are supported; training needs a worker adapter${adapters.length === 0 ? "." : ` such as ${adapters}.`}`,
  };
}

export const RESEARCH_TASK_TYPE_OPTIONS: ReadonlyArray<{
  readonly value: ResearchTaskType;
  readonly label: string;
}> = [
  { value: "bandit", label: "Bandit" },
  { value: "tabular", label: "Tabular / small MDP" },
  { value: "discrete-control", label: "Discrete control" },
  { value: "continuous-control", label: "Continuous control" },
  { value: "robotics", label: "Robotics" },
  { value: "games", label: "Games / planning" },
  { value: "multi-agent", label: "Multi-agent" },
  { value: "offline-dataset", label: "Offline dataset" },
  { value: "imitation", label: "Imitation" },
  { value: "model-based", label: "Model-based / world model" },
  { value: "llm-alignment", label: "LLM alignment" },
  { value: "llm-reasoning", label: "LLM reasoning" },
  { value: "custom", label: "Custom" },
];

export const RESEARCH_REWARD_SOURCE_OPTIONS: ReadonlyArray<{
  readonly value: ResearchRewardSource;
  readonly label: string;
}> = [
  { value: "environment", label: "Environment reward" },
  { value: "demonstrations", label: "Demonstrations" },
  { value: "human-feedback", label: "Human feedback (RLHF)" },
  { value: "ai-feedback", label: "AI feedback (RLAIF)" },
  { value: "learned-reward-model", label: "Learned reward model" },
  { value: "verifiable", label: "Verifiable reward (RLVR)" },
  { value: "rule-based", label: "Rule-based reward" },
  { value: "hybrid", label: "Hybrid reward" },
  { value: "custom", label: "Custom" },
];
