import { createRlEnvironmentAtoms } from "@t3tools/client-runtime/state/rl";

import { connectionAtomRuntime } from "../connection/runtime";

export const rlEnvironment = createRlEnvironmentAtoms(connectionAtomRuntime);
