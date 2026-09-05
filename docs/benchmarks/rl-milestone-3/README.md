# RL milestone 3 performance profile

Run `vp exec tsx scripts/rl-study-comparison-benchmark.ts` from the repository root. The fixture
uses 32 paired run seeds, 128 paired samples per seed, and 10,000 deterministic hierarchical
bootstrap resamples. It emits CPU time, RSS delta, the bounded comparison response size, artifact
growth, and renderer object count. Study lifecycle writes are bounded to one study row plus two
member transitions per scheduled run; metric evidence remains in the existing bounded run tables.

Reference run on 2026-09-04 (Linux x64, local Node runtime): see `result.json`. This milestone adds
no periodic repaint, sends one aggregate comparison object instead of sample evidence over the
WebSocket, and creates no derived artifact.
