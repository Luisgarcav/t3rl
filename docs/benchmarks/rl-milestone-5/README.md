# RL milestone 5 performance profile

Run `node --experimental-strip-types scripts/rl-offline-methods-benchmark.ts`. The fixture parses and
hashes 10,000 typed DPO JSONL records twenty times. SFT/DPO reuse the existing bounded metric,
artifact, checkpoint, and WebSocket paths, so the method adapter adds no SQLite table, continuous
renderer work, or duplicate artifact representation.

The reference result in `result.json` was recorded on Linux x64 on 2026-09-05. GPU smoke tests are
separately gated with `T3RL_RUN_GPU_SMOKE=1` so normal CI proves deterministic CPU contracts without
claiming CUDA availability.
