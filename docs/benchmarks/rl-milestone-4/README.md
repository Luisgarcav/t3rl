# RL milestone 4 performance profile

Run `node --experimental-strip-types scripts/rl-project-experiment-benchmark.ts` from the repository
root. The fixture hashes a project definition, a 1 MiB dataset, and a verifier 100 times. Validation
does not write SQLite or create artifacts. Starting a run copies each resolved local input once into
the run input snapshot; clients receive only the bounded validation report.

The reference result in `result.json` was recorded on Linux x64 on 2026-09-04. Project catalogs are
read on demand so committed edits appear without a watcher, polling loop, or continuously repainting
client state.
