# RL Milestone 2 checkpoint and lineage profile

This deterministic CPU fixture exercises the Milestone 2 control-plane costs without loading a
model. It atomically publishes and hashes 24 seven-file checkpoint directories with the server's
canonical sorted content-manifest algorithm, persists their evidence and a 16-edge lineage chain in
SQLite, folds ready/trash events through the real client projection, retains two intermediate
checkpoints, and copies one verified source into a child input directory.

Run it from the repository root:

```bash
vp run benchmark:rl-checkpoints
```

The first checked-in raw result is
[`2026-09-04-linux-x64.json`](./2026-09-04-linux-x64.json). Its reference host was Linux x64 on an
AMD Ryzen 9 8945HS (16 logical CPUs), 16 GB RAM, and Node 24.20.0.

| Signal                                      |             Milestone 2 profile |
| ------------------------------------------- | ------------------------------: |
| Wall clock                                  |                       44.125 ms |
| User / system CPU                           |              85.374 / 16.007 ms |
| RSS start / peak                            | 187,650,048 / 190,402,560 bytes |
| SQLite statements / physical bytes          |                    62 / 592,808 |
| Serialized WebSocket bytes                  |                          73,206 |
| Produced / retained parent checkpoint bytes |                986,348 / 82,194 |
| Child input bytes                           |                          41,090 |
| Reducer events / artifact row visits        |                        47 / 597 |
| Lineage edge visits                         |                              16 |

The correctness gates are exact: 24 artifact rows remain as audit evidence, only two intermediate
checkpoints remain ready, 22 are trashed, all 16 lineage rows persist, the child copy matches the
source hash, and the parent hash does not change. Timing, memory, and physical SQLite size are
comparison signals rather than cross-machine budgets; compare medians from five runs on the same
otherwise idle host before calling a regression.

The fixture models WebSocket serialization and SQLite writes but does not open a socket or browser.
It also excludes model loading, GPU memory, framework-native checkpoint serialization, and browser
paint time; those remain hardware and integrated-client release checks.
