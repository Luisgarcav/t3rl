# RL milestone 3 comparison responsiveness profile

Run from the repository root:

```bash
node scripts/rl-study-comparison-benchmark.ts
```

The fixture calls the same cooperative estimator used by public study comparison. A `setImmediate`
heartbeat measures how often computation returns control to the event loop. Version 2 honors the
declared statistical unit, missing-pair policy, and a maximum budget of 20,000,000 random draws.
The budget is checked before resampling; sample preparation and resampling both yield in bounded
slices. Version 1 remains readable as historical evidence and is refused for new calculations.

The [2026-09-05 raw result](./result-2026-09-05.jsonl) records two scenarios:

| Scenario      | Training seed pairs | Paired samples per seed | Requested resamples | Result                              |    Elapsed | Maximum observed heartbeat gap |
| ------------- | ------------------: | ----------------------: | ------------------: | ----------------------------------- | ---------: | -----------------------------: |
| Within budget |                  32 |                     512 |               1,000 | Interval                            | 306.764 ms |                       6.415 ms |
| Over budget   |                  64 |                  10,000 |             100,000 | Explicit computation-budget refusal | 102.764 ms |                      12.596 ms |

These are single-run observations on the development host, not supported cross-machine latency
budgets. A regression comparison must record source revision, runtime/hardware, and repeated
measurements on the same otherwise idle host. The command checks the expected conclusion for each
scenario, a nonzero heartbeat count, and a maximum 50 ms heartbeat gap, including the final interval
before cleanup. Exceeding that host regression limit fails the command; it is not a latency promise
for arbitrary hardware.

The benchmark measures estimator wall time and event-loop responsiveness. It excludes artifact
reading/hashing, sample-index SQLite work, socket serialization/transmission, browser rendering,
model loading, GPU training, and second-machine reproduction. Those need separate evidence.

The historical [2026-09-04 result](./result.json) used the original synchronous estimator. Its
`sqliteWritesPerStudy`, `artifactGrowthBytes`, and `rendererComparisonObjects` fields were modeled
counts, and its serialized comparison size was not an actual WebSocket measurement. Preserve that
record for attribution; use the dated version-2 profile for the corrected estimator rather than
presenting those old fields as measured integrated costs.
