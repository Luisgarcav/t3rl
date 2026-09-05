# RL Gate 0 fake-stream baseline

This fixture exercises the durable/live split without launching a trainer: 5,000 distinct semantic
metric batches are written to SQLite at a modeled 50 batches/second, while only two batches/second
cross the serialized subscriber boundary. It also folds every published event through the real
client projection and writes three fixed-size artifacts.

Run it from the repository root:

```bash
vp run benchmark:rl
```

The first checked-in raw result is
[`2026-09-04-linux-x64.json`](./2026-09-04-linux-x64.json). Its reference host was Linux x64 on an
AMD Ryzen 9 8945HS (16 logical CPUs), 16 GB RAM, and Node 24.20.0.

| Signal                              |                 Gate 0 baseline |
| ----------------------------------- | ------------------------------: |
| Wall clock                          |                       71.707 ms |
| User / system CPU                   |              36.611 / 30.918 ms |
| RSS start / peak                    | 189,566,976 / 193,499,136 bytes |
| SQLite statements / physical bytes  |               5,000 / 5,000,696 |
| Serialized WebSocket bytes          |                          35,011 |
| Artifact bytes                      |                         331,776 |
| Reducer events / chart point visits |                    201 / 16,860 |

The correctness gates are exact: all 5,000 distinct steps must persist and the transport must emit
exactly 200 metric batches. Timing, RSS, and physical SQLite size are comparison signals, not
cross-machine budgets. For a regression decision, run the command five times on the same otherwise
idle host and compare medians to a baseline captured on that host.

This benchmark intentionally excludes model loading, GPU work, real socket framing, and browser
paint time. Those require separate gated hardware and integrated-client measurements in later
milestones.
