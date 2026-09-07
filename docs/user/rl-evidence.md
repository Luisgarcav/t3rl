# Export and verify RL Lab evidence

Open a completed, cancelled, failed, or interrupted run and select **Export evidence**. The
download dialog shows the archive size, file count, omissions, and SHA-256 hashes for the archive
and its index. Download the `.tar` file from that dialog. Downloads work through the connected
environment, including remote connections.

The archive contains the recorded run, bounded metric history, artifact inventory, and verified
evaluation, summary, configuration, dataset, source, and adapter artifacts when available. It
also retains project input snapshots and recorded environment setup files. Checkpoints and logs
require explicit selection through the agent tool. Every omitted artifact is listed in the index;
an artifact whose recorded hash no longer matches its bytes stops the export.

## Check an archive without its original environment

Keep the two hashes displayed in the download dialog. First compare the archive's SHA-256 with
the displayed **Archive SHA-256**:

```sh
sha256sum run-evidence.tar
```

On macOS, use `shasum -a 256 run-evidence.tar`. After the archive hash matches, obtain its verifier
and run it with Python 3 and the displayed **Index SHA-256**:

```sh
tar -xOf run-evidence.tar verify_evidence.py > verify_evidence.py
python3 verify_evidence.py run-evidence.tar --index-sha256 EXPECTED_INDEX_SHA256
```

The verifier uses only Python's standard library. It checks every included file against the index,
rejects missing, changed, or unlisted files, and reports the number of omissions. It does not need
the originating server, database, model framework, or original file paths. A successful check
reports `"verified": true`.

Inspect `index.json` to see omissions and the scope of the evidence:

```sh
tar -xOf run-evidence.tar index.json
```

## What the evidence establishes

| Property                   | Required evidence                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| File integrity             | The archive and included file hashes match the retained expected hashes.                                                                    |
| Environment reconstruction | Retained setup files, required packages and model weights, and a compatible platform; inspect omissions.                                    |
| Trainer-state resume       | An explicitly included complete checkpoint and compatible model, optimizer, framework, and runtime configuration.                           |
| Numerical reproducibility  | A rerun compared with a declared numerical tolerance. Exporting a checkpoint alone does not establish this.                                 |
| Empirical repeatability    | Independent evaluation and sufficient replication for the intended claim. A verified archive can contain a negative or inconclusive result. |

A recorded source revision and dirty-state flag do not contain the source bytes. If an exact source
snapshot was not retained, the index declares that omission. Older runs may also lack retained
environment setup files. The metric file is a bounded reconnect snapshot, rather than a promise of
complete training history.

## Studies and research records

The agent can use `rl_export_evidence` to select up to twelve runs, a study comparison, research
record IDs, and additional checkpoint or log artifacts. The bundle contains the study's evaluation
protocol and estimator settings alongside the computed result. Unselected study members remain
visible as omissions. Exports are limited to 512 MiB of included evidence and 4,096 files.

Use `rl_record_research` to retain a hypothesis, references to observed evidence, a proposed change,
an authorization reference, an outcome, an interpretation, and limitations. The server verifies
the referenced evidence and records its hash and source identity. The authorization reference is
an author's assertion; execution still uses the normal run permissions.

Research records are immutable. Repeating the same request ID with the same content returns the
existing record. To record a result or a revised interpretation, create a new record that names
the earlier record as its parent. `rl_get_research_record` retrieves the retained record. Exporting
a child includes its ancestors, subject to the sixteen-record limit, so the hypothesis and the
later outcome can travel together.

The run's export button creates a single-run bundle. Use the agent tool when the package needs a
study, research history, or complete trainer checkpoint.

An agent export returns the retained archive path on the connected environment, so the agent can
verify or copy that package there. It also returns the resource identity used by clients to request
a signed download link.
