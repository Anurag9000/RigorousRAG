# RigorousRAG operator procedures

This document covers privileged local maintenance operations that are intentionally not
exposed through the HTTP API. Run them only against a backed-up service state with the
application stopped or with an operational maintenance window appropriate to the action.

## Inspect corrupt durable ingestion rows

Normal startup recovery skips malformed rows rather than replaying unsafe state. To list
sanitized records:

```bash
python -m tools.operator_repair list
```

Use an explicit database when `JOB_DB_PATH` is not set:

```bash
python -m tools.operator_repair --job-db /srv/rigorousrag/jobs.sqlite3 list --limit 1000
```

The output contains:

- SQLite row ID;
- an exact SHA-256 fingerprint of the complete selected row;
- bounded corruption-reason identifiers;
- privacy-masked bounded public filename/status fields;
- whether a private source path is recorded;
- a valid non-negative update timestamp when available.

It never prints the private source path or raw private row values.

## Retire one unchanged corrupt row

Retirement is deliberately two-step. First inspect the row and copy its `rowid` and
`fingerprint`. Construct the required token:

```text
RETIRE-<rowid>-<first 12 fingerprint characters>
```

Then run:

```bash
python -m tools.operator_repair retire \
  --rowid 42 \
  --fingerprint 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --confirm RETIRE-42-0123456789ab \
  --reason "Malformed legacy recovery row reviewed under incident INC-123"
```

The command opens an immediate SQLite transaction, re-reads the complete row, and refuses
the action if the fingerprint changed or the row is currently valid. A successful action:

- deletes only that corrupt row from `jobs`;
- does not delete its source file;
- does not delete vector rows;
- does not delete document-registry rows;
- writes an append-only record to `operator_repairs`.

After retirement, reconcile any preserved source deliberately. Do not delete it merely
because the queue row was corrupt; it may still be referenced by a retained document or a
separate recovery record.

## Generate platform-specific hashed runtime locks

Application and development requirement files intentionally use bounded version ranges.
A release lock must be resolved on each target operating system and Python minor version.
Install the isolated compatible toolchain:

```bash
python -m pip install --upgrade "pip>=25,<26"
python -m pip install -r requirements-lock.txt
```

Generate the lock for the current platform/interpreter:

```bash
python scripts/generate_release_lock.py --upgrade
```

The default output is:

```text
locks/runtime-<linux|windows|macos>-py<major><minor>.txt
```

Validate it:

```bash
python scripts/verify_release_lock.py locks/runtime-linux-py312.txt
python -m pip install --require-hashes --no-deps --dry-run \
  -r locks/runtime-linux-py312.txt
```

The generator refuses unsafe input/output paths and creates exact pins with SHA-256 hashes
without embedding package-index URLs or trusted-host configuration. Do not hand-author or
copy a lock across operating systems or Python minors.

The `Generate release locks` GitHub Actions workflow performs the same process on Linux,
Windows, and macOS for Python 3.10, 3.11, and 3.12, then publishes each generated lock as a
short-lived workflow artifact. Release packaging should retrieve the successful artifact
matching the deployment platform and interpreter.

## Frontend launch location

The bundled browser interface is resolved relative to the installed RigorousRAG module,
not the shell working directory. `python /path/to/RigorousRAG/server.py` may therefore be
launched from another directory without changing process CWD. Startup fails closed if a
required bundled frontend asset is missing, non-regular, or a symbolic link.

## Release gate

A generated lock, source inspection, or written test is not a release certificate. Before
merging or publishing a release, require successful exact-head results for:

- dependency consistency;
- compilation and fatal lint;
- full unit/integration tests and measured branch coverage;
- focused Windows classic-storage tests;
- Compose validation and container build;
- every platform/Python release-lock job;
- a final audit of all changes made to fix workflow failures.
