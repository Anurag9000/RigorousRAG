# Executable verification ledger

This ledger records observed workflow execution. It distinguishes a source-level test
contract from a successful run against a concrete pull-request head or merge ref.

## Release-lock matrix

The first release-lock runs exposed and corrected three independent workflow defects:

1. the isolated lock environment lacked `typing-extensions` required by pip-tools;
2. pip-tools 7.6 was incompatible with pip 26 through its sync-module import boundary;
3. a `shell: python` Actions step executed from a temporary directory and could not import
   the repository's `scripts` package.

The final lock generator:

- installs an isolated `pip>=25,<26` and pip-tools 7.6 toolchain;
- resolves with backtracking;
- pins unsafe bootstrap packages rather than leaving warnings;
- emits exact SHA-256 hashes;
- publishes its verified absolute output path directly through `GITHUB_OUTPUT`;
- validates every requirement is exactly pinned and hashed;
- performs a `pip --require-hashes --no-deps --dry-run` installation check;
- uploads the generated lock artifact.

Observed successful run:

- workflow run: `30547701731`;
- source head: `5268f9168dbb184be0b09e41af6f8931f2444aaf`;
- result: all nine jobs succeeded;
- platforms: Linux, Windows, macOS;
- Python: 3.10, 3.11, 3.12;
- every job passed generation, lock verification, hash-required installation dry run, and
  artifact upload.

The later workflow consolidation changes orchestration only; the lock generator, verifier,
toolchain input, and unit tests remain the same. The full exact-head workflow must still
rerun after every subsequent commit.

## Consolidated exact-head workflow

The previously silent CI workflow did not produce an observable run. Every gate has now
been consolidated into the already-active workflow identity. The consolidated workflow
contains:

- Linux dependency consistency, whitespace comparison, compilation, fatal Ruff checks,
  pytest, and measured branch coverage on Python 3.10–3.12;
- focused Windows classic-storage compilation and regressions on Python 3.10 and 3.12;
- Docker Compose validation and container build;
- the nine-job release-lock matrix above;
- pull-request, branch-push, merge-queue, tag, and manual triggers;
- one concurrency group that cancels superseded exact-head runs.

No merge-readiness claim is permitted until this consolidated workflow succeeds against
the final pull-request head and every failure discovered by it has been corrected.
