# Exact-head compatibility repair — 2026-08-07

## Purpose

This note records the direct-to-`main` compatibility and lifecycle repair that followed the exhaustive exact-head audit. It is provenance, not a release-readiness claim: release readiness is established only by the exact-head verification workflow on the final unchanged commit.

## Repair commit

The reviewed repair was applied in commit `3e03fd93c22bf3b713afbffb10585f4462d78f19` (`fix: restore exact-head compatibility and lifecycle contracts`). Temporary patch-staging and source-snapshot workflows were removed by that same commit.

## Repaired surfaces

- Restored bounded batch-ingestion vector-generation compatibility while retaining authoritative vector capture.
- Added restore-entrypoint custody preflight before mutating restore commands.
- Hardened release-lock destination identity checks against same-inode content/metadata replacement during generation.
- Stabilized public `SearchAgent` and `ToolExecution` identities and evidence-graph tool installation across import orders.
- Synchronized authoritative ingestion integration tests with the coordinated lifecycle contract.
- Preserved stricter claim-extractor, relation-review, governed-publish, custody-signature, custody-timestamp, deletion, hold-boundary, migration, reconciliation, job-store, RAG, and source-identity invariants while updating obsolete assertions.
- Added bounded behavior for adaptive-route telemetry, BibTeX processing, document-store compatibility, claim-extractor JSON handling, relation policy, index-coordinator structural interfaces, lifecycle boundary compatibility, logging telemetry, migration recovery, and RAG-tool HyDE handling.

## Verification discipline

The repair was syntax-compiled and passed `git diff --check` before publication. A normal direct-to-`main` documentation commit follows this note so GitHub Actions evaluates the repaired code through the complete exact-head matrix. Until that exact final-head run is green, this document deliberately does not label the repository release-ready.

## Cleanup invariant

The temporary `.github/repair/`, `.github/workflows/source-snapshot.yml`, and `.github/workflows/apply-exact-head-repair.yml` staging surfaces are not part of the steady-state repository and must remain absent after the repair.
