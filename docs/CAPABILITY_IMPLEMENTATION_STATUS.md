# Capability implementation status

Last updated: 2026-08-01

This ledger is the authoritative capability-expansion status for `main`. It complements the historical remediation audits. A checked item means source and focused local tests were completed and committed; it does not imply that the repository-wide exact-head release matrix has passed.

## Repository policy

- Development is committed directly to `main`.
- No feature branches or pull requests are used for this repository.
- Commits must keep implementation, tests, documentation, configuration and this ledger aligned.
- Release claims require the authoritative exact-head workflow to pass on one unchanged `main` SHA.

## Wave 1 — retrieval and evaluation foundation

### Implemented

- [x] Typed, bounded retrieval candidates and component score traces.
- [x] Candidate-pool BM25 with correct document-frequency counting.
- [x] Reciprocal-rank fusion and normalized weighted-score fusion.
- [x] MMR relevance/diversity selection with source caps.
- [x] Dense, lexical and hybrid uploaded-document ranking modes.
- [x] Dependency-free heuristic reranker.
- [x] Lazy optional cross-encoder reranker with fail-safe fallback.
- [x] Owner and document filtering before any fusion or reranking.
- [x] Backward-compatible dense/no-reranker default behavior.
- [x] Preservation of raw dense relevance alongside fused/component scores.
- [x] Strict handling of hostile metadata mappings, malformed iterables and non-finite scores.
- [x] Normalized BEIR-style corpus/query/qrels loader.
- [x] Precision, recall, hit-rate, reciprocal-rank, MAP and NDCG metrics.
- [x] Citation precision, recall, F1, coverage and unsupported-citation rate.
- [x] Deterministic experiment matrices with stable run IDs.
- [x] Immutable resumable SQLite experiment results.
- [x] Offline BM25 benchmark CLI.
- [x] Focused regression tests for ranking, evaluation, persistence and adapter boundaries.

### Verification completed

- 12 focused tests passed locally.
- Python compilation passed for all Wave 1 modules, CLI and tests.
- AST parsing passed for every changed Python file.
- Git object IDs were compared with the locally tested bytes before atomic commits.

### Verification not claimed

- Ruff could not run in the constrained local environment because it was not installed and external package resolution was unavailable.
- The full repository test matrix, coverage matrix, Windows jobs, Compose validation and Docker build have not yet run on the final Wave 1 head.
- The exact-head workflow remains the release authority.

## Wave 2 — persistent hybrid index and model registry

### Next implementation slice

- [ ] Declarative embedding model profiles with fingerprints and index schema versions.
- [ ] Owner-scoped persistent fielded sparse index.
- [ ] Title, abstract, heading, body, caption, table and reference field weights.
- [ ] Sparse document snapshots and exact restore.
- [ ] Vector document snapshots and exact restore.
- [ ] Compensating vector+sparse replacement coordinator.
- [ ] Coordinated vector+sparse deletion.
- [ ] Ingestion, batch CLI, retry and recovery integration.
- [ ] Corpus-level dense+sparse retrieval rather than candidate-pool-only lexical scoring.
- [ ] Reindex and embedding-profile migration commands.
- [ ] Generation manifests recording model/profile fingerprints.

## Permanent non-claims

- Retrieval rank is not proof of factual correctness.
- Citation presence is not proof that a claim is entailed.
- Heuristic and learned rerankers may encode bias and require benchmark validation.
- Regex masking is not certified de-identification.
- SQLite/process-local transactions are not distributed exactly-once infrastructure.
- Scientific conclusions require source inspection, expert review and replication.
