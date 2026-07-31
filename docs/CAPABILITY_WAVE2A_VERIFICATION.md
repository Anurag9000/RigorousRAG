# Capability Wave 2A verification record

## Implemented foundation

- declarative built-in and operator embedding profiles;
- stable SHA-256 profile fingerprints;
- historical `all-MiniLM-L6-v2` compatibility resolution;
- explicit unknown-contract compatibility profiles;
- strict duplicate-key/non-standard-number/unknown-field rejection;
- owner-scoped SQLite sparse documents, fields and postings;
- field-weighted BM25-style scoring with document and field filters;
- page, section, field and token-position provenance;
- transactional replacement with optimistic generation checks;
- complete document snapshots, exact restoration and scoped deletion;
- corrupt metadata/posting refusal;
- symlink, Windows reparse-point, parent replacement and database replacement checks.

## Local constrained verification

```text
Wave 2A focused tests: 21 passed
Combined Wave 1 + Wave 2A focused tests: 33 passed
Python compileall: passed
AST parse of every Wave 2A Python file: passed
Sparse replace/search/ping smoke test: passed
Built-in profile fingerprint uniqueness: 7/7
```

Ruff and the repository-wide exact-head matrix were not available in the constrained local
environment. Wave 2A is a storage/model-registry foundation and is intentionally not yet
part of authoritative document ingestion. Wave 2B must wire vector+sparse snapshots,
compensation, deletion, retry/recovery and corpus-level hybrid search before the sparse
store becomes authoritative.
