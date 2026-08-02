# Wave 5 publication-recovery verification addendum — 2026-08-02

This addendum records the durable graph-set publication-attempt journal added after the baseline-inclusive 114-test Wave 5 verification. It does not replace or retroactively extend that exact-current count.

## Newly committed capabilities

- [x] Deterministic immutable publication operation IDs.
- [x] Explicit expected-current or no-current pointer binding.
- [x] SQLite phase journal with database/path identity defenses.
- [x] Planned, running, completed, compensated, failed and cancelled states.
- [x] Planned, candidate-stored, pointer-activated, verified and compensated phases.
- [x] Expiring exclusive leases, renewal, reclaim and attempt ceilings.
- [x] Durable previous/candidate IDs, digests and counts.
- [x] Deterministic candidate reconstruction from reviewed proposals.
- [x] Recovery after candidate storage before candidate-phase persistence.
- [x] Recovery after pointer commit before pointer-phase persistence.
- [x] Exact first-publication pointer clearing.
- [x] Exact previous-pointer restoration for failed replacement.
- [x] Post-compensation recovery when terminal persistence was interrupted.
- [x] External-pointer-change refusal without overwrite.
- [x] Generic failure and bounded compensation-error recording.
- [x] Explicit reviewed retry and exact-confirmation cancellation.
- [x] Privacy-safe seed/status/list/execute/reconcile/retry/cancel CLI.
- [x] Dedicated script wrapper and configuration path.
- [x] Previously omitted GraphRAG run/baseline database paths added to `.env.example`.

## Focused verification performed

Two local suites were executed against the publication-journal implementation:

1. **19/19 design and fault-injection contracts passed**, covering detailed crash windows, pointer compensation, lease expiry, retry, cancellation, external pointer changes, tampering, path defenses, runtime caching and CLI behavior.
2. **9/9 repository-compatible committed contracts passed**, exercising the real journal/recovery state machine while replacing only graph construction and authority dependencies with deterministic test doubles.

The combined focused invocation passed **28/28 tests**. Python compilation passed for all publication journal, transition, recovery, runtime and CLI modules.

## Important verification boundary

The earlier exact-current archive verification remains:

- 114/114 evidence-graph focused tests;
- complete repository pytest pass;
- `pip check` pass;
- whole-tree compilation pass.

Those checks preceded the publication-journal commits. They have **not yet been rerun on the new final head** in this addendum because a fresh repository archive or CI execution was not available in the constrained environment.

Therefore this addendum does not claim:

- a new exact-current evidence-graph total;
- full repository regression success after the publication-journal commits;
- Ruff success;
- Windows success;
- Docker/Compose success;
- distributed or multi-process correctness;
- release readiness.

## Remaining publication work

1. Run the complete repository suite on one unchanged final `main` SHA.
2. Add filesystem, SQLite write, process-kill and disk-full fault injection at every durable phase.
3. Add database-scoped or distributed leadership across publication workers and legacy publisher callers.
4. Add reviewer authorization, role separation and auditable principal identity.
5. Add retention, export, legal hold, backup and restore for publication attempts and graph-set history.
6. Add operational dashboards for expired leases, exhausted attempts and compensation failures.
7. Mechanically verify GraphRAG conversion into the existing server-owned citation registry before agent/API integration.

## Permanent non-claims

- A completed publication operation proves pointer and member-generation alignment, not scientific truth.
- A compensated operation is not a successful publication.
- A SQLite lease is not distributed consensus.
- Process-local document locks do not coordinate unrelated processes.
- Candidate history remains immutable after compensation.
- Release readiness is not claimed.
