# Wave 5 status addendum — publication recovery and canonical citations

Last updated: 2026-08-02

This addendum supersedes the open-item status in Sections 8–10 of `WAVE5_IMPLEMENTATION_STATUS_2026-08-02.md`. It does not replace that ledger’s graph-foundation history or extend its earlier exact-current verification count.

## Section 8 — compensating graph-set publication

Now additionally implemented:

- [x] Durable deterministic publication operation IDs.
- [x] Immutable expected-current/no-current pointer contracts.
- [x] SQLite publication phase journal.
- [x] Expiring leases, renewal, reclaim and retry ceilings.
- [x] Durable candidate/previous IDs, digests and counts.
- [x] Recovery after candidate storage before phase persistence.
- [x] Recovery after pointer commit before phase persistence.
- [x] Recovery after compensation before terminal persistence.
- [x] Exact first-publication pointer clearing.
- [x] Exact prior-pointer restoration for replacement failure.
- [x] External-pointer change refusal.
- [x] Completed/compensated/failed/cancelled outcomes.
- [x] Privacy-safe execute/reconcile/status/list/retry/cancel CLI.
- [x] Publication audit and expired/exhausted/compensation-failure classification.
- [x] Planning-only old-terminal retention candidates.

Still open:

- [ ] Distributed/database-scoped leadership across unrelated processes.
- [ ] Crash, disk-full and SQLite write-failure injection in the exact repository stack.
- [ ] Signed audit export, legal hold and backup/restore drills.
- [ ] Separately reviewed destructive compaction.
- [ ] Reviewer authorization and role separation.

## Section 9 — authoritative GraphRAG selection

Now additionally implemented:

- [x] Conversion through the existing `tools.models.Citation` model.
- [x] Graph-set, generation, graph, node, evidence and path lineage metadata.
- [x] Omission of owner IDs, raw queries, raw matched terms and source paths.
- [x] Canonical `local://` uploaded-document citation compatibility.
- [x] Closed `search_evidence_graph` tool schema.
- [x] Server-only owner injection.
- [x] Authoritative runtime-store resolution.
- [x] Empty-list abstention without answer generation.
- [x] Lazy idempotent registration on the existing research agent.
- [x] Existing schema validation, admission, timeout, registry, relabeling and serialization reuse.
- [x] Production-agent integration contracts committed.

Still open:

- [ ] Exact-current execution of the production live-agent contract.
- [ ] API response regression tests for graph citations.
- [ ] Browser safe-DOM citation rendering tests.
- [ ] Connected-provider tool-selection tests.
- [ ] Learned/semantic graph retrieval adapters behind governed benchmarks.

## Section 10 — evaluation and reproducibility

The prior ledger also listed these as open, but they are committed:

- [x] Text-free resumable live benchmark run storage.
- [x] Exact-plan interruption recovery and completed-run reuse.
- [x] Governed append-only historical baseline registry.
- [x] Explicit first-baseline and exact-current replacement expectations.
- [x] Eligible regression-report requirement for baseline promotion.
- [x] Policy-separated baseline pointers and lineage.

Still open:

- [ ] Governed public/scientific graph benchmark dataset cards and licenses.
- [ ] Bootstrap/permutation intervals and multiple-comparison controls.
- [ ] Measured latency, memory, backend I/O and monetary cost.
- [ ] Real connected-provider and multi-process benchmark orchestration.

## Post-ledger focused verification

After the older exact-current verification, the locally available post-head suites passed:

- **32/32** publication journal, recovery, operations and retention-planning contracts;
- **13/13** citation conversion, GraphRAG tool and agent-hook contracts;
- **45/45** when those suites were executed together.

Python compilation passed for the new publication and citation/agent modules.

A production live-agent test is committed but was not executed in that isolated harness. The complete repository suite has not been rerun after these newest commits.

## Exact-current verification boundary

The last unchanged-head full repository evidence remains the pre-journal baseline recorded elsewhere:

- 114/114 evidence-graph focused tests;
- full repository pytest pass;
- `pip check` pass;
- whole-tree compilation pass.

Those results preceded the publication-recovery and canonical citation/agent commits. This addendum does not claim a new exact-current full-suite total.

## Permanent non-claims

- Durable recovery is not distributed consensus.
- Citation conversion establishes provenance, not truth.
- Agent registration does not prove correct model tool selection.
- A retention candidate is not deletion authorization.
- Focused tests are not the complete release matrix.
- Release readiness is not claimed.
