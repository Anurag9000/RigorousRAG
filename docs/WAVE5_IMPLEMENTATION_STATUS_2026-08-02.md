# Wave 5 evidence-graph implementation ledger

Last updated: 2026-08-02

This ledger consolidates the evidence-graph work committed directly to `main`. Checked implementation means source, focused contracts and documentation are committed. It does not imply that the full cross-platform release matrix has passed.

## 1. Typed provenance graph foundation

Implemented:

- [x] Typed document, section, claim, entity, method, dataset and citation nodes.
- [x] Deterministic generation-scoped node IDs.
- [x] Typed containment, support, contradiction, citation, method, dataset, derivation, mention and equivalence edges.
- [x] Deterministic edge IDs and provenance digests.
- [x] Explicit-only semantic annotations and relations.
- [x] Structural document/section graph construction without inferred semantics.
- [x] Strict bounded metadata, text, identifiers, counts and numeric validation.
- [x] Deterministic graph batch digests.

Permanent boundary: the builder does not infer claim support, contradiction, entity equivalence, methods, datasets or citations from text.

## 2. Transactional generation-scoped graph storage

Implemented:

- [x] Owner/document/generation-isolated SQLite graph versions.
- [x] Immutable generation rows and idempotent same-digest replay.
- [x] Optimistic current-generation pointers.
- [x] Strict nested JSON reconstruction on every read.
- [x] Exact-digest deletion of non-current versions only.
- [x] Owner isolation and current-pointer corruption refusal.
- [x] Symlink/reparse, parent and database identity defenses.

## 3. Single-document graph retrieval and analysis

Implemented:

- [x] Deterministic bounded lexical node search.
- [x] Node-type filters.
- [x] Directed outgoing-neighbor retrieval.
- [x] Cycle-safe bounded path explanations.
- [x] Edge-type filters.
- [x] Explicit-edge-only support and contradiction clusters.
- [x] Read-only privacy-conscious CLI for status, history, search, paths and analysis.

Permanent boundary: retrieved paths explain stored assertions; they do not establish causality or entailment.

## 4. Derived-graph reconciliation and authority

Implemented:

- [x] Deterministic exact-generation graph job identities.
- [x] Durable leased job journal with planned/running/completed/failed/cancelled states.
- [x] Retry ceilings, lease expiry/reclamation and generic failure types.
- [x] Structural graph reconstruction from authoritative sparse snapshots.
- [x] Deleted-generation tombstone graph publication.
- [x] Exact generation, content, profile and sparse-generation checks.
- [x] Shared owner/document locking during reconciliation.
- [x] Idempotent replay and stale-generation refusal.
- [x] Authority resolver binding graph current state to authoritative generation current state.
- [x] Read refusal for stale or missing derived graphs.
- [x] Operator-driven job/reconciliation CLI.

Still open:

- [ ] Automatic ingestion/deletion outbox integration.
- [ ] Multi-process reconciliation leadership.
- [ ] Crash injection across job claim, graph commit and pointer update.

## 5. Operational audit and retention planning

Implemented:

- [x] Queue-state counts and expired-lease detection.
- [x] Retryable-failure and exhausted-dead-letter classification.
- [x] Superseded nonterminal job detection.
- [x] Current, stale and missing/mismatched artifact classification.
- [x] Conservative terminal-job retention candidate planning.
- [x] Deterministic privacy-safe audit/plan digests.
- [x] Planning-only CLI with no deletion authorization.

Still open:

- [ ] Signed or append-only archival export.
- [ ] Legal hold and minimum-retention policy.
- [ ] Exact-confirmation destructive compaction.
- [ ] Backup/restore validation.

## 6. Cross-document graph sets

Implemented:

- [x] Exact generation-bound graph member references.
- [x] Member content/profile/graph/authority digests.
- [x] Text-free cross-document node references with labels and locators.
- [x] Explicit cross-document cites/same-as/supports/contradicts/derived-from/mentions edges.
- [x] Deterministic set identity over logical key and exact member generations.
- [x] Bounded neighbors and cycle-safe cross-document paths.
- [x] Append-only set versions.
- [x] Optimistic logical-key current pointer.
- [x] Strict nested revalidation and exact non-current deletion.
- [x] Fail-closed authority if any member generation or graph digest moves.
- [x] Read-only status/history/neighbors/path CLI.

Permanent boundary: no citation matching, entity resolution or relation inference is performed automatically.

## 7. Reviewed semantic-relation governance

Implemented:

- [x] Deterministic text-free relation proposals.
- [x] Endpoint generation, graph, node and provenance identities.
- [x] Human/model/rule proposer classification.
- [x] Required extractor name/version for model and rule proposals.
- [x] Immutable approved/rejected/superseded reviewer decisions.
- [x] Replacement-proposal validation and supersession lineage.
- [x] Pending/approved/rejected/superseded filters.
- [x] Approved-relation conversion with current endpoint revalidation.
- [x] Text-free proposal/review CLI with no automatic approval.

Still open:

- [ ] Reviewer authorization and separation of duties.
- [ ] Inter-annotator agreement reports.
- [ ] Human review UI and correction queue integration.
- [ ] Governed bounded model/rule proposal adapters.

## 8. Compensating graph-set publication

Implemented:

- [x] Explicit approved-proposal list requirement.
- [x] Explicit first-publication or exact-current pointer expectation.
- [x] Deterministic member lock ordering.
- [x] Current member graph resolution and endpoint revalidation.
- [x] Non-current immutable candidate persistence.
- [x] Pre-activation member authority check.
- [x] Optimistic pointer compare-and-swap.
- [x] Post-activation member/pointer verification.
- [x] Previous-pointer restoration after replacement failure.
- [x] Exact candidate-pointer clearing after first-publication failure.
- [x] Compensation verification and bounded error labels.
- [x] Privacy-safe publication CLI.

Still open:

- [ ] Durable publication-attempt phase journal.
- [ ] Multi-process/database-scoped publication leases.
- [ ] Crash and disk-failure injection at every publication boundary.
- [ ] Failed-candidate retention policy.

## 9. Bounded authoritative GraphRAG selection

Implemented:

- [x] Current graph-set authority resolution before selection.
- [x] Deterministic member locking and before/after authority rechecks.
- [x] Per-document lexical seeds with global budgets.
- [x] Closed node and edge filters.
- [x] Stored within-document edge expansion.
- [x] Reviewed cross-document edge expansion.
- [x] Target generation/graph/node provenance verification.
- [x] Cycle prevention and depth/per-seed/global ceilings.
- [x] Generation-scoped deduplication and deterministic ordering.
- [x] Step-level traversal lineage.
- [x] Conservative empty-evidence abstention.
- [x] Query digest rather than raw query in selection output.
- [x] Permanent no-answer/no-citation-conversion type boundary.

Still open:

- [ ] Server-owned conversion to the existing citation/evidence registry.
- [ ] Agent tool registration.
- [ ] API/browser propagation and safe-DOM tests.
- [ ] Semantic/learned graph retrieval adapters behind benchmarks.

## 10. Evaluation, reproducibility and historical regression

Implemented:

- [x] Generation-scoped node precision/recall/F1.
- [x] Document precision/recall/F1.
- [x] Required traversal-edge precision/recall/F1.
- [x] Complete required-path rate.
- [x] Expanded-lineage completeness.
- [x] Abstention accuracy.
- [x] Evidence/traversal/work accounting.
- [x] Macro aggregation and deterministic digests.
- [x] Strict query-digest-only gold cases.
- [x] Repeated-run and explicit-seed benchmark fixtures.
- [x] Contract fingerprint excluding selection outputs.
- [x] Text-free selection observations.
- [x] Strict benchmark CLI, path defenses and atomic report writes.
- [x] Historical aggregate floors.
- [x] Paired per-run normal-approximation confidence intervals.
- [x] Metric-specific non-inferiority margins.
- [x] Estimated-work ratio ceiling.
- [x] Strict eligible/blocked comparison CLI.
- [x] Live query-resolver bridge with digest verification before selector invocation.
- [x] Immediate live-selection reduction to text-free observations.
- [x] Direct execution through the authoritative current-set selector.

Still open:

- [ ] Governed benchmark dataset cards, checksums, versions, splits and licenses.
- [ ] Bootstrap/permutation intervals and multiple-comparison controls.
- [ ] Measured latency, memory, backend I/O and cost.
- [ ] Resumable append-only benchmark run storage.
- [ ] Versioned release baseline registry.

## Focused verification

The current Wave 5 implementation has focused contracts for:

- graph types, builders, storage, retrieval, analysis and CLI;
- derived jobs, reconciliation, authority and operations;
- cross-document sets, persistence, authority and traversal;
- relation proposals, decisions and supersession;
- compensating reviewed set publication;
- authoritative GraphRAG selection;
- per-case evaluation and macro aggregation;
- strict repeated-run benchmarks and CLI;
- historical regression engine and CLI;
- live selector execution and text reduction.

A fresh exact-current `main` archive passed:

- **96/96 evidence-graph focused tests** after the live benchmark bridge;
- the repository’s complete pytest suite in the available Linux environment;
- `pip check`;
- whole-tree Python compilation.

Ruff and Docker/Compose are kept as separate environment-dependent checks and are not inferred as green. Windows, full container build/readiness, connected providers, multi-process fault injection and one final unchanged-head line-by-line release audit remain required.

## Permanent non-claims

- Evidence graph identity and authority establish provenance, not truth.
- Explicit support/contradiction relations are reviewed assertions, not automatic entailment.
- Stored paths are explanations of graph structure, not causal proof.
- GraphRAG selection is not an answer.
- Retrieval metrics do not establish scientific correctness.
- Normal-approximation intervals are not exact small-sample inference.
- Estimated work is not measured latency, memory or cost.
- A green focused or Linux test suite is not the complete release matrix.
- Release readiness is not claimed.
