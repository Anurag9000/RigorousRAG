# Capability Wave 5 — provenance evidence graph and derived reconciliation

Last updated: 2026-08-02

## Scope

Wave 5 now provides two deliberately separated layers:

1. a typed, generation-scoped provenance graph for documents, sections, claims, entities, methods, datasets and citations;
2. an operator-driven derived-index reconciliation path that rebuilds document/section structure from one exact authoritative sparse generation or publishes a deleted-generation tombstone.

The semantic graph remains explicit-only. The derived reconciler never infers support, contradiction, entity equivalence, citation intent, methods or datasets from text. Semantic relations enter the graph only through validated `GraphAnnotation` and `ExplicitGraphRelation` inputs supplied by a reviewed extractor or operator.

The graph is still a rebuildable derived index. It is not an automatically coordinated fifth participant in authoritative ingestion.

## Typed deterministic graph contracts

`tools/evidence_graph_types.py` provides:

- validated `EvidenceNode` and `EvidenceEdge` values;
- owner/document/generation isolation;
- deterministic SHA-256 node and edge identities;
- bounded labels, privacy-finalized text, page/section provenance and metadata;
- finite edge weights and self-loop refusal;
- exact endpoint validation;
- exactly one document node per graph batch;
- immutable `EvidenceGraphBatch` digests excluding creation time;
- ordered path values with exact adjacency validation.

Supported node types:

```text
document, section, claim, entity, method, dataset, citation
```

Supported edge types:

```text
contains, mentions, supports, contradicts, cites,
uses_method, uses_dataset, derived_from, same_as
```

## Explicit graph construction

`tools/evidence_graph_builder.py` builds a graph from one finalized document plus explicit annotations and relations.

It verifies document identity and finalized content, creates deterministic document/section nodes, attaches explicitly supplied annotation nodes to their declared container, and adds only explicitly supplied semantic relations. Duplicate keys, unknown endpoints, invalid section references and caller-supplied containment edges are rejected.

Textual similarity or disagreement alone never creates a support or contradiction edge.

## Transactional graph generations

`tools/evidence_graph_store.py` provides:

- immutable owner/document/generation graph rows;
- strictly encoded complete batches;
- idempotent same-generation/same-digest replay;
- generation collision refusal;
- one optimistic current pointer per owner/document;
- exact expected-current checks for commit and activation;
- bounded history;
- exact-digest deletion of non-current generations only;
- strict JSON, digest, pointer and owner-scope revalidation;
- symlink/reparse and database/parent identity defenses.

The graph database is append-only except for its current pointer and exact deletion of a non-current generation.

## Retrieval and explicit conflict analysis

`tools/evidence_graph_retrieval.py` provides deterministic lexical node search, node-type filters, directed cycle-safe breadth-first paths, edge/intermediate-type filters and deterministic neighbor inspection.

`tools/evidence_graph_analysis.py` reports structural counts and incoming explicit support/contradiction clusters for claim nodes. A conflict is reported only when stored edges explicitly provide both relation types. Unlinked contradictory-looking text produces no semantic cluster.

## Durable exact-generation graph jobs

`tools/evidence_graph_jobs.py` adds an operator-driven SQLite journal for immutable authoritative-generation jobs.

A job identity binds:

- owner and document;
- authoritative generation sequence and state;
- finalized content SHA-256;
- embedding profile fingerprint;
- authoritative sparse generation.

Job IDs are deterministic SHA-256 values. The state machine is:

```text
planned -> running -> completed
                    -> failed
planned/failed -> cancelled
failed -> planned (explicit reviewed retry)
```

The journal provides:

- idempotent exact-identity seeding;
- exclusive worker leases;
- lease expiry and reclaim;
- bounded attempt ceilings;
- lease renewal;
- generic failure types rather than private exception text;
- owner-scoped retry and cancellation;
- exact SHA-256 graph digests on completion;
- finite timestamp validation;
- strict path, symlink/reparse, parent and database identity defenses.

A completed job cannot be claimed again. A changed authoritative generation produces a new job ID rather than mutating the old job.

## Structural derived-graph reconciliation

`tools/evidence_graph_reconcile.py` reconciles one leased job while holding the same process-local striped owner/document lock used by index coordination.

For active or restored authoritative generations it:

1. verifies the current generation exactly matches the immutable job;
2. captures the authoritative sparse document snapshot;
3. verifies owner, document, sparse generation, profile and any declared content hash;
4. creates one document node and one section node per authoritative sparse field;
5. creates only deterministic `contains` edges;
6. records field ID, field type, position, token count, page and section provenance;
7. rechecks the authoritative generation after graph construction;
8. refuses to move the graph pointer backwards;
9. commits or idempotently reuses the exact graph generation;
10. rechecks the authoritative generation after publication;
11. marks the job completed only with the published graph digest.

For deleted authoritative generations it requires the sparse snapshot to be absent and publishes a one-document-node tombstone graph with no edges.

The reconciler never mutates:

- vector rows;
- sparse rows;
- retained source files;
- document registry state;
- authoritative generation history or current pointers.

Only the evidence-graph database and graph-job journal are mutated.

## Runtime and operator commands

Configuration:

```dotenv
EVIDENCE_GRAPH_DB_PATH=data/evidence_graph.sqlite3
EVIDENCE_GRAPH_JOB_DB_PATH=data/evidence_graph_jobs.sqlite3
```

Read-only graph inspection remains available through `tools/evidence_graph_cli.py` and `scripts/evidence_graphs.py`.

Derived reconciliation is exposed through `tools/evidence_graph_jobs_cli.py` and `scripts/evidence_graph_jobs.py`:

```bash
python -m tools.evidence_graph_jobs_cli seed \
  --owner-id alice --doc-id <document-id> --max-attempts 3

python -m tools.evidence_graph_jobs_cli status <job-id>

python -m tools.evidence_graph_jobs_cli list \
  --owner-id alice --state failed --limit 100

python -m tools.evidence_graph_jobs_cli reconcile-one \
  --owner-id alice --worker-id graph-worker-1 --lease-seconds 60

python -m tools.evidence_graph_jobs_cli retry <job-id> \
  --owner-id alice --confirm-job-id <same-job-id>

python -m tools.evidence_graph_jobs_cli cancel <job-id> \
  --owner-id alice --confirm-job-id <same-job-id>
```

Operator output contains IDs, states, counts, sequence/profile/content identities, lease metadata, graph digests and generic failure types. It contains no graph text, sparse text, retained path or provider response. Every job summary reports:

```json
{
  "authoritative_mutation_performed": false,
  "semantic_inference_performed": false
}
```

## Verification evidence

The original clean-archive Wave 5 foundation passed **28 focused tests** covering deterministic graph identities, explicit construction, transactional generations, lexical retrieval, path traversal, explicit support/contradiction analysis, corruption defenses and the read-only CLI.

The derived reconciliation slice passed **14 focused local tests** covering:

- immutable exact-generation job IDs;
- NaN, Infinity, boolean and negative timestamp refusal;
- deleted-generation sparse-count constraints;
- idempotent seeding;
- exclusive claims, expiry/reclaim and attempt ceilings;
- completion digests and generic failures;
- owner-scoped retry/cancellation;
- active document/section structural graphs;
- deleted-generation tombstones;
- absence of semantic edges;
- exact-generation idempotent publication;
- stale generation and sparse identity refusal before publication;
- path-free CLI output;
- bounded not-found and idle behavior;
- exact confirmation and invalid policy refusal.

Python compilation passed for the job journal, reconciler, runtime, CLI and script in the constrained local harness. Full exact-head repository, Windows, container and multi-process fault testing remains required.

## Deliberate safety boundaries

### No automatic ingestion hook

Authoritative ingestion does not automatically enqueue or execute graph jobs. Operators must seed and reconcile them explicitly. This avoids silently introducing an uncoordinated fifth write participant into the current four-store lifecycle.

### Single-process locking only

The shared striped owner/document lock coordinates graph reconciliation with index work only inside one Python process. It is not distributed leadership or a database-wide consensus lock. Independent processes can still race unless operators serialize them externally.

The reconciler performs generation checks before construction, before publication and after publication, but automatic multi-process execution remains disabled until durable leadership and crash/fault injection exist.

### Derived availability is not authoritative correctness

Graph availability is not required to validate vector/sparse/generation correctness. Consumers that eventually use the graph for retrieval must compare graph generation, content hash and profile fingerprint with the authoritative current generation before publishing graph-derived evidence.

## Remaining Wave 5 work

- Add startup or periodic scheduling only after single-leader/multi-process coordination is implemented.
- Add an authoritative-generation-aware graph reader that refuses stale current graph pointers.
- Add graph-job audit export, retention, compaction and dead-letter policy.
- Add exact crash injection around claim, graph insert, current-pointer publication and job completion.
- Add cross-document graph-set types without collapsing owner/document/generation provenance.
- Add reviewed cross-document citation and entity-resolution workflows.
- Add bounded GraphRAG retrieval over cross-document paths.
- Add path-aware evidence selection and server-owned citation conversion.
- Add graph retrieval benchmarks, path-completeness metrics and historical regression thresholds.
- Add reviewed scientific extraction adapters with closed schemas and human-auditable provenance.
- Add graph database backup/restore and disaster-recovery policy.
- Run exact-head concurrency, corruption, Windows and container verification.

## Permanent non-claims

- A stored `supports` edge is an explicit annotation, not independently verified entailment.
- A stored `contradicts` edge is an explicit annotation, not proof of scientific falsification.
- Lexical graph rank is not semantic truth.
- A graph path is a provenance explanation of stored edges, not causal proof.
- `same_as` is an explicit asserted relation, not automatic entity-resolution certainty.
- A completed derived job does not make the graph an authoritative lifecycle participant.
- The process-local document lock is not distributed coordination.
- The graph is not yet complete for every authoritative document.
- Release readiness is not claimed.
