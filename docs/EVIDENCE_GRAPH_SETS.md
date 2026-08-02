# Provenance-preserving cross-document evidence graph sets

Last updated: 2026-08-02

## Purpose

Cross-document graph sets connect exact authoritative evidence-graph generations without collapsing document identity. Every set member retains its owner, document ID, authoritative generation, finalized content hash, embedding profile fingerprint, graph digest and authority digest.

Relations are explicit reviewed assertions. The implementation does not automatically match citations, resolve entities, infer entailment or detect contradictions from text.

## Immutable contracts

`tools/evidence_graph_sets.py` provides:

- `GraphGenerationReference` for one exact authoritative member graph;
- `CrossDocumentNodeReference` containing node identity, type, label, page/section locator and provenance digest but no node text;
- `ExplicitCrossDocumentRelation` as the reviewed input boundary;
- `CrossDocumentEdge` with deterministic identity and bounded metadata;
- `EvidenceGraphSet` with deterministic identity over its logical key and exact member generations;
- bounded directed neighbors and cycle-safe cross-document paths.

Supported explicit cross-document edge types are:

```text
cites, same_as, supports, contradicts, derived_from, mentions
```

`contains` and other within-document structural relations are not accepted as cross-document edges.

A graph set requires at least two unique documents. Every member authority view must be `authoritative_current=true` when the set is constructed. If any member generation changes, the graph-set ID changes.

## No text duplication

Graph sets do not copy graph node text. Endpoint references contain:

- document and generation;
- graph and node identities;
- node type and label;
- page and section locator;
- node provenance digest.

The underlying authoritative graph remains the source of any privacy-finalized node text.

## Transactional versions and authority

`tools/evidence_graph_set_store.py` provides:

- append-only immutable graph-set versions;
- idempotent same-ID/same-digest replay;
- collision refusal;
- one optimistic current pointer per owner/logical set key;
- expected-current compare-and-swap checks;
- bounded history;
- exact-digest deletion of non-current versions only;
- strict nested JSON revalidation;
- payload, pointer, symlink/reparse and database-identity defenses.

A logical current set is served only when every member still matches:

- the authoritative generation sequence;
- finalized content SHA-256;
- embedding profile fingerprint;
- the current evidence-graph generation and graph digest.

If any member is missing or stale, current resolution fails closed with `stale_graph_set`. Historical versions remain inspectable and are marked non-current.

## Read-only operator surface

Configuration:

```dotenv
EVIDENCE_GRAPH_SET_DB_PATH=data/evidence_graph_sets.sqlite3
```

Commands:

```bash
python -m tools.evidence_graph_set_cli status \
  --owner-id alice --graph-set-key review-2026

python -m tools.evidence_graph_set_cli status \
  --owner-id alice --graph-set-key review-2026 \
  --graph-set-id <historical-set-id>

python -m tools.evidence_graph_set_cli history \
  --owner-id alice --graph-set-key review-2026 --limit 100

python -m tools.evidence_graph_set_cli neighbors \
  --owner-id alice --graph-set-key review-2026 \
  --doc-id <source-document> --node-id <source-node>

python -m tools.evidence_graph_set_cli paths \
  --owner-id alice --graph-set-key review-2026 \
  --source-doc-id <doc-a> --source-node-id <node-a> \
  --target-doc-id <doc-b> --target-node-id <node-b> \
  --max-depth 6 --max-paths 20
```

The CLI is read-only. It exposes identities, labels, locators, explicit edge types, weights and provenance digests, but no node text. It reports:

```json
{
  "mutation_performed": false,
  "semantic_inference_performed": false
}
```

There is intentionally no generic JSON relation-import command. Cross-document semantic relations must pass a separately reviewed closed-schema construction workflow before persistence.

## Focused verification

The focused graph-set suite passed **14 tests** covering:

- deterministic identities independent of member ordering and creation time;
- exact member generation/provenance preservation;
- absence of node text from graph-set payloads;
- rejection of stale authority views, duplicate documents, same-document relations, unsupported relation types and unknown endpoints;
- new set identity after any member-generation change;
- bounded directed neighbors and filtered multi-document paths;
- append-only versions and optimistic current-pointer updates;
- idempotent replay, collision and exact-deletion boundaries;
- strict JSON and database-identity tamper detection;
- fail-closed member authority;
- read-only CLI output and historical inspection.

These are focused local results, not the exact-head Linux/Windows/container release matrix.

## Remaining work

- Add a reviewed relation-import format with schema version, annotator/extractor identity, model/version where applicable, source evidence and human-review state.
- Add human correction and supersession lineage for relations.
- Add graph-set audit, retention and backup/restore policy.
- Add query-driven bounded GraphRAG selection over exact member graphs.
- Convert selected graph paths into server-owned evidence/citations without citation laundering.
- Add cross-document path completeness, relation precision and stale-set regression benchmarks.
- Add distributed coordination for concurrent graph-set publication.
- Add API/browser integration only after authority-aware citation contracts are complete.

## Permanent non-claims

- `same_as` is an explicit assertion, not automatic entity-resolution certainty.
- `supports` is an explicit relation, not verified semantic entailment.
- `contradicts` is an explicit relation, not proof of scientific falsification.
- A cross-document path explains stored assertions; it is not causal proof.
- A current graph-set pointer is valid only while all member authority checks pass.
- Release readiness is not claimed.
