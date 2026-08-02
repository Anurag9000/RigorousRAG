# Authoritative evidence-graph-set discovery

Last updated: 2026-08-02

## Purpose

`search_evidence_graph` requires one exact logical `graph_set_key`. An agent or operator must not invent that key. The discovery layer lists the authenticated owner’s current reviewed graph sets and revalidates each member against the authoritative generation and derived graph stores before the key is used for retrieval.

## Privacy and authority contract

`list_evidence_graph_sets` returns only:

- logical graph-set key;
- immutable graph-set ID and digest;
- member and relation-edge counts;
- creation timestamp;
- current-authority boolean and authority digest;
- aggregate stale and missing member counts.

It does not return:

- authenticated owner ID;
- member document IDs;
- graph nodes or node text;
- relation evidence;
- reviewer identities;
- source paths;
- raw queries or matched terms;
- provider responses.

Unavailable/stale sets are hidden by default. `include_unavailable=true` exposes only aggregate stale/missing counts, never the affected document identities.

## Store boundary

The discovery adapter prefers a future public `EvidenceGraphSetStore.list_current` method when present. The current transactional store does not expose one, so the adapter uses the store’s own identity-verified lock, connection and strict row decoder to perform one bounded owner-scoped join between current pointers and immutable graph-set rows.

It revalidates:

- pointer schema version;
- pointer/set owner scope;
- pointer graph-set ID;
- pointer graph-set digest;
- nested graph-set payload identity;
- authoritative member generation and graph identities.

A corrupt pointer or decoded set fails the entire request rather than being silently skipped.

## Agent tools

The existing research agent now receives two ordered tools:

1. `list_evidence_graph_sets` — discover valid owner-scoped keys;
2. `search_evidence_graph` — search one discovered key and return canonical citations.

The discovery tool schema contains only:

```json
{
  "limit": 20,
  "include_unavailable": false
}
```

`owner_id` is never caller-controlled; it is injected from the authenticated `SearchAgent` instance.

Discovery returns a bounded JSON tool result and no citations. Search continues through the canonical citation/evidence registry.

## Operator commands

```bash
python -m tools.evidence_graph_set_discovery_cli \
  --owner-id alice \
  --limit 20

python -m tools.evidence_graph_set_discovery_cli \
  --owner-id alice \
  --limit 20 \
  --include-unavailable
```

Script wrapper:

```bash
python scripts/evidence_graph_set_discovery.py ...
```

Every successful CLI response reports:

```json
{
  "mutation_performed": false,
  "semantic_inference_performed": false,
  "source_text_returned": false
}
```

## Focused verification

The local discovery suite passed **7/7 tests** covering:

- closed schema and server-owned identity;
- public future `list_current` compatibility;
- exact current SQLite pointer join;
- strict pointer ID/digest/schema revalidation;
- authority filtering;
- aggregate-only unavailable reporting;
- owner-scope and budget refusal;
- read-only CLI output and contained errors.

Discovery plus citation conversion, GraphRAG retrieval-tool and agent-registration contracts passed **18/18** together. The complete available local post-head harness passed **52/52 tests**, and Python compilation succeeded.

The production live-agent discovery contract is committed but requires the next exact-current repository run.

## Remaining work

- Add a first-class public `EvidenceGraphSetStore.list_current` method during a separately reviewed store API revision.
- Execute production live-agent discovery and search on one unchanged final head.
- Add connected-provider tests proving the model calls discovery before search when no key is known.
- Add API/browser end-to-end tests with a real current reviewed graph set.
- Add multi-process authority and pointer-race fault injection.

## Permanent non-claims

- A discoverable graph set is provenance-current, not scientifically true.
- Authority checks do not validate reviewer correctness.
- Discovery does not infer relations or select evidence.
- Focused local tests are not the complete release matrix.
- Release readiness is not claimed.
