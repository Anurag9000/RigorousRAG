# Reviewed cross-document relation proposals

Last updated: 2026-08-02

## Purpose and boundary

Cross-document semantic relations are not inferred or published directly from text. They enter a separate immutable review ledger as text-free proposals whose endpoints bind exact document, graph generation, graph digest, node ID and node provenance digest.

A proposal does not become usable merely because it exists. It requires one explicit terminal reviewer decision. Only `approved` proposals can be converted into `ExplicitCrossDocumentRelation` values, and conversion revalidates every endpoint against current authoritative graph views.

## Proposal contract

`CrossDocumentRelationProposal` records:

- owner and logical graph-set key;
- a unique relation key;
- source and target endpoint identities;
- explicit relation type and weight;
- proposer kind and proposer identity;
- evidence SHA-256;
- extractor name/version for model or rule proposals;
- bounded metadata;
- deterministic proposal ID and digest.

Endpoint identity contains:

```text
document ID, graph generation, graph digest,
node ID, node provenance digest
```

It does not contain node text, retrieved passages, retained-source paths or provider responses.

Supported proposer kinds are:

```text
human, model, rule
```

Model and rule proposals must declare both extractor name and extractor version. Human proposals may not claim an extractor identity.

Supported relation types remain the reviewed cross-document set:

```text
cites, same_as, supports, contradicts, derived_from, mentions
```

## Immutable decisions and supersession

Each proposal can receive exactly one terminal decision:

```text
approved, rejected, superseded
```

A decision records reviewer identity and a bounded reason code. `superseded` additionally requires an already persisted replacement proposal under the same owner. A different second decision is refused rather than overwriting review history.

Supersession creates correction lineage; it does not mutate the old proposal.

## Approved relation conversion

`approved_relations(...)` accepts explicit proposal IDs and current authoritative graph views. It requires:

1. proposal owner and graph-set key match the requested scope;
2. a stored terminal `approved` decision exists;
3. every graph view remains authoritative current;
4. source and target graph generations equal the proposal;
5. graph digests, node IDs and node provenance digests still match;
6. relation keys are unique.

The converted relation metadata retains proposal ID, review decision ID, reviewer identity and evidence digest. If either member graph moves, conversion fails closed and the proposal must be reviewed against a new endpoint identity.

Approval never bypasses graph-set authority checks.

## Durable ledger

`RelationReviewLedger` uses SQLite with:

- immutable proposal rows;
- one terminal decision row per proposal;
- idempotent replay of identical proposals/decisions;
- owner-scoped proposal and replacement validation;
- pending/approved/rejected/superseded filters;
- strict reconstruction through the public validated dataclasses;
- database parent/file identity checks.

Configuration:

```dotenv
EVIDENCE_GRAPH_RELATION_DB_PATH=data/evidence_graph_relations.sqlite3
```

## Operator commands

```bash
python -m tools.evidence_graph_relation_cli propose \
  --owner-id alice --graph-set-key review-2026 \
  --relation-key paper-a-supports-paper-b \
  --source-doc-id <doc-a> --source-generation 4 \
  --source-graph-digest <sha256> --source-node-id <sha256> \
  --source-provenance-digest <sha256> \
  --target-doc-id <doc-b> --target-generation 7 \
  --target-graph-digest <sha256> --target-node-id <sha256> \
  --target-provenance-digest <sha256> \
  --edge-type supports --proposer-kind human \
  --proposer-id annotator-17 --evidence-digest <sha256>

python -m tools.evidence_graph_relation_cli decide <proposal-id> \
  --owner-id alice --decision approved \
  --reviewer-id reviewer-3 --reason-code evidence_verified

python -m tools.evidence_graph_relation_cli decide <proposal-id> \
  --owner-id alice --decision superseded \
  --reviewer-id reviewer-3 --reason-code corrected_endpoint \
  --replacement-proposal-id <replacement-id>

python -m tools.evidence_graph_relation_cli status <proposal-id>

python -m tools.evidence_graph_relation_cli list \
  --owner-id alice --graph-set-key review-2026 \
  --decision pending --limit 100
```

No command automatically approves proposals, infers relations or publishes a graph set. Output contains only bounded identities and review metadata and explicitly reports that source text is absent and automatic approval did not occur.

## Focused verification

The focused relation-review harness passed **8 tests** covering:

- deterministic text-free proposals;
- model/rule extractor identity requirements;
- idempotent proposal submission;
- immutable terminal approvals/rejections;
- supersession and replacement-proposal validation;
- pending/rejected/approved filters;
- approved conversion with endpoint generation/graph/node/provenance revalidation;
- stale endpoint refusal;
- text-free CLI propose/status/list/decide behavior;
- bounded not-found and invalid-input errors.

These are focused local results, not the complete exact-head Linux, Windows and container matrix.

## Remaining work

- Add an orchestration layer that builds and commits a new graph-set version from an explicit approved proposal list under optimistic pointer control.
- Recheck every member graph before and after graph-set publication.
- Add proposal/decision archival export, retention and legal-hold policy.
- Add reviewer authorization and separation-of-duties controls.
- Add inter-annotator agreement and relation-precision benchmarks.
- Add closed-schema model/rule proposal adapters with bounded provider calls and stored response digests only.
- Add human correction queues and UI/API integration.

## Permanent non-claims

- Human approval is not proof of semantic entailment or scientific truth.
- Model/rule proposals are never trusted merely because extractor metadata exists.
- `same_as` remains an explicit reviewed assertion, not certain entity identity.
- `supports` and `contradicts` remain reviewed relations, not automated factual judgments.
- A proposal or approval does not publish a graph set by itself.
- Release readiness is not claimed.
