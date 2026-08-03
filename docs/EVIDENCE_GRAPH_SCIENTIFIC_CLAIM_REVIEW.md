# Reviewed scientific claim candidate extraction

Last updated: 2026-08-03

RigorousRAG now includes a first closed-schema scientific claim extraction and human-review workflow. The adapter identifies bounded sentence spans that are plausible claim candidates, stores only exact provenance and digests, and requires an immutable terminal reviewer decision before a candidate can be materialized as a `GraphAnnotation(node_type="claim")` contract.

The workflow does not automatically approve a claim, assess scientific truth, mutate an evidence graph, publish a graph generation, or infer cross-document semantics.

## 1. Durable ledger configuration

```bash
EVIDENCE_GRAPH_CLAIM_REVIEW_DB_PATH=data/evidence_graph_claim_reviews.sqlite3
```

The SQLite ledger stores:

- owner and document identity;
- authoritative generation, content hash and embedding-profile fingerprint;
- authoritative graph digest;
- exact section-node and section-provenance identity;
- section, page and sentence indexes;
- exact character start/end offsets;
- SHA-256 of the exact source span;
- SHA-256 of whitespace-normalized span text;
- extractor name, version and configuration digest;
- proposer kind and proposer identity;
- extraction score and structural feature flags;
- text-free extractor metadata;
- immutable terminal review decisions and actor-binding provenance.

It does **not** store the claim text itself.

The ledger validates parent-directory identity, database inode identity, regular-file status, and symlink/reparse-point refusal before every connection.

## 2. Candidate extraction

```bash
python scripts/evidence_graph_claim_review.py extract \
  --owner-id alice \
  --doc-id DOCUMENT_ID
```

Optional bounded controls:

```bash
--min-characters 40
--max-characters 1200
--min-words 6
```

The initial extractor is deterministic and rule-based. It operates only on the exact authoritative current evidence graph and considers section-node sentence spans containing one or more structural signals:

- scientific predicate verbs such as *show*, *demonstrate*, *observe*, *reduce*, *increase*, *associate* or *predict*;
- comparative language;
- causal language;
- numeric results;
- citation-shaped references.

Questions are excluded by default. Candidate and document budgets are bounded.

The resulting `extraction_score` represents rule-signal strength and candidate prioritization only. It is not a probability that the claim is true, supported, reproducible or important.

The extraction command writes candidate records to the claim-review ledger but returns no claim text:

- candidate IDs;
- batch and authority digests;
- counts and safety flags.

No review decision, graph annotation or graph publication is performed.

## 3. Text-free queue inspection

```bash
python scripts/evidence_graph_claim_review.py list \
  --owner-id alice \
  --doc-id DOCUMENT_ID
```

```bash
python scripts/evidence_graph_claim_review.py status CANDIDATE_ID
```

These commands return provenance, span coordinates, digests, extractor identity, scores, feature flags and terminal-decision status. They do not return claim text.

## 4. Explicit preview

```bash
python scripts/evidence_graph_claim_review.py preview CANDIDATE_ID \
  --owner-id alice
```

Preview is the only claim-review command that deliberately returns the candidate text. Before doing so it:

1. resolves the document's current authoritative graph;
2. requires exact owner/document/generation/content/profile/graph identity;
3. locates exactly one section node;
4. validates section provenance;
5. validates character bounds;
6. re-computes exact and normalized text digests.

A changed generation, graph, section, provenance value or span digest fails closed.

## 5. Reviewer decision

A process-owned reviewer actor must be configured through the existing actor-binding boundary. The reviewer cannot be the candidate proposer.

```bash
python scripts/evidence_graph_claim_review.py decide CANDIDATE_ID \
  --owner-id alice \
  --decision approved \
  --reviewer-id REVIEWER_ID \
  --reason-code scientifically-supported \
  --confirm-candidate-id CANDIDATE_ID
```

Supported terminal decisions:

- `approved`
- `rejected`
- `superseded`

A superseded decision must name exactly one replacement candidate:

```bash
--replacement-candidate-id REPLACEMENT_ID
```

The replacement must remain within the exact owner, document, authoritative generation, graph and section scope. This prevents a review action from silently laundering a different source, graph generation or section into the original candidate lineage.

The decision record commits:

- candidate and owner;
- terminal decision;
- reviewer identity;
- actor-binding method and digest;
- signed-actor assertion digest, issuer and expiry when applicable;
- reason code;
- optional exact replacement candidate;
- deterministic decision ID and audit timestamp.

A later conflicting terminal decision is refused. Exact replay is idempotent.

The decision command validates exact candidate confirmation before loading the actor or mutating the ledger.

## 6. Materialize approved annotation contracts

```bash
python scripts/evidence_graph_claim_review.py annotations \
  --owner-id alice \
  --doc-id DOCUMENT_ID
```

For each approved candidate, the command re-resolves current authority and revalidates the exact text span before producing a `GraphAnnotation` contract with:

- `node_type="claim"`;
- exact reviewed text;
- page number;
- candidate and decision IDs;
- section node and provenance;
- sentence and character offsets;
- source-text digest;
- extractor/version/config lineage;
- extraction score and feature flags;
- reviewer identity reduced to SHA-256;
- explicit `human_review_required=true`;
- explicit `automatic_approval=false`;
- explicit `graph_publication_performed=false`.

The command intentionally omits annotation text from its JSON output. It returns annotation IDs, page/provenance metadata and safety flags only.

Materialization is not graph publication. A later governed graph-generation workflow must deliberately consume approved annotations and publish a new derived graph generation.

## 7. Privacy and integrity boundaries

- Candidate rows contain no source or claim text.
- General list/status/extract output contains no claim text.
- Preview returns text only after exact owner and authoritative-span revalidation.
- Annotation conversion rehydrates text from current authority rather than trusting a stored copy.
- Reviewer identities are hashed before entering annotation metadata.
- Extraction metadata cannot override generation, graph, span or review lineage.
- Candidate IDs and decision IDs are deterministic digests over their immutable governed scope.
- No automatic approval, graph mutation or graph publication exists.

## 8. Current limitations and next work

The first adapter deliberately favors auditability over recall. Remaining work includes:

- a closed-schema model-assisted claim extractor with deterministic fallback;
- reviewer correction/edit workflows that create replacement candidates rather than editing immutable rows;
- entity-mention and entity-normalization review pipelines;
- citation-link extraction and review;
- semantic claim-support and entailment evaluation;
- contradiction adjudication;
- representative scientific dataset cards and extraction-quality gates;
- connected reviewer UI and multi-party review thresholds;
- complete exact-current repository, platform and container verification.

## 9. Verification boundary

Focused reconstructed execution of the committed adapter contracts passed:

```text
7 passed
```

Focused Python compilation also passed. This evidence covers deterministic text-free extraction, bounded candidate generation, immutable ledger behavior, proposer–reviewer separation, supersession scope, exact text rehydration, approved annotation conversion, source-drift refusal, database-identity refusal, and CLI privacy/confirmation boundaries.

This is not the complete exact-current repository test, coverage, Ruff, Windows, Docker/Compose or independent-process matrix. Release readiness is not claimed.
