# Reviewed scientific claim extraction

Last updated: 2026-08-05

## Purpose

RigorousRAG now has a bounded, provenance-preserving workflow for converting scientific-claim extractor output into reviewable evidence-graph annotations.

The workflow deliberately separates five stages:

1. closed-schema extraction;
2. immutable proposal submission;
3. governed human review;
4. approved annotation conversion;
5. explicit graph construction.

An extractor cannot write an evidence graph, approve a proposal, create a support or contradiction relation, or bypass reviewer policy.

## 1. Exact extraction scope

`tools/evidence_graph_claim_extraction.py` accepts one finalized document generation and extractor output with exactly this top-level schema:

```json
{
  "schema_version": 1,
  "claims": []
}
```

Each claim may contain only:

```text
claim_key
claim_text
claim_type
modality
section_index
page_number (optional)
char_start
char_end
confidence
supersedes_proposal_id (optional)
metadata (optional)
```

Supported claim types are:

```text
finding
hypothesis
causal
associational
comparative
methodological
limitation
null_result
negative_result
recommendation
```

Supported modalities are:

```text
asserted
suggested
conditional
uncertain
negated
```

The adapter rejects:

- unknown fields;
- duplicate JSON keys;
- unsupported schema versions;
- duplicate claim keys;
- non-finite confidence values;
- out-of-range sections or character offsets;
- empty evidence spans;
- page values inconsistent with finalized section provenance;
- document content hashes inconsistent with finalized text;
- model/rule proposals without extractor name and version.

The extractor does not provide the evidence digest. The adapter slices the exact finalized section text at the validated character offsets and computes `evidence_sha256` itself.

Every proposal is bound to:

- owner;
- document ID;
- authoritative generation;
- finalized content SHA-256;
- embedding/profile fingerprint;
- exact evidence locator and digest;
- claim text and taxonomy;
- proposer and extractor identity;
- confidence;
- correction predecessor;
- bounded metadata.

Creation time is excluded from deterministic proposal identity.

## 2. Programmatic extraction

```python
from tools.evidence_graph_claim_extraction import (
    extract_scientific_claim_proposals,
)
from tools.evidence_graph_claim_runtime import (
    get_scientific_claim_review_store,
)
from tools.evidence_graph_claim_submission import (
    submit_scientific_claim_proposals,
)

batch = extract_scientific_claim_proposals(
    finalized_document,
    extractor_output,
    owner_id="alice",
    generation=authoritative_generation,
    profile_fingerprint=profile_fingerprint,
    proposer_id="scientific-claim-extractor",
    extractor_name="closed-schema-claims",
    extractor_version="1.0.0",
)

stored = submit_scientific_claim_proposals(
    get_scientific_claim_review_store(),
    batch.proposals,
)
```

`extract_scientific_claim_proposals` performs no persistence. `submit_scientific_claim_proposals` topologically orders same-batch correction chains and delegates one atomic low-level store transaction. Returned values preserve caller order.

There is intentionally no CLI that accepts arbitrary proposal JSON without revalidating it against finalized document sections.

## 3. Immutable correction lineage

A corrected proposal names `supersedes_proposal_id`.

The review store requires corrections to remain in the exact same:

- owner;
- document;
- generation;
- content hash;
- profile fingerprint.

Each proposal may have at most one correction successor. Correction branches are refused.

A corrected proposal cannot be approved until its predecessor has a terminal `superseded` decision naming that exact replacement. This prevents an unreviewed correction from silently replacing an earlier claim.

The canonical submission boundary accepts predecessor and successor in any input order, detects same-batch cycles, and inserts predecessor-first.

## 4. Governed review

Configure the dedicated claim-review store:

```dotenv
EVIDENCE_GRAPH_CLAIM_REVIEW_DB_PATH=data/evidence_graph_claim_reviews.sqlite3
```

Configure exactly one policy source:

```dotenv
EVIDENCE_GRAPH_CLAIM_REVIEW_POLICY_PATH=config/evidence_graph_claim_review_policy.example.json
EVIDENCE_GRAPH_CLAIM_REVIEW_POLICY_JSON=
```

The example file contains placeholders and is not active by default. Replace every placeholder before selecting it.

Claim review reuses the process-owned reviewer actor boundary:

```dotenv
EVIDENCE_GRAPH_REVIEW_ACTOR_ID=reviewer-1
```

or one of the protected descriptor-file/signed-assertion actor modes documented by the relation-review governance stack.

The claim policy scopes reviewers by:

- reviewer ID;
- owner;
- document ID;
- allowed terminal decisions;
- optional expiry.

Supported decisions are:

```text
approved
rejected
superseded
```

The service enforces:

- process-owned reviewer identity;
- policy scope and expiry;
- proposer–reviewer separation;
- replacement-author separation;
- exact correction scope;
- atomic terminal decision plus authorization insertion;
- deterministic authorization receipts;
- stable exact replay preserving original stored timestamps.

Review a proposal:

```bash
python scripts/evidence_graph_claims.py decide PROPOSAL_ID \
  --owner-id alice \
  --decision approved \
  --reviewer-id reviewer-1 \
  --reason-code scientific-review-complete
```

Supersede a proposal:

```bash
python scripts/evidence_graph_claims.py decide ORIGINAL_PROPOSAL_ID \
  --owner-id alice \
  --decision superseded \
  --reviewer-id reviewer-1 \
  --reason-code corrected-claim \
  --replacement-proposal-id CORRECTED_PROPOSAL_ID
```

## 5. Privacy-conscious inspection

```bash
python scripts/evidence_graph_claims.py status PROPOSAL_ID
```

```bash
python scripts/evidence_graph_claims.py list \
  --owner-id alice \
  --doc-id DOCUMENT_ID \
  --generation GENERATION \
  --decision pending
```

The CLI reports:

- IDs and digests;
- owner/document/generation scope;
- content/profile digests;
- claim taxonomy and confidence;
- claim-text digest and length;
- exact section/page/character locators;
- evidence and locator digests;
- proposer/extractor identity;
- correction lineage;
- decision and authorization provenance.

It does not return claim text, evidence text, source paths, document text, or model response text.

## 6. Conversion to explicit graph annotations

Only proposals with an exact `approved` decision and matching authorization can become graph annotations.

```python
from tools.evidence_graph_claim_review import approved_claim_annotations

annotations = approved_claim_annotations(
    owner_id="alice",
    doc_id=document_id,
    generation=generation,
    content_sha256=content_sha256,
    profile_fingerprint=profile_fingerprint,
    proposal_ids=approved_proposal_ids,
    store=claim_review_store,
)
```

The conversion revalidates:

- requested generation scope;
- decision/authorization identity;
- reviewer identity;
- correction predecessor state;
- absence of a newer approved correction.

Each result is the repository’s existing `GraphAnnotation` with `node_type="claim"` and metadata containing proposal, decision, authorization, policy, grant, evidence-locator, extractor and correction-lineage digests.

No graph is mutated by conversion. The resulting annotations must still be passed explicitly to `build_evidence_graph` and committed through the existing authoritative-generation graph job/store controls.

The conversion does not create `supports`, `contradicts`, `derived_from`, `same_as`, citation, method or dataset relations.

Privacy-conscious annotation inspection:

```bash
python scripts/evidence_graph_claims.py annotations \
  --owner-id alice \
  --doc-id DOCUMENT_ID \
  --generation GENERATION \
  --content-sha256 CONTENT_SHA256 \
  --profile-fingerprint PROFILE_FINGERPRINT \
  --proposal-id APPROVED_PROPOSAL_ID
```

The command returns label/text digests rather than claim text and reports:

```text
mutation_performed: false
graph_mutation_performed: false
semantic_relation_inference_performed: false
source_text_returned: false
```

## 7. Extraction evaluation

`tools/evidence_graph_claim_evaluation.py` performs deterministic one-to-one matching.

A benchmark match requires either:

- exact evidence digest; or
- both configured minimum span intersection-over-union and minimum claim token-F1.

The following metrics are reported:

- extraction precision, recall and F1;
- exact evidence-digest accuracy;
- exact locator accuracy;
- mean span intersection-over-union;
- mean claim token-F1;
- claim-type accuracy;
- modality accuracy;
- confidence Brier score.

Claim type and modality are evaluated separately from detection matching.

This evaluator does not perform semantic entailment. A lexical/provenance benchmark match is not proof that a claim is scientifically supported.

Run a strict local fixture:

```bash
python scripts/evidence_graph_claim_evaluation.py path/to/fixture.json
```

The fixture reader uses descriptor-based no-follow reads, bounded size, duplicate-key refusal, NaN/Infinity refusal and exact dataclass reconstruction. Output includes only metrics, IDs and digests. Claim and evidence text are never echoed.

The report digest is independently revalidated by `verify_scientific_claim_evaluation_report` and commits the configured matching thresholds.

## 8. Current verification boundary

Executed in a reconstructed focused workspace using the committed claim contracts, extraction, store, review, runtime and CLI implementations with API-faithful stubs only for older repository boundaries:

```text
8 passed
```

Focused compilation passed.

Additional repository-native contracts now cover:

- topological correction submission;
- strict extraction schema and locator validation;
- governed review/correction lineage;
- policy, actor and database tamper refusal;
- operator privacy;
- extraction evaluation and report integrity;
- descriptor-safe evaluation fixture execution.

Those additional repository-native contracts have not yet been executed together in a fresh exact-current complete checkout.

## Permanent boundaries

- Extractor output is a proposal, not a graph fact.
- Extractor confidence is not scientific certainty.
- Human approval records a governed review decision; it is not independent replication.
- A claim annotation does not imply support, contradiction, causality or truth.
- Correction lineage never rewrites or deletes the earlier proposal or decision.
- Evaluation token-F1 and span overlap are not semantic entailment.
- No automatic graph publication is performed by this workflow.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
