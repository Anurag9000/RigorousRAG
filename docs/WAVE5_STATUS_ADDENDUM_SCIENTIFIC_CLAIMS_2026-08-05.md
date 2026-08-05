# Wave 5 status addendum — reviewed scientific claims

Last updated: 2026-08-05

This addendum records the implementation and verification boundary for the reviewed scientific-claim extraction foundation.

## Implemented

### Closed-schema extraction

- [x] Strict top-level and per-claim schemas.
- [x] Duplicate-key and non-finite-number refusal.
- [x] Bounded scientific claim type and modality vocabularies.
- [x] Exact finalized document content-hash validation.
- [x] Exact section, page and character-span validation.
- [x] Server-computed evidence-span SHA-256.
- [x] Generation and profile-fingerprint binding.
- [x] Deterministic proposal identities excluding creation timestamps.
- [x] Extractor name/version and proposer identity.
- [x] Confidence and correction-predecessor binding.
- [x] No persistence or graph mutation during extraction.

### Immutable proposal and correction storage

- [x] Dedicated path-safe SQLite review database.
- [x] Immutable proposal rows with strict payload reconstruction.
- [x] Atomic bounded batch submission.
- [x] Canonical predecessor-first submission independent of caller order.
- [x] Same-generation/content/profile correction scope.
- [x] One correction successor per proposal.
- [x] Correction branch refusal.
- [x] Database, parent, row-column, JSON and digest tamper refusal.

### Governed terminal review

- [x] Dedicated owner/document/decision-scoped claim policy.
- [x] Policy expiry and fail-closed source selection.
- [x] Existing process-owned or signed reviewer actor boundary.
- [x] Proposer–reviewer separation.
- [x] Correction-author–reviewer separation.
- [x] Exact replacement-scope validation.
- [x] Atomic decision and authorization insertion.
- [x] Deterministic authorization receipt.
- [x] Stable exact replay preserving original timestamps.
- [x] Corrected claim approval only after exact predecessor supersession.

### Explicit graph annotation conversion

- [x] Only exact approved and authorization-backed proposals convert.
- [x] Exact owner/document/generation/content/profile revalidation.
- [x] Correction-predecessor and obsolete-successor checks.
- [x] Existing `GraphAnnotation(node_type="claim")` output.
- [x] Proposal, decision, authorization, policy, grant, locator and extractor provenance.
- [x] No graph mutation during conversion.
- [x] No support, contradiction, citation, method, dataset, derivation or equivalence inference.

### Operator and evaluation surfaces

- [x] Text-free status and bounded listing.
- [x] Actor-bound terminal decision command.
- [x] Text-free approved-annotation inspection.
- [x] Claim/evidence text hashes and lengths instead of raw text.
- [x] Extraction precision, recall and F1.
- [x] Exact evidence-digest and locator accuracy.
- [x] Span-IoU and claim token-F1.
- [x] Claim-type and modality accuracy.
- [x] Confidence Brier score.
- [x] Deterministic one-to-one matching.
- [x] Text-free report digests and independent reconstruction verification.
- [x] Descriptor-safe strict local fixture CLI.

## Configuration

```dotenv
EVIDENCE_GRAPH_CLAIM_REVIEW_DB_PATH=data/evidence_graph_claim_reviews.sqlite3
EVIDENCE_GRAPH_CLAIM_REVIEW_POLICY_PATH=
# EVIDENCE_GRAPH_CLAIM_REVIEW_POLICY_PATH=config/evidence_graph_claim_review_policy.example.json
# EVIDENCE_GRAPH_CLAIM_REVIEW_POLICY_JSON={"schema_version":1,"reviewers":[...]}
```

The example policy contains placeholders. No claim-review policy is selected by default.

Reviewer identity continues to use exactly one existing process actor source:

```dotenv
EVIDENCE_GRAPH_REVIEW_ACTOR_ID=
EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH=
EVIDENCE_GRAPH_REVIEW_ACTOR_ASSERTION_PATH=
```

## Operator commands

```bash
python scripts/evidence_graph_claims.py status PROPOSAL_ID
```

```bash
python scripts/evidence_graph_claims.py list \
  --owner-id OWNER \
  --doc-id DOCUMENT_ID \
  --generation GENERATION \
  --decision pending
```

```bash
python scripts/evidence_graph_claims.py decide PROPOSAL_ID \
  --owner-id OWNER \
  --decision approved \
  --reviewer-id REVIEWER \
  --reason-code REASON
```

```bash
python scripts/evidence_graph_claims.py annotations \
  --owner-id OWNER \
  --doc-id DOCUMENT_ID \
  --generation GENERATION \
  --content-sha256 CONTENT_SHA256 \
  --profile-fingerprint PROFILE_FINGERPRINT \
  --proposal-id APPROVED_PROPOSAL_ID
```

```bash
python scripts/evidence_graph_claim_evaluation.py FIXTURE.json
```

## Executed focused verification

A reconstructed focused workspace executed the committed claim contracts, extraction, SQLite store, governed review, runtime and operator modules. Older repository dependencies were represented by API-faithful stubs only for:

- owner normalization;
- existing `GraphAnnotation` construction;
- existing process-owned reviewer actor binding.

Executed result:

```text
8 passed
```

Focused Python compilation passed.

The executed checks cover:

1. exact locator/evidence validation;
2. closed-schema refusal;
3. deterministic proposal identity;
4. atomic governed review;
5. proposer–reviewer separation;
6. correction supersession and approved conversion;
7. timestamp-stable exact replay;
8. privacy-safe runtime/CLI behavior.

This reconstructed result predates the later evaluation fixture and expanded repository-native tests and is not a complete exact-current repository run.

## Committed repository-native contracts

The repository now contains **26 scientific-claim tests**:

- extraction contracts: 5;
- governed review, correction and storage contracts: 7;
- extraction evaluation contracts: 5;
- evaluation report-integrity contracts: 2;
- runtime/operator privacy contracts: 4;
- strict evaluation fixture CLI contracts: 3.

These tests include reverse-order correction submission, taxonomy drift, locator mismatch, policy expiry, actor mismatch, correction branching, database tampering, pending/rejected conversion refusal, one-to-one evaluation, report-digest tampering, duplicate fixture keys, non-finite fixtures and path redirects.

The 26 repository-native tests have not yet been executed together from a fresh exact-current complete checkout.

## Live repository verification boundary

At the time this addendum was prepared:

- all scientific-claim implementation and documentation updates were committed directly to `main`;
- no development branch or pull request was intentionally created;
- no force push or history rewrite was performed.

A final post-addendum head/topology/status audit remains required and must not be inferred from earlier audits.

## Still open

- [ ] Complete exact-current repository pytest and coverage.
- [ ] Ruff and full-tree compilation.
- [ ] Windows, Docker/Compose and restart matrices.
- [ ] Independent-process claim proposal/review contention.
- [ ] SQLite busy/locked, WAL, I/O-error and disk-full fault injection.
- [ ] Process-kill testing around proposal and atomic decision/authorization transactions.
- [ ] Production model/rule extractor adapters and governed extractor registry.
- [ ] Reviewed entity, citation, method and dataset extraction proposals.
- [ ] Semantic claim-support and entailment metrics.
- [ ] Explicit support/contradiction proposal and adjudication workflows.
- [ ] Inter-reviewer agreement reports and quorum review.
- [ ] Connected-corpus claim extraction benchmarks and dataset cards.
- [ ] Coordinated authoritative-generation graph-job integration.

## Permanent non-claims

- Extractor output is a proposal, not a graph fact.
- Extractor confidence is not scientific certainty.
- Human approval is a governed review record, not independent replication.
- A claim annotation does not imply truth, support, contradiction or causality.
- Span overlap and token-F1 are not semantic entailment.
- No automatic graph publication is performed by this workflow.
- No exact-current CI or full-suite success is claimed.
- Release readiness is not claimed.
