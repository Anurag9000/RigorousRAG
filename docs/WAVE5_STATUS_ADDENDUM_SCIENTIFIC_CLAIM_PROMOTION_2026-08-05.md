# Wave 5 status addendum — scientific claim extractor benchmark promotion

Last updated: 2026-08-05

This addendum extends the reviewed scientific-claim and governed extractor-registry status ledgers with policy-gated benchmark promotion and append-only rollback.

## Implemented benchmark contract

- [x] Verified evaluation-report input.
- [x] Complete evaluated proposal identity reconciliation.
- [x] Exact proposal-to-registry-record digest binding.
- [x] Exact owner/name/version scope.
- [x] Unique case IDs and dataset digests.
- [x] Deterministic case and suite digests.
- [x] Input-order-independent suite aggregation.
- [x] Micro precision, recall and F1.
- [x] Matched-count-weighted evidence, locator, span, lexical, type and modality quality.
- [x] Proposal-count-weighted confidence Brier score.
- [x] No claim or evidence text in cases or suites.

## Implemented promotion governance

- [x] Separate promotion database and policy.
- [x] Minimum case/gold count floors.
- [x] Precision, recall and F1 floors.
- [x] Evidence, locator, span, lexical, type and modality floors.
- [x] Confidence Brier score ceiling.
- [x] Owner/extractor/action-scoped promotion administrator grants.
- [x] Optional policy expiry.
- [x] Strict descriptor-based policy loading.
- [x] Duplicate-key and NaN/Infinity refusal.
- [x] Process-owned/signed actor binding.
- [x] Deterministic failure reasons.
- [x] Immutable eligible and ineligible assessment reports.
- [x] Ineligible report retention without pointer activation.

## Implemented activation and rollback

- [x] Append-only activation history.
- [x] One current pointer per owner/extractor name.
- [x] Exact expected-current activation confirmation.
- [x] Exact active registry-record revalidation.
- [x] Eligible promotion-report revalidation.
- [x] Timestamp-stable report replay.
- [x] Timestamp-stable activation replay.
- [x] Recovery after activation-row insertion before pointer publication.
- [x] Recovery after pointer publication before caller acknowledgement.
- [x] Stale pointer refusal.
- [x] Current exact-version resolution.
- [x] Append-only rollback activation.
- [x] Rollback only to an earlier eligible active exact version.
- [x] Retired target refusal.
- [x] No version reactivation or mutable history rewrite.

## Implemented operator surface

```bash
python scripts/evidence_graph_claim_extractor_promotions.py assess SUITE.json
python scripts/evidence_graph_claim_extractor_promotions.py promote SUITE.json
python scripts/evidence_graph_claim_extractor_promotions.py current ...
python scripts/evidence_graph_claim_extractor_promotions.py history ...
python scripts/evidence_graph_claim_extractor_promotions.py resolve ...
python scripts/evidence_graph_claim_extractor_promotions.py rollback ...
```

The suite reader is descriptor-safe, bounded, no-follow and digest-validating. Outputs contain only IDs, digests, counts, metrics, reasons and activation provenance.

## Configuration

```dotenv
EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_DB_PATH=data/evidence_graph_claim_extractor_promotions.sqlite3
EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_PATH=
# EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_PATH=config/evidence_graph_claim_extractor_promotion_policy.example.json
# EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_JSON={"schema_version":1,"thresholds":{...},"administrators":[...]}
```

The policy example contains placeholder identities and example thresholds and remains inactive by default.

## Repository-native test inventory

The scientific-claim family now contains **47 committed tests**:

- extraction: 5;
- governed review/storage/corrections: 7;
- extraction evaluation: 5;
- evaluation-report verification: 2;
- claim runtime/operator privacy: 4;
- evaluation fixture CLI: 3;
- exact-version extractor registry/registered execution: 6;
- extractor registry runtime/CLI: 4;
- benchmark case/suite aggregation: 4;
- promotion assessment/activation/rollback: 4;
- promotion runtime/CLI: 3.

Promotion tests cover:

- proposal/record digest binding;
- deterministic aggregation;
- duplicate dataset and cross-version refusal;
- evaluation/proposal identity drift;
- all threshold failure reasons;
- ineligible non-activation;
- first activation and exact replay;
- optimistic pointer refusal;
- exact-version upgrade;
- append-only rollback;
- retired rollback target refusal;
- stored-row tampering;
- descriptor-safe suite loading;
- text-free assessment/current/history/resolve;
- stale rollback confirmation;
- generic policy and suite-tamper failure output.

## Executed verification boundary

The earlier reconstructed reviewed-claim core/operator workspace still passes:

```text
8 passed
```

It covers claim contracts, extraction, immutable review storage, governed decisions, correction conversion, runtime and claim CLI. It does not include later evaluation fixture, exact-version registry, benchmark or promotion modules.

The 47 repository-native scientific-claim tests have not been executed together from a fresh exact-current complete checkout.

## Still open

- [ ] Complete exact-current pytest and coverage.
- [ ] Ruff and full-tree compilation.
- [ ] Execute all 47 scientific-claim tests together.
- [ ] Windows and Docker/Compose persistence/restart matrices.
- [ ] Independent-process registration, promotion, rollback and retirement contention.
- [ ] Process-kill testing around assessment/report/activation transactions.
- [ ] SQLite busy/locked, WAL, I/O-error and disk-full injection.
- [ ] Production model/rule extractor implementations.
- [ ] Governed benchmark dataset manifests and gold-review provenance.
- [ ] Historical-baseline non-inferiority confidence intervals.
- [ ] Deprecation reasons and compatibility windows.
- [ ] Signed registry/benchmark/promotion exports and transparency logs.
- [ ] Semantic claim-support and entailment evaluation.
- [ ] Explicit support/contradiction adjudication.

## Permanent non-claims

- Registration is not benchmark promotion.
- Promotion means one benchmark suite passed one configured policy.
- Promotion is not proof of general scientific correctness, safety or robustness.
- The promotion layer does not validate dataset or gold-annotation quality independently.
- Lexical/span metrics are not semantic entailment.
- Current-version selection never bypasses exact active registry validation.
- Rollback never reactivates retired versions or rewrites history.
- Promotion does not approve claim proposals or publish graph nodes.
- No exact-current CI or full-suite success is claimed.
- Release readiness is not claimed.
