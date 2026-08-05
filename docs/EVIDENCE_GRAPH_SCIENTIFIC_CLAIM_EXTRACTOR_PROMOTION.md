# Governed scientific claim extractor benchmark promotion

Last updated: 2026-08-05

## Purpose

The exact-version extractor registry identifies governed implementation and configuration bytes. The promotion layer adds measured, policy-gated selection of one active exact version for an owner/extractor name.

Promotion does not modify extractor records, claim proposals, review decisions, evidence graphs or source documents. It stores immutable assessment reports, append-only activation events and one optimistic current pointer.

## Configuration

```dotenv
EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_DB_PATH=data/evidence_graph_claim_extractor_promotions.sqlite3
EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_PATH=
# EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_PATH=config/evidence_graph_claim_extractor_promotion_policy.example.json
# EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_JSON={"schema_version":1,"thresholds":{...},"administrators":[...]}
```

Both policy sources are empty by default. The example policy contains placeholder identities and example metric floors. Review every threshold and replace every placeholder before activation.

The promotion database must remain separate from the immutable extractor registry. The runtime rejects canonical-path equality and existing hard-link aliasing.

Promotion administrators use the existing process-owned or signed actor boundary. The promotion policy is separate from extractor-registration and scientific-claim-review policies.

## 1. Build text-free benchmark cases

A promotion benchmark case is built from:

- a verified `ScientificClaimEvaluationReport`;
- the complete proposal set represented by the report;
- one exact `ScientificClaimExtractorRecord`;
- a unique case ID;
- a dataset digest.

```python
from tools.evidence_graph_claim_extractor_benchmark import (
    build_scientific_claim_extractor_benchmark_case,
)

case = build_scientific_claim_extractor_benchmark_case(
    case_id="dataset-a-fold-1",
    dataset_digest=dataset_sha256,
    evaluation_report=verified_report,
    proposals=proposals,
    extractor_record=registered_record,
    minimum_span_iou=0.5,
    minimum_claim_token_f1=0.5,
)
```

The builder requires:

- report-digest verification under the declared matching thresholds;
- exact owner scope;
- exact proposal count and proposal IDs;
- the exact extractor registry record digest in every proposal;
- exact extractor name/version provenance.

The case stores only IDs, digests, counts and metrics. It does not store claim or evidence text.

## 2. Aggregate a benchmark suite

```python
from tools.evidence_graph_claim_extractor_benchmark import (
    aggregate_scientific_claim_extractor_benchmark,
)

suite = aggregate_scientific_claim_extractor_benchmark(
    benchmark_id="scientific-claims-v1",
    cases=cases,
)
```

All cases must have unique case IDs and dataset digests and must bind one exact owner/name/version/record digest.

Suite aggregation uses:

- micro precision, recall and F1 from total gold/proposal/matched counts;
- matched-count weighting for evidence, locator, span, lexical, type and modality quality;
- proposal-count weighting for confidence Brier score.

The suite is deterministic and digest-bound. Input case order does not change the suite digest.

## 3. Promotion policy

The policy defines:

- minimum case count;
- minimum gold count;
- minimum precision, recall and F1;
- minimum exact evidence accuracy;
- minimum exact locator accuracy;
- minimum mean span IoU;
- minimum mean claim token-F1;
- minimum claim-type accuracy;
- minimum modality accuracy;
- maximum confidence Brier score;
- administrator owner/extractor/action scopes and optional expiry.

Allowed actions are:

```text
promote
rollback
```

An assessment records every failed floor/ceiling as a deterministic reason. Ineligible reports are retained but cannot move the current pointer.

## 4. Serialize a suite for operator use

Write the exact `dataclasses.asdict(suite)` JSON to a local protected file. The operator reader:

- uses descriptor-based no-follow reads;
- enforces a bounded file size;
- rejects duplicate keys and NaN/Infinity;
- reconstructs every case and the suite;
- revalidates every case and suite digest;
- rejects redirects and identity changes during the read.

The serialized suite contains no claim or evidence text.

## 5. Assess without mutation

```bash
python scripts/evidence_graph_claim_extractor_promotions.py assess SUITE.json
```

The command requires an exact active registry record and the configured policy. It does not store a report or move the pointer.

Output contains:

- exact owner/name/version/record digest;
- benchmark and suite digests;
- policy and threshold digests;
- eligibility and deterministic reasons;
- assessment timestamp and report digest;
- explicit text-free/non-mutating flags.

## 6. Promote an exact version

First activation:

```bash
python scripts/evidence_graph_claim_extractor_promotions.py promote SUITE.json
```

Subsequent activation:

```bash
python scripts/evidence_graph_claim_extractor_promotions.py promote SUITE.json \
  --expected-current-activation-id CURRENT_ACTIVATION_ID
```

Promotion requires:

- process-owned promotion-administrator identity;
- policy authorization for owner/extractor/action;
- active exact registry record;
- suite/record identity equality;
- all policy gates passing;
- exact expected current activation.

The transaction stores an immutable activation event and updates the current pointer. A stale expected pointer fails closed.

Assessment/report identity excludes assessment time. Exact report replay preserves the first stored assessment timestamp.

Activation identity excludes activation time. Exact replay after activation-row or pointer commit recognizes the same previous pointer, report, action and actor binding and preserves the first activation timestamp.

## 7. Inspect and resolve the current version

```bash
python scripts/evidence_graph_claim_extractor_promotions.py current \
  --owner-id alice \
  --extractor-name scientific-claims
```

```bash
python scripts/evidence_graph_claim_extractor_promotions.py history \
  --owner-id alice \
  --extractor-name scientific-claims
```

```bash
python scripts/evidence_graph_claim_extractor_promotions.py resolve \
  --owner-id alice \
  --extractor-name scientific-claims
```

Current resolution revalidates:

- current activation identity;
- exact active registry version;
- registry record digest;
- eligible promotion report and report/record identity.

Retiring the current exact version makes resolution fail closed until another eligible active version is promoted or rolled back.

## 8. Append-only rollback

Rollback does not delete or rewrite later history. It creates a new activation event pointing to a previously eligible exact version.

```bash
python scripts/evidence_graph_claim_extractor_promotions.py rollback \
  --target-promotion-report-digest EARLIER_ELIGIBLE_REPORT_DIGEST \
  --expected-current-activation-id CURRENT_ACTIVATION_ID
```

Rollback requires:

- process-owned promotion-administrator identity;
- rollback permission;
- exact current activation confirmation;
- an earlier stored eligible report;
- the target exact registry record still being active.

A retired target is refused.

## Privacy and integrity

Promotion storage and operator output contain no:

- claim text;
- evidence text;
- document text;
- source paths;
- extractor prompts or responses;
- model credentials;
- executable implementation bytes.

The canonical transactional store rejects:

- duplicate JSON keys;
- non-finite stored values;
- payload/column divergence;
- database/path identity replacement;
- report-digest collision;
- activation-identity collision;
- unexpected pointer movement.

## Verification boundary

Repository-native contracts cover:

- benchmark case proposal/record binding;
- deterministic aggregate metrics;
- duplicate dataset and cross-version refusal;
- evaluation/proposal identity drift;
- threshold reasons and ineligible non-activation;
- first promotion and timestamp-stable exact replay;
- optimistic pointer refusal;
- exact-version upgrade;
- append-only rollback;
- retired-target refusal;
- stored-row tampering;
- descriptor-safe CLI suites;
- current/history/resolve privacy;
- stale rollback confirmation and generic policy/tamper failures.

These promotion contracts have not yet been executed in a fresh exact-current complete checkout.

## Permanent boundaries

- Promotion is evidence that one benchmark suite satisfies one configured policy.
- Promotion is not proof of general scientific correctness, robustness or safety.
- Promotion does not independently validate the benchmark dataset or gold annotations.
- Lexical/span claim metrics are not semantic entailment.
- A current pointer does not bypass exact-version registry and active-state validation.
- Rollback does not reactivate a retired version.
- Promotion never approves claim proposals or publishes graph nodes.
- No automatic provider execution is performed.
- Release readiness is not claimed.
