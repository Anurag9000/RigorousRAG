# Governed scientific claim extractor registry

Last updated: 2026-08-05

## Purpose

The reviewed scientific-claim adapter records extractor name and version in every proposal. The governed extractor registry makes those identifiers meaningful by binding an exact owner-scoped version to immutable implementation, configuration and output-schema digests plus declared capabilities.

Registration does not store:

- API keys or access tokens;
- endpoint URLs containing credentials;
- model prompts;
- model responses;
- source or claim text;
- executable binaries;
- arbitrary configuration payloads.

It stores only identity, state, capability values and SHA-256 digests.

## Configuration

```dotenv
EVIDENCE_GRAPH_CLAIM_EXTRACTOR_REGISTRY_DB_PATH=data/evidence_graph_claim_extractors.sqlite3
```

Configure exactly one administrator-policy source:

```dotenv
EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_PATH=
# EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_PATH=config/evidence_graph_claim_extractor_policy.example.json
# EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_JSON={"schema_version":1,"administrators":[...]}
```

Both sources are empty by default. The example policy contains placeholders and must not be activated until every placeholder is replaced.

Administrative commands use the existing process-owned or signed reviewer actor boundary:

```dotenv
EVIDENCE_GRAPH_REVIEW_ACTOR_ID=extractor-administrator-1
```

The extractor policy is separate from the claim reviewer policy. This allows extractor administration and scientific claim review to be assigned to different governed roles.

## Registry record

An immutable active extractor version binds:

- owner ID;
- extractor name;
- extractor version;
- extractor kind: `model` or `rule`;
- implementation SHA-256;
- configuration SHA-256;
- the repository-owned closed claim-output schema SHA-256;
- supported claim types;
- supported modalities;
- supported languages;
- registering actor identity, binding method and binding digest;
- registration timestamp;
- deterministic record digest.

The output-schema digest is not supplied by operators. It must equal the repository-owned scientific-claim output schema.

Registration is idempotent only when the exact version has the same immutable registration scope. A version registered differently is refused.

A retired version cannot be reactivated. A changed implementation or configuration requires a new version.

## Administrator policy

Each administrator grant scopes:

- administrator ID;
- owners;
- extractor names;
- allowed actions: `register` and/or `retire`;
- optional expiry.

The policy loader uses a strict schema, bounded descriptor-based file reads, duplicate-key refusal and NaN/Infinity refusal.

## Register an extractor version

```bash
python scripts/evidence_graph_claim_extractors.py register \
  --owner-id alice \
  --extractor-name scientific-claims \
  --extractor-version 1.0.0 \
  --extractor-kind model \
  --implementation-sha256 IMPLEMENTATION_SHA256 \
  --configuration-sha256 CONFIGURATION_SHA256 \
  --claim-type finding \
  --claim-type limitation \
  --modality asserted \
  --modality uncertain \
  --language en
```

Registration requires:

- one configured process-owned actor;
- an active policy grant for the actor, owner, extractor name and `register` action;
- valid exact digests;
- bounded capability arrays;
- supported claim types and modalities.

## Inspect registered versions

```bash
python scripts/evidence_graph_claim_extractors.py status \
  --owner-id alice \
  --extractor-name scientific-claims \
  --extractor-version 1.0.0
```

```bash
python scripts/evidence_graph_claim_extractors.py list \
  --owner-id alice \
  --extractor-name scientific-claims \
  --state active
```

Output contains only registry fields and explicitly reports that credentials, prompts, model responses and source text are absent.

## Retire a version

First inspect the current record digest. Then retire with exact confirmation:

```bash
python scripts/evidence_graph_claim_extractors.py retire \
  --owner-id alice \
  --extractor-name scientific-claims \
  --extractor-version 1.0.0 \
  --confirm-record-digest CURRENT_RECORD_DIGEST
```

Retirement is monotonic and records:

- retiring actor ID;
- binding method;
- binding digest;
- retirement timestamp;
- new deterministic record digest.

A retired version is refused by the registered extraction boundary.

## Canonical registered extraction

```python
from tools.evidence_graph_claim_registered_extraction import (
    extract_governed_scientific_claim_proposals,
)
from tools.evidence_graph_claim_extractor_runtime import (
    get_scientific_claim_extractor_registry,
)

batch = extract_governed_scientific_claim_proposals(
    finalized_document,
    extractor_output,
    owner_id="alice",
    generation=authoritative_generation,
    profile_fingerprint=profile_fingerprint,
    proposer_id="scientific-claim-extractor-runtime",
    extractor_name="scientific-claims",
    extractor_version="1.0.0",
    language="en",
    registry=get_scientific_claim_extractor_registry(),
)
```

The boundary requires the exact active registry record and checks:

- requested owner/name/version;
- registered language;
- emitted claim types;
- emitted modalities;
- model/rule proposer kind.

Every resulting proposal additionally commits:

- registry record digest;
- implementation digest;
- configuration digest;
- output-schema digest;
- normalized language.

For rule extractors, the canonical boundary reconstructs deterministic proposals with `proposer_kind="rule"`; it never falsely labels a rule result as model output.

The compatibility function `extract_registered_scientific_claim_proposals` delegates to this same canonical boundary.

## Current verification boundary

Repository-native contracts cover:

- deterministic registration and idempotent replay;
- policy expiry and administrator identity;
- monotonic retirement and no reactivation;
- rule-extractor provenance;
- language, claim-type, modality and schema capability refusal;
- database payload and file-identity tampering;
- secret-free register/status/list output;
- exact record-digest retirement confirmation;
- generic failure output;
- canonical runtime caching.

These new registry contracts have not yet been executed in a fresh exact-current complete checkout.

## Permanent boundaries

- A registered extractor is not automatically trusted for scientific correctness.
- Implementation and configuration digests identify bytes/configuration; they do not prove safety or quality.
- Registration is not benchmark promotion.
- Retirement prevents future registered execution; it does not delete historical proposals.
- The registry does not store credentials, prompts, responses or source text.
- Extractor output remains a proposal requiring governed human review.
- No automatic graph publication or semantic relation inference is performed.
- Release readiness is not claimed.
