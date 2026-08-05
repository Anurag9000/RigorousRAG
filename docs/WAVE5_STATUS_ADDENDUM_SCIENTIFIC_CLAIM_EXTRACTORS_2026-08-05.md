# Wave 5 status addendum — governed scientific claim extractors

Last updated: 2026-08-05

This addendum extends `WAVE5_STATUS_ADDENDUM_SCIENTIFIC_CLAIMS_2026-08-05.md` with exact-version extractor governance.

## Implemented

### Immutable exact-version registry

- [x] Owner-scoped extractor name/version identity.
- [x] Extractor kinds: `model` and `rule`.
- [x] Implementation SHA-256.
- [x] Configuration SHA-256.
- [x] Repository-owned closed output-schema SHA-256.
- [x] Supported claim-type capability set.
- [x] Supported modality capability set.
- [x] Supported language capability set.
- [x] Process actor and binding provenance.
- [x] Deterministic record digest.
- [x] Strict SQLite payload/column/path/database identity validation.
- [x] Idempotent exact registration replay.
- [x] Different-scope version collision refusal.
- [x] Monotonic retirement.
- [x] No reactivation of retired versions.
- [x] Exact record-digest retirement confirmation.

### Governance

- [x] Separate extractor-administrator policy.
- [x] Owner, extractor-name and action scopes.
- [x] Optional policy expiry.
- [x] Strict descriptor-based policy file loading.
- [x] Duplicate-key and NaN/Infinity refusal.
- [x] Existing process-owned/signed actor boundary.
- [x] Separately configurable extractor administrator and claim reviewer roles.
- [x] Fail-closed absent or multiply configured policy source.

### Registered execution

- [x] Exact active owner/name/version requirement.
- [x] Registered language enforcement.
- [x] Registered claim-type enforcement.
- [x] Registered modality enforcement.
- [x] Correct model/rule proposer-kind provenance.
- [x] Registry record digest embedded in each resulting proposal.
- [x] Implementation/configuration/schema digests embedded in each resulting proposal.
- [x] Normalized language embedded in each resulting proposal.
- [x] Retired-version execution refusal.
- [x] Compatibility function delegated to the same canonical execution path.
- [x] No credentials, prompts, model responses or source text stored in the registry.

### Operator surface

```bash
python scripts/evidence_graph_claim_extractors.py register ...
python scripts/evidence_graph_claim_extractors.py status ...
python scripts/evidence_graph_claim_extractors.py list ...
python scripts/evidence_graph_claim_extractors.py retire ...
```

Status/list are domain-read-only. Registration and retirement require the configured actor and extractor-administrator policy. Retirement additionally requires the exact current record digest.

## Configuration

```dotenv
EVIDENCE_GRAPH_CLAIM_EXTRACTOR_REGISTRY_DB_PATH=data/evidence_graph_claim_extractors.sqlite3
EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_PATH=
# EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_PATH=config/evidence_graph_claim_extractor_policy.example.json
# EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_JSON={"schema_version":1,"administrators":[...]}
```

The example policy contains placeholders and is inactive by default.

## Repository-native contracts

Ten new extractor-governance tests are committed:

- deterministic registration and replay;
- policy expiry and administrator identity;
- monotonic retirement and no reactivation;
- governed rule-extractor provenance;
- language, taxonomy and schema capability refusal;
- registry payload and file-identity tamper refusal;
- secret-free register/status/list output;
- exact retirement confirmation;
- generic actor/policy failure output;
- canonical runtime cache scoping.

Combined scientific-claim repository-native test count after this addendum:

```text
36 tests
```

Breakdown:

- extraction: 5;
- governed review/storage/corrections: 7;
- evaluation: 5;
- evaluation-report verification: 2;
- claim runtime/operator privacy: 4;
- evaluation fixture CLI: 3;
- extractor registry and registered execution: 6;
- extractor registry runtime/CLI: 4.

## Executed evidence

The earlier reconstructed claim core/operator harness still passes:

```text
8 passed
```

It covers the claim contracts, extraction, immutable review store, governed review, correction conversion, runtime and claim CLI. It does not include the later evaluation fixture or extractor-registry modules.

The 36 repository-native scientific-claim tests have not yet been executed together from a fresh exact-current complete checkout.

## Still open

- [ ] Complete exact-current pytest, coverage, Ruff and full-tree compilation.
- [ ] Execute all 36 scientific-claim tests together.
- [ ] Windows and Docker/Compose registry persistence tests.
- [ ] Independent-process registration/retirement contention.
- [ ] SQLite busy/locked, WAL, I/O-error and disk-full injection.
- [ ] Process-kill testing around registration and retirement.
- [ ] Actual production model/rule extractor implementations.
- [ ] Governed benchmark promotion and rollback reports.
- [ ] Active-version promotion pointer while preserving exact-version execution.
- [ ] Deprecation reasons and compatibility windows.
- [ ] External IAM or hardware-backed administrator identity.
- [ ] Signed registry export and transparency log.

## Permanent non-claims

- Registration is not benchmark promotion.
- A digest identifies bytes/configuration but does not prove scientific quality or safety.
- A registered extractor still produces proposals requiring governed human review.
- Retirement does not delete historical proposals or decisions.
- No model credential, prompt, response or source text is stored.
- No automatic graph publication or semantic relation inference is performed.
- No exact-current full-suite or CI success is claimed.
- Release readiness is not claimed.
