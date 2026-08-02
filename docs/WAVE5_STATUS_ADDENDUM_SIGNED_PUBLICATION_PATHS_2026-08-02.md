# Wave 5 status addendum — signed actor-use publication paths

Last updated: 2026-08-02

This addendum supersedes the “not yet canonical” operator-path section in `WAVE5_STATUS_ADDENDUM_SIGNED_REVIEW_ACTOR_2026-08-02.md` by introducing dedicated signed-provenance publication commands. The historical authorization-only commands remain available for compatibility and direct-actor deployments.

## 1. Immediate signed publication

Implemented:

```bash
python scripts/evidence_graph_set_signed_publish.py publish-approved ...
```

Properties:

- [x] Requires committed reviewer-authorization receipts.
- [x] Loads the durable signed actor-use store.
- [x] Validates actor-use provenance before graph-set construction.
- [x] Preserves compare-and-swap pointer expectations.
- [x] Preserves post-activation authority verification.
- [x] Preserves first-publication clearing and replacement restoration compensation.
- [x] Reports signed provenance validation explicitly.
- [x] Returns no source text, assertion body, signature or key.

## 2. Durable signed publication

Implemented:

```bash
python scripts/evidence_graph_set_signed_publication.py seed ...
python scripts/evidence_graph_set_signed_publication.py status ...
python scripts/evidence_graph_set_signed_publication.py list ...
python scripts/evidence_graph_set_signed_publication.py execute ...
python scripts/evidence_graph_set_signed_publication.py reconcile-one ...
python scripts/evidence_graph_set_signed_publication.py retry ...
python scripts/evidence_graph_set_signed_publication.py cancel ...
```

Properties:

- [x] Seed validates authorization and signed actor-use provenance before journaling.
- [x] Uses the established deterministic logical publication operation identity.
- [x] Uses an isolated signed-only publication phase journal.
- [x] Rejects canonical path equality with the authorization-only journal.
- [x] Rejects pre-existing hard-link aliases to the authorization-only journal.
- [x] Execute reconstructs signed provenance before publication mutation.
- [x] Reconcile captures one finite timestamp for claim discovery and execution.
- [x] Retry and cancel retain exact operation-ID confirmation.
- [x] Status and list remain read-only.
- [x] Crash-recovery pointer authority and compensation remain delegated to the existing recovery engine.
- [x] Outputs remain source-text-free and secret-free.

## 3. Publication provenance contract

For each relation proposal, the signed publication ledger:

- resolves the committed governed authorization receipt;
- reads actor-use records by deterministic decision ID;
- requires every returned actor-use record to be committed;
- revalidates decision, proposal, owner, graph-set key, decision type and reviewer identity;
- sorts uses deterministically by assertion digest;
- emits a deterministic aggregate actor-use digest and count;
- emits a deterministic zero-use digest for direct actor decisions;
- never mutates the original proposal or proposal ID.

## 4. Assurance-level recovery isolation

Authorization-only commands remain:

```bash
python scripts/evidence_graph_set_publish.py ...
python scripts/evidence_graph_set_publication.py ...
```

They continue to require committed governed authorization receipts. They do not include the additional actor-use aggregate in relation metadata.

Signed assertion deployments should use only the dedicated signed commands when publication-level signed actor provenance is required.

The command families share proposal, authorization, actor-use, graph-set and pointer stores, but they do **not** share phase journals:

```bash
EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH=data/evidence_graph_set_publications.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH=data/evidence_graph_set_signed_publications.sqlite3
```

This separation prevents a signed command from recovering a `candidate_stored` phase created by the authorization-only path, which would otherwise bypass signed metadata construction.

Pre-isolation non-terminal attempts in the authorization-only journal are not automatically migrated. They must be inspected and safely cancelled through the authorization-only command family, then re-seeded through the signed command with an explicit pointer expectation. No destructive cleanup is automatic.

## 5. Committed focused contracts

Contracts now cover:

- zero-use direct actor metadata;
- deterministic committed multi-use aggregation;
- reserved actor-use refusal;
- reviewer/scope mismatch refusal;
- immediate signed publication delegation;
- durable signed execution delegation;
- one finite reconcile timestamp;
- invalid timestamp refusal;
- signed seed dependency validation;
- signed execute dependency injection;
- idle reconcile response;
- secret-free compensation output;
- production dataclass serialization in the durable CLI test;
- distinct default publication-journal paths;
- signed journal override;
- canonical path alias refusal;
- hard-link alias refusal.

## 6. Verification boundary

Executed in reconstructed focused workspaces using the live signed modules and minimal stubs only for unrelated repository services:

- **12/12** signed assertion, actor-binding and actor-use runtime checks passed;
- **17/17** signed publication adapter, timestamp boundary, immediate CLI, durable CLI and isolation tests passed;
- Python compilation passed for both focused slices.

The first signed-publication execution exposed two test-only defects, both corrected on `main`:

1. a `SimpleNamespace` fixture was incompatible with production `dataclasses.asdict` serialization;
2. a `setdefault(...) or operation_id` lambda returned the stored dictionary instead of the intended operation ID.

No production failure was hidden by those fixes.

Not yet executed in an exact-current full repository checkout:

- complete repository pytest;
- coverage and Ruff;
- Windows and container matrices;
- process-kill, disk-full and SQLite write-failure publication injection;
- real multi-process signed/authorization-only contention.

GitHub exposes no status checks or workflow runs for the current head. The dedicated signed publication paths are implemented, isolated and focused-tested, but release readiness is not claimed.
