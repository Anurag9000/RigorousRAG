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
- [x] Uses the existing deterministic publication operation identity.
- [x] Uses the existing durable publication phase journal.
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

## 4. Assurance-level compatibility

Authorization-only commands remain:

```bash
python scripts/evidence_graph_set_publish.py ...
python scripts/evidence_graph_set_publication.py ...
```

They continue to require committed governed authorization receipts. They do not include the additional actor-use aggregate in relation metadata.

Signed assertion deployments should use only the dedicated signed commands when publication-level signed actor provenance is required.

No existing publication journal or graph-set migration is needed because both command families use the same stores and deterministic operation identities.

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
- secret-free compensation output.

## 6. Verification boundary

Executed in the reconstructed focused workspace:

- exact signed assertion, actor-binding and actor-use modules compiled;
- 12/12 core signed actor runtime checks passed.

Not yet executed in an exact-current repository checkout:

- signed publication adapter tests;
- immediate signed publication CLI tests;
- durable signed publication CLI tests;
- complete repository pytest;
- coverage, Ruff, Windows and container matrices;
- process-kill and disk-failure publication injection.

The dedicated signed publication paths are implemented and documented, but release readiness is not claimed.
