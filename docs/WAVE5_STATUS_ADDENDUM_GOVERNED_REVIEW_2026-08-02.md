# Wave 5 status addendum — governed semantic-relation review

Last updated: 2026-08-02

This addendum supersedes the reviewer-governance open items in `WAVE5_IMPLEMENTATION_STATUS_2026-08-02.md` and `WAVE5_STATUS_ADDENDUM_PUBLICATION_CITATIONS_2026-08-02.md`. It does not rewrite their historical verification evidence and does not claim release readiness.

## 1. Reviewer authorization and scope

Now implemented:

- [x] Strict versioned reviewer-policy schema.
- [x] Stable reviewer identifiers.
- [x] Exact owner scopes.
- [x] Exact graph-set-key scopes.
- [x] Decision scopes for approved, rejected and superseded decisions.
- [x] Optional finite reviewer-grant expiry.
- [x] Owner and graph-set wildcards only as sole scope values.
- [x] Decision wildcards prohibited.
- [x] Duplicate reviewer IDs and duplicate JSON keys rejected.
- [x] Oversized, non-finite, malformed and unsupported policy values rejected.
- [x] Bounded descriptor-based policy-file reads.
- [x] Symbolic-link/reparse traversal refusal.
- [x] Exactly one inline or file policy source.
- [x] Missing or multiply configured policy fails closed.
- [x] Policy reloaded for every new terminal decision so revocation, narrowing and expiry take effect immediately.
- [x] Reviewer policy left unconfigured by default in `.env.example`.

Still open:

- [ ] Cryptographic or external-IAM binding of reviewer IDs to authenticated human/operator identities.
- [ ] Group/directory synchronization and automated reviewer deprovisioning.
- [ ] Organization-wide emergency policy-revocation workflow.

## 2. Separation of duties and supersession

Now implemented:

- [x] Proposal authors may not review their own proposals.
- [x] Replacement-proposal authors may not authorize their own replacement.
- [x] Superseding proposals must differ from the original proposal.
- [x] Superseding proposals must retain the same owner.
- [x] Superseding proposals must retain the same graph-set key.
- [x] Superseding proposals must retain the same relation key.
- [x] Separation-of-duties and replacement-scope results are committed into authorization identity.

Still open:

- [ ] Two-person or quorum approval policies.
- [ ] Required second review for high-impact relation types.
- [ ] Conflict-of-interest declarations beyond proposer/reviewer identity inequality.
- [ ] Time-separated or rotating-reviewer policies.

## 3. Durable authorization receipts

Now implemented:

- [x] Separate SQLite authorization-receipt journal.
- [x] Immutable authorization identity per decision ID.
- [x] Monotonic `authorized -> committed` state.
- [x] Finite and monotonic prepared, committed and updated timestamps.
- [x] Idempotent same-identity prepare and commit replay.
- [x] Receipt path, parent and database-identity defenses.
- [x] Strict nested reconstruction on every read.
- [x] Row/payload identity verification.
- [x] No receipt deletion, mutation, retry or compaction command.
- [x] Read-only status/list CLI.
- [x] Owner, graph-set and state filters.
- [x] No source text, node text or extracted evidence in receipt outputs.

The deterministic authorization digest commits:

- proposal ID;
- decision ID;
- owner ID;
- graph-set key;
- decision type;
- reviewer ID;
- complete policy digest;
- exact reviewer-grant digest;
- separation-of-duties result;
- replacement-scope-validation result.

Every reconstructed authorization recomputes this digest and refuses any mismatch.

Still open:

- [ ] Signed external audit export.
- [ ] Legal hold and minimum-retention policy.
- [ ] Backup/restore and disaster-recovery drills.
- [ ] Hardware-backed or external timestamp attestation.

## 4. Crash-safe governed decisions

Now implemented:

1. Load current policy.
2. Resolve proposal and replacement proposal.
3. Enforce reviewer scope and separation of duties.
4. Compute deterministic authorization identity.
5. Persist an `authorized` receipt.
6. Persist the immutable terminal decision.
7. Transition the receipt to `committed`.
8. Reread and verify committed state.

Recovery behavior:

- [x] A crash after receipt preparation but before terminal decision leaves an auditable authorized receipt and no approved relation.
- [x] A crash after terminal decision but before receipt commit is recoverable by exact replay.
- [x] Replay requires the existing decision to equal the requested decision exactly.
- [x] Replay requires a matching prepared receipt.
- [x] Proposal, decision, owner, graph-set key, reviewer and decision type are revalidated.
- [x] A legacy terminal decision without a governed receipt is not retroactively trusted.
- [x] A different terminal decision for the same proposal is refused.

Still open:

- [ ] Exact repository-stack process-kill and SQLite fault injection across each decision phase.
- [ ] Disk-full, fsync and database-corruption recovery drills.
- [ ] Cross-process reviewer-decision leadership or serialization beyond SQLite locking.

## 5. Governed graph-set publication

Now implemented:

- [x] Immediate operator publication requires committed receipts for every approved proposal.
- [x] Durable publication execution requires committed receipts before claiming or mutating graph-set state.
- [x] Every crash-recovery replay revalidates committed receipts.
- [x] Publication intents have no approval authority by themselves.
- [x] Legacy approvals without receipts are non-executable.
- [x] Authorized-but-uncommitted decisions are non-executable.
- [x] Receipt proposal, decision, owner, graph-set key and reviewer identities are revalidated.
- [x] Original proposal objects and deterministic proposal IDs remain unchanged.
- [x] A delegated read-only proposal view supplies authorization provenance to the relation converter.
- [x] Published relation metadata includes authorization, policy and grant digests.
- [x] Published relation metadata records committed authorization state and separation of duties.
- [x] Existing compare-and-swap pointer activation, authority verification and compensation remain unchanged.

The lower-level compensating publication functions remain internal composition/testing primitives. Operator automation is required to use the governed publication CLI/runtime boundary.

Still open:

- [ ] Policy-driven quorum publication requirements.
- [ ] Signed graph-set publication manifests.
- [ ] Distributed publication leadership across unrelated processes.
- [ ] Exact-current fault injection through governed receipt checks and pointer compensation.

## 6. Operator surfaces

Implemented:

```text
python -m tools.evidence_graph_relation_cli propose
python -m tools.evidence_graph_relation_cli decide
python -m tools.evidence_graph_relation_cli status
python -m tools.evidence_graph_relation_cli list
python scripts/evidence_graph_relation_authorizations.py status
python scripts/evidence_graph_relation_authorizations.py list
python scripts/evidence_graph_set_publish.py publish-approved
python scripts/evidence_graph_set_publication.py execute
python scripts/evidence_graph_set_publication.py reconcile-one
```

Properties:

- [x] Proposal creation never approves a proposal.
- [x] Decision creation fails closed without policy and durable receipt storage.
- [x] Status/list identify older ungoverned decisions explicitly.
- [x] Authorization audit CLI is read-only.
- [x] Publication outputs state that committed review authorizations are required.
- [x] Outputs contain no source text.

Still open:

- [ ] Human review web UI.
- [ ] Reviewer assignment queue.
- [ ] Correction, appeal and escalation UI.
- [ ] Notification and service-level-objective tracking.

## 7. Focused contracts committed

New contracts cover:

- reviewer-policy schema, file loading and duplicate-key refusal;
- owner, graph-set, decision and expiry scope;
- proposer/reviewer separation;
- supersession scope and replacement-author independence;
- authorization preparation, commit and exact replay;
- refusal of ungoverned legacy decisions;
- receipt storage and payload tamper detection;
- deterministic authorization reconstruction;
- governed relation CLI behavior;
- read-only authorization audit CLI;
- immediate publication delegation through governed ledger view;
- durable publication delegation through governed ledger view;
- refusal of missing or uncommitted receipts;
- preservation of deterministic proposal identity;
- authorization provenance in relation-converter metadata;
- GraphRAG agent import ordering and partially initialized module recovery.

## 8. Verification boundary

Locally available verification performed for this slice:

- Python compilation passed for the new policy, receipt, runtime, CLI, governed-publication and import-hook modules.
- A synthetic end-to-end governance harness passed using the exact new modules with minimal stubs for older repository dependencies. It exercised policy scope, terminal decision, `authorized -> committed` receipt transition, deterministic tamper refusal, governed publication delegation and legacy-approval refusal.
- The deferred GraphRAG agent hook was exercised with a synthetic partially initialized module and installed exactly once after the final required assignment.
- All corresponding repository-native tests are committed.

Not yet performed on this newest head:

- complete exact-current repository pytest;
- exact-current evidence-graph focused suite total;
- coverage report;
- Ruff;
- Windows matrix;
- Docker/Compose build and readiness;
- connected-provider tests;
- multi-process and disk-failure injection.

The earlier 114/114 focused and full-repository pytest evidence remains tied to the older unchanged archive and is not extended to these newest commits.

## 9. Permanent non-claims

- Reviewer authorization does not establish scientific truth.
- Reviewer independence by identifier inequality is not a complete conflict-of-interest system.
- A committed receipt is an internal governance record, not external peer review.
- An approved relation remains an explicit reviewed assertion, not automatic entailment.
- Durable SQLite state is not distributed consensus.
- Focused or synthetic tests are not the complete release matrix.
- Release readiness is not claimed.
