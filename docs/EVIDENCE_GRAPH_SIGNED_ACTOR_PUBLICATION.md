# Signed actor-use graph-set publication

Last updated: 2026-08-02

RigorousRAG exposes two publication assurance levels. The durable command families use separate phase journals so a candidate created under one assurance level cannot be resumed under the other.

## 1. Authorization-only compatibility path

Existing commands:

```bash
python scripts/evidence_graph_set_publish.py publish-approved ...
python scripts/evidence_graph_set_publication.py execute ...
python scripts/evidence_graph_set_publication.py reconcile-one ...
```

These paths require committed governed reviewer-authorization receipts and preserve the established compensating pointer/publication semantics.

They do not embed signed actor-use aggregate digests into reviewed relation metadata.

## 2. Signed actor-use provenance path

Use these commands when signed reviewer assertions were used or when publication-level actor provenance is required:

```bash
python scripts/evidence_graph_set_signed_publish.py publish-approved ...
python scripts/evidence_graph_set_signed_publication.py seed ...
python scripts/evidence_graph_set_signed_publication.py execute ...
python scripts/evidence_graph_set_signed_publication.py reconcile-one ...
```

The signed path validates both:

1. committed reviewer-authorization receipts; and
2. signed actor-use reservations linked to each deterministic terminal decision.

## 3. Relation metadata

For every approved proposal, the signed publication ledger adds:

```json
{
  "review_signed_actor_use_count": 1,
  "review_signed_actor_use_digest": "sha256",
  "review_signed_actor_use_required": true
}
```

The digest commits the sorted actor-use receipt digests for that proposal.

No assertion body, nonce, raw signature, key material or source text is copied into relation metadata.

For a direct process/file actor decision with no signed actor-use records:

```json
{
  "review_signed_actor_use_count": 0,
  "review_signed_actor_use_digest": "deterministic-sha256-of-empty-use-set",
  "review_signed_actor_use_required": false
}
```

This preserves a deterministic provenance statement without claiming that a signed assertion existed.

## 4. Validation rules

Before candidate graph-set construction, every actor-use record must:

- be in `committed` state;
- reference the exact deterministic decision ID;
- reference the exact proposal ID;
- match proposal owner and graph-set key;
- match the terminal decision type;
- identify the same reviewer as the terminal decision;
- remain below the bounded per-decision actor-use ceiling.

Any reserved, mismatched, malformed or excessive actor-use collection fails closed.

## 5. Immediate publication

First publication:

```bash
python scripts/evidence_graph_set_signed_publish.py publish-approved \
  --owner-id alice \
  --graph-set-key systematic-review-2026 \
  --proposal-id PROPOSAL_ID \
  --expect-no-current
```

Replacement publication:

```bash
python scripts/evidence_graph_set_signed_publish.py publish-approved \
  --owner-id alice \
  --graph-set-key systematic-review-2026 \
  --proposal-id PROPOSAL_ID \
  --expected-current-set-id CURRENT_SET_ID
```

The command retains the existing compare-and-swap, post-activation authority verification and compensation behavior.

## 6. Durable signed publication

Configure a signed-only journal path distinct from the authorization-only path:

```bash
EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH=data/evidence_graph_set_publications.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH=data/evidence_graph_set_signed_publications.sqlite3
```

The signed runtime rejects:

- the same canonical path in both settings;
- relative/absolute spellings that resolve to one path;
- two existing paths that hard-link to the same inode.

Seed only after all required receipts are committed:

```bash
python scripts/evidence_graph_set_signed_publication.py seed \
  --owner-id alice \
  --graph-set-key systematic-review-2026 \
  --proposal-id PROPOSAL_ID \
  --expect-no-current
```

Execute one exact operation:

```bash
python scripts/evidence_graph_set_signed_publication.py execute OPERATION_ID \
  --worker-id publication-worker-1 \
  --lease-seconds 60
```

Reconcile the next claimable operation:

```bash
python scripts/evidence_graph_set_signed_publication.py reconcile-one \
  --owner-id alice \
  --worker-id publication-worker-1 \
  --lease-seconds 60
```

Read-only inspection and controlled retry/cancel:

```bash
python scripts/evidence_graph_set_signed_publication.py status OPERATION_ID
python scripts/evidence_graph_set_signed_publication.py list --owner-id alice
python scripts/evidence_graph_set_signed_publication.py retry OPERATION_ID \
  --owner-id alice \
  --confirm-operation-id OPERATION_ID
python scripts/evidence_graph_set_signed_publication.py cancel OPERATION_ID \
  --owner-id alice \
  --confirm-operation-id OPERATION_ID
```

Every seed, execute and reconcile operation uses only the signed publication journal. Every execute or reconcile replay reconstructs the governed authorization view and signed actor-use view before claiming or mutating graph-set state.

## 7. Why journal isolation is required

The logical publication operation ID is deterministic over owner, graph-set key, proposal IDs and expected pointer. It intentionally does not encode the command family.

If both assurance levels shared one durable journal, the following unsafe sequence would be possible:

1. the authorization-only path stores a candidate without signed actor-use metadata;
2. the process stops after the `candidate_stored` phase;
3. the signed command resumes the same operation ID;
4. recovery loads the already-created candidate rather than rebuilding its relation metadata.

Separate journal databases eliminate that recovery downgrade while retaining the mature pointer compare-and-swap and compensation engine.

## 8. Compatibility and transition

The command families still share:

- relation proposal ledger;
- authorization receipt store;
- signed actor-use store;
- immutable graph-set store;
- deterministic logical operation IDs;
- pointer compare-and-swap and compensation engine.

They do **not** share publication phase journals.

Attempts created by signed commands before journal isolation may exist in the authorization-only journal. They are not automatically migrated and must not be resumed through the signed command family. Operators should:

1. inspect the old attempt and current graph-set pointer through the authorization-only status command;
2. allow an actively leased worker to finish or wait for the lease to expire;
3. cancel the old non-terminal attempt with exact operation-ID confirmation when safe;
4. re-seed through the signed command with an explicit current/no-current pointer expectation.

Immutable non-current candidates may remain as historical records. No destructive cleanup is implied or automatically performed.

The compatibility commands remain available because direct actor deployments may not require signed assertion provenance and silently changing an operator command’s assurance contract would be unsafe.

## 9. Verification boundary

Executed in reconstructed focused workspaces using the live signed modules and minimal stubs only for unrelated repository services:

- **12/12** signed assertion, actor-binding and actor-use runtime checks passed;
- **17/17** signed publication adapter, timestamp boundary, immediate CLI, durable CLI and journal-isolation tests passed;
- Python compilation passed for both focused slices.

The 17-test publication slice includes:

- deterministic zero-use metadata;
- committed multi-use aggregation;
- reserved and mismatched use refusal;
- immediate and durable publication delegation;
- finite reconcile timestamp capture;
- corrected production-dataclass serialization fixture;
- seed/execute dependency injection;
- idle reconciliation;
- secret-free output;
- distinct default journal paths;
- explicit signed path override;
- canonical path alias refusal;
- hard-link alias refusal.

This is stronger than static inspection but is not an exact-current full repository checkout. The complete repository pytest, coverage, Ruff, Windows, container, process-kill and disk-failure matrices remain unexecuted after these commits. GitHub exposes no status checks or workflow runs for the current head. Release readiness is not claimed.
