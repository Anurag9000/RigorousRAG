# Signed actor-use graph-set publication

Last updated: 2026-08-02

RigorousRAG now exposes two publication assurance levels.

## 1. Authorization-only compatibility path

Existing commands:

```bash
python scripts/evidence_graph_set_publish.py publish-approved ...
python scripts/evidence_graph_set_publication.py execute ...
python scripts/evidence_graph_set_publication.py reconcile-one ...
```

These paths require committed governed reviewer-authorization receipts and preserve the established compensating pointer/publication semantics.

They do not yet embed signed actor-use aggregate digests into reviewed relation metadata.

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

## 6. Durable publication

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

Read-only inspection and controlled retry/cancel use the same command names as the compatibility journal CLI:

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

Every execute or reconcile replay reconstructs the governed authorization view and the signed actor-use view before claiming or mutating graph-set state.

## 7. Time boundary

`reconcile-one` captures one finite non-negative timestamp and uses it for both claim discovery and execution. `None`, NaN, infinity and negative values are never forwarded to the durable journal.

## 8. Compatibility and migration

The signed commands use the same:

- relation proposal ledger;
- authorization receipt store;
- signed actor-use store;
- graph-set store;
- publication phase journal;
- deterministic operation IDs;
- pointer compare-and-swap and compensation engine.

No data migration is required. Existing publication attempts can be executed through the signed path only when their referenced proposals pass signed-use provenance validation.

The compatibility commands remain available because direct actor deployments may not require signed assertion provenance and because silently changing an operator command’s assurance contract would be unsafe.

## 9. Verification boundary

Committed contracts cover:

- deterministic zero-use metadata;
- committed multi-use aggregation;
- reserved and mismatched use refusal;
- immediate publication delegation;
- durable publication delegation;
- finite reconcile timestamp capture;
- seed-time signed provenance validation;
- execute-time authorization/actor-use dependency injection;
- idle reconciliation;
- secret-free success/failure output.

The newest signed-publication modules have not yet been run in an exact-current repository checkout. Release readiness is not claimed.
