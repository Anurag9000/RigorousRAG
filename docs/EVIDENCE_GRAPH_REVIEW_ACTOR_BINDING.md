# Evidence-graph review actor binding

Last updated: 2026-08-02

The governed semantic-relation decision CLI requires two independent conditions:

1. the reviewer ID must be authorized by the current reviewer policy; and
2. the same reviewer ID must be bound to the running operator process.

The second condition prevents a caller from selecting an arbitrary authorized reviewer merely by passing `--reviewer-id`.

## 1. Configure exactly one actor source

### Environment value

```bash
EVIDENCE_GRAPH_REVIEW_ACTOR_ID=reviewer-42
```

### Descriptor-read file

```bash
EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH=/run/secrets/rigorousrag-review-actor-id
```

The file contains only the stable reviewer ID:

```text
reviewer-42
```

Configuring neither source or both sources fails closed.

## 2. Decision command

```bash
python -m tools.evidence_graph_relation_cli decide PROPOSAL_ID \
  --owner-id alice \
  --decision approved \
  --reviewer-id reviewer-42 \
  --reason-code independently_verified
```

`--reviewer-id` is retained as an explicit operator declaration, but it must exactly equal the process-owned actor ID. A mismatch is rejected before a decision or authorization receipt is created.

The actual decision always uses the actor ID resolved by the process boundary, not the untrusted command-line value.

## 3. File safety

The actor-file loader:

- resolves relative paths against the current working directory;
- rejects control characters and oversized paths;
- validates every existing path component;
- rejects symbolic links and Windows reparse points;
- opens with `O_NOFOLLOW` where available;
- requires a regular file;
- permits at most 4,096 bytes;
- requires UTF-8 text;
- validates one non-empty bounded actor identifier.

## 4. Actor binding identity

The process constructs a deterministic binding digest over:

```json
{
  "scope": "rigorousrag-review-actor-binding-v1",
  "actor_id": "reviewer-42",
  "binding_method": "process_environment"
}
```

or the corresponding `descriptor_file` method.

The decision command returns this binding digest and method in a one-time `review_actor_binding` object. It explicitly marks `durable_receipt_field=false`: the current authorization receipt durably records the reviewer ID, policy digest and grant digest, but not this process-binding digest.

This distinction is deliberate and prevents the implementation from claiming stronger identity assurance than it provides.

## 5. Trust boundary

This mechanism provides **process-owned reviewer selection**, not cryptographic identity authentication.

It protects against:

- accidentally naming the wrong authorized reviewer;
- choosing another reviewer through a free CLI string;
- reading an actor identity through a redirected file path;
- ambiguous multiple actor sources.

It does not prove:

- who launched or configured the process;
- that the operating-system account belongs to the named human;
- that the environment or secret mount was provisioned by an external identity provider;
- that the reviewer possessed a private signing key;
- that the decision was made interactively by that reviewer.

## 6. Operational recommendations

Until signed or IAM-backed assertions are implemented:

- run review commands in a dedicated restricted service account or job;
- inject the actor ID through a protected secret mount rather than shell history;
- restrict write access to policy, actor and authorization-database paths;
- use a separate process per reviewer identity;
- avoid wildcard reviewer grants;
- preserve authorization-receipt and process audit logs;
- rotate or revoke the actor configuration when reviewer duties end.

## 7. Remaining work

Still open:

- signed short-lived actor assertions;
- external IAM/OIDC or directory binding;
- hardware-backed signing or attestation;
- durable actor-binding digest in the authorization receipt schema;
- nonce/replay controls for remote review submissions;
- multi-party and quorum approval;
- service-account and human-identity distinction in policy.

Permanent non-claim: matching a process-owned actor ID and policy grant does not establish scientific truth or external peer review.
