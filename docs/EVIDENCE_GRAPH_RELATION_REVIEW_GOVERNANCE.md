# Evidence-graph semantic relation review governance

Last updated: 2026-08-02

This document describes the fail-closed authorization boundary for terminal semantic-relation decisions and reviewed graph-set publication.

The system still does **not** infer, approve or publish semantic relations automatically. A proposal is text-free governance metadata over exact generation-scoped graph endpoints. A terminal decision is accepted only when the reviewer is authorized by the current policy and is independent from the proposal author. Authoritative operator publication additionally requires a committed authorization receipt for every approved proposal.

## 1. Durable stores

Three separate durable records are intentionally retained:

1. `EVIDENCE_GRAPH_RELATION_DB_PATH`
   - immutable proposals;
   - one immutable terminal decision per proposal;
   - no authorization policy state.
2. `EVIDENCE_GRAPH_REVIEW_AUTH_DB_PATH`
   - immutable authorization identity per terminal decision;
   - monotonic `authorized -> committed` state;
   - policy, grant and authorization digests;
   - no proposal text or graph-node text.
3. `EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH`
   - durable publication intents and recovery phases;
   - exact proposal IDs and pointer expectations;
   - no approval authority of its own.

Separating these stores prevents an approval row, an authorization receipt and a publication attempt from being mistaken for the same security decision.

## 2. Policy configuration

Configure exactly one policy source:

```bash
EVIDENCE_GRAPH_REVIEW_POLICY_PATH=config/review-policy.json
```

or:

```bash
EVIDENCE_GRAPH_REVIEW_POLICY_JSON={"schema_version":1,"reviewers":[...]}
```

Configuring neither source or both sources fails closed for new review decisions.

A bounded example is committed at:

```text
config/evidence_graph_relation_review_policy.example.json
```

Every placeholder must be replaced before operational use.

### Policy schema

```json
{
  "schema_version": 1,
  "reviewers": [
    {
      "reviewer_id": "reviewer-42",
      "owners": ["alice"],
      "graph_set_keys": ["systematic-review-2026"],
      "decisions": ["approved", "rejected", "superseded"],
      "expires_at": 1798761600.0
    }
  ]
}
```

Rules:

- reviewer IDs are stable explicit identifiers;
- owner and graph-set scopes are exact by default;
- a sole `"*"` may be used for owner or graph-set scope, but cannot be combined with explicit entries;
- decision scope contains only `approved`, `rejected` and/or `superseded`;
- decision wildcards are forbidden;
- expired grants fail closed;
- duplicate reviewer IDs, duplicate JSON keys, non-finite numbers, unsupported fields and oversized policies fail closed;
- policy files are read through a bounded descriptor and reject symbolic-link/reparse traversal.

## 3. Separation of duties

For every terminal decision:

- the reviewer must have a current grant for the proposal owner, graph-set key and decision type;
- the reviewer ID must differ from the proposal proposer ID;
- for `superseded`, the replacement proposal must:
  - differ from the original proposal;
  - have the same owner;
  - have the same graph-set key;
  - have the same relation key;
  - have a proposer different from the reviewer.

These checks occur before the authorization receipt is prepared and before the terminal decision is written.

## 4. Authorization identity

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

Every reconstructed `ReviewAuthorization` recomputes this digest. Changing any governed field while retaining the old digest is rejected as corruption.

`authorized_at` is recorded but intentionally not part of the deterministic identity. Receipt transition timestamps are separately required to be finite and monotonic.

## 5. Crash-safe decision sequence

The governed decision service executes:

1. load the current policy;
2. resolve the proposal and any replacement proposal;
3. enforce scope and separation of duties;
4. compute the deterministic authorization identity;
5. persist an idempotent `authorized` receipt;
6. write the immutable terminal decision;
7. transition the receipt to `committed`;
8. reread and verify durable committed state.

If the process stops after step 6 but before step 7, exact replay is allowed only when:

- the existing terminal decision exactly equals the requested decision;
- a matching prepared authorization receipt exists;
- proposal, decision, owner, graph-set key, reviewer and decision type all match.

The replay then performs only the monotonic receipt transition. A pre-existing decision with no governed receipt is not retroactively trusted.

## 6. Policy updates and revocation

The policy is reloaded for every **new** decision. Therefore:

- removing a reviewer blocks new decisions immediately;
- narrowing scopes blocks new out-of-scope decisions immediately;
- expiry blocks decisions after the expiry time;
- changing a grant produces a new grant digest and authorization identity.

Committed historical receipts remain immutable evidence that the decision was authorized under the recorded policy and grant. Current policy changes do not rewrite or erase past decisions. Revoking already committed decisions requires an explicit new governance workflow, such as a superseding proposal/decision; silent retroactive mutation is intentionally unsupported.

## 7. Governed publication boundary

The operator publication commands use `GovernedPublicationLedger`, a read-only view over the immutable proposal ledger.

Before graph-set mutation, it requires for every proposal:

- exact owner and graph-set scope;
- terminal `approved` decision;
- committed authorization receipt;
- matching proposal ID and decision ID;
- matching reviewer ID;
- matching owner and graph-set key;
- `separation_of_duties_enforced=true`.

The original proposal object and deterministic proposal ID are never modified. The relation converter receives a delegated proposal view whose metadata additionally carries:

- `review_authorization_digest`;
- `review_policy_digest`;
- `review_grant_digest`;
- `review_authorization_state=committed`;
- `review_separation_of_duties=true`.

Thus reviewed graph edges preserve authorization provenance without changing proposal identity.

Both operator paths are governed:

```bash
python scripts/evidence_graph_set_publish.py publish-approved ...
python scripts/evidence_graph_set_publication.py execute ...
```

The durable path revalidates receipts before every execution or recovery replay. A journal intent has no approval authority by itself.

The lower-level compensating publication functions remain internal primitives for isolated testing and composition. Operator automation must use the governed CLI/runtime boundary.

## 8. CLI workflow

### Submit a proposal

```bash
python -m tools.evidence_graph_relation_cli propose \
  --owner-id alice \
  --graph-set-key systematic-review-2026 \
  --relation-key paper-a-supports-paper-b \
  ...
```

Proposal creation never approves the proposal.

### Record a governed terminal decision

```bash
python -m tools.evidence_graph_relation_cli decide PROPOSAL_ID \
  --owner-id alice \
  --decision approved \
  --reviewer-id reviewer-42 \
  --reason-code independently_verified
```

The command fails closed when policy is absent, the reviewer is unauthorized, the reviewer authored the proposal, replacement scope is invalid, or durable receipt storage is unavailable.

### Inspect authorization receipts

```bash
python scripts/evidence_graph_relation_authorizations.py status DECISION_ID
python scripts/evidence_graph_relation_authorizations.py list \
  --owner-id alice \
  --graph-set-key systematic-review-2026 \
  --state committed
```

This CLI is read-only. It cannot prepare, commit, retry, alter or delete receipts.

### Publish

```bash
python scripts/evidence_graph_set_publish.py publish-approved \
  --owner-id alice \
  --graph-set-key systematic-review-2026 \
  --proposal-id PROPOSAL_ID \
  --expect-no-current
```

Every listed proposal must have a committed governed receipt.

## 9. Legacy decisions

Older terminal decisions without authorization receipts remain visible in proposal status/list output with:

```json
{"governed_review": false, "review_authorization": null}
```

They cannot pass the governed operator publication boundary. They are not silently migrated, backfilled or treated as authorized.

A deliberate migration must create new governed proposals/decisions or another explicit, separately audited migration protocol. This implementation provides no automatic grandfathering.

## 10. Privacy and non-claims

The policy and receipt surfaces contain identifiers, scopes, digests, timestamps and decision metadata only. They do not contain source document text, graph-node text, extracted evidence passages or model prompts.

Permanent non-claims:

- reviewer authorization does not establish scientific truth;
- approval does not establish semantic entailment;
- a committed receipt proves an internal governance transition, not external peer review;
- a graph edge remains an explicit reviewed assertion;
- a green focused suite does not establish release readiness.
