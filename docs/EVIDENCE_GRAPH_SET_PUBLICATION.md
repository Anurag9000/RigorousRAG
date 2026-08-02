# Compensating publication of reviewed graph sets

Last updated: 2026-08-02

## Purpose

The reviewed publication layer turns an explicit list of terminally approved cross-document relation proposals into one immutable graph-set version and, under exact compare-and-swap, makes that version the logical current set.

It does not infer relations, approve proposals, rewrite member graphs or mutate authoritative vector, sparse or generation stores.

## Preconditions

Publication requires:

- an owner and logical graph-set key;
- at least one explicit proposal ID;
- every proposal to belong to that owner/key;
- one stored terminal `approved` decision for every proposal;
- exact current authoritative evidence graphs for every source/target document;
- proposal endpoint generation, graph digest, node ID and provenance digest to match those current graphs;
- an explicit pointer expectation:
  - `--expect-no-current` for first publication; or
  - `--expected-current-set-id <sha256>` for replacement.

Blind overwrite is not supported.

## Publication protocol

`tools/evidence_graph_set_publish.py` performs:

1. validate and sort unique proposal IDs;
2. read proposal scope and determine the exact member-document set;
3. acquire every process-local owner/document lock in deterministic document-ID order;
4. read the existing graph-set pointer and compare it with the explicit expectation;
5. resolve every member graph through the fail-closed authority resolver;
6. convert only approved proposals after endpoint revalidation;
7. build the deterministic graph set;
8. persist the immutable graph-set version without changing the pointer;
9. recheck every member’s authoritative generation and graph identity;
10. activate the candidate with an optimistic current-pointer compare-and-swap;
11. recheck every member after activation;
12. verify the current pointer targets the candidate.

The graph-set version is append-only. Creation time does not alter its deterministic identity or digest.

## Compensation

If failure occurs after activation:

- replacement publication restores the previous set pointer under exact compare-and-swap;
- first publication clears the candidate pointer only if it still equals the activated candidate;
- the compensated pointer is reread and verified;
- compensation errors are returned as bounded generic phase/type labels.

A failed candidate version may remain as a non-current immutable historical artifact. It is not served as current and may later be considered by reviewed retention policy.

If another publisher changes the pointer before compensation, this implementation refuses to overwrite that newer pointer and reports incomplete compensation.

## Single-process lock boundary

Member document locks prevent same-process authoritative mutations while a reviewed set is being built and activated. They are not distributed locks. Multi-process publication requires external serialization or a future database/distributed leadership layer.

Even after successful publication, read-time graph-set authority checks remain mandatory. If a member generation changes immediately after lock release, the logical current set fails closed as stale.

## Operator command

```bash
python -m tools.evidence_graph_set_publish_cli publish-approved \
  --owner-id alice \
  --graph-set-key review-2026 \
  --proposal-id <approved-proposal-1> \
  --proposal-id <approved-proposal-2> \
  --expect-no-current
```

Replacement requires the exact current ID:

```bash
python -m tools.evidence_graph_set_publish_cli publish-approved \
  --owner-id alice \
  --graph-set-key review-2026 \
  --proposal-id <approved-proposal> \
  --expected-current-set-id <current-set-sha256>
```

Successful output contains set/member/edge/proposal identities, authority digest and whether the pointer changed. It explicitly reports:

```json
{
  "reviewed_proposals_required": true,
  "automatic_approval_performed": false,
  "semantic_inference_performed": false,
  "source_text_returned": false
}
```

Failure after activation reports whether compensation completed and bounded compensation error labels. It never returns graph node text or proposal evidence content.

## Focused verification

The focused publication harness passed **9 tests** covering:

- first publication under explicit no-current expectation;
- replacement under exact current-set expectation;
- idempotent same-proposal replay;
- blind expectation mismatch refusal;
- pending/unapproved proposal refusal;
- post-activation failure clearing a first-publication pointer;
- post-activation failure restoring a previous pointer;
- exact compare-and-swap refusal for pointer clearing;
- compensation-pointer verification;
- privacy-safe CLI output and bounded compensation diagnostics.

The complete expanded Wave 5 local suite passed **57 tests** after this layer, and all changed modules/scripts compiled. These are focused local results, not the exact-head Linux, Windows and container release matrix.

## Remaining work

- Add durable publication-attempt phase journaling and crash recovery.
- Add multi-process leadership or database-scoped publication leases.
- Inject crashes around immutable set insertion, pointer activation, post-activation validation and compensation.
- Add archival/retention policy for failed non-current candidate versions.
- Add reviewer authorization and separation-of-duties enforcement.
- Add read-only bounded GraphRAG evidence selection over current authoritative graph sets.
- Add server-owned citation conversion before agent/API/browser integration.

## Permanent non-claims

- Reviewer approval is not proof of scientific truth.
- Process-local locks are not distributed coordination.
- Compare-and-swap protects one pointer; it is not a distributed transaction across member graph databases.
- A compensated failure may retain a non-current immutable candidate version.
- Current graph sets must still pass authority checks at every read.
- Release readiness is not claimed.
