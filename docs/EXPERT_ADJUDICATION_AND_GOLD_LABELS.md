# Expert adjudication and gold-label workflow

`evaluation.expert_adjudication` is the durable workflow layer above the analytical agreement metrics in `evaluation.expert_review`.

## Authority model

A case is scoped to one owner and binds an item SHA-256, an immutable set of authoritative evidence digests, and a versioned label schema. Raw queries, answers, documents, images, evidence text and reviewer rationale do not enter the adjudication database; rationale is referenced by SHA-256 only.

The first case round is deterministic. A resolved case can be reopened only by creating a new child round with a correction-reason digest and actor-derived audit identity. Earlier judgments and resolutions remain historical records and are never overwritten.

## Independent review and fencing

Reviewers obtain expiring claims with monotonic fencing tokens. Multiple independent `reviewer` identities may work on one case. The `adjudicator` role is exclusive: the same identity may not act as both reviewer and adjudicator on one case, role separation is checked both when the claim is acquired and when a judgment commits, and a different adjudicator cannot silently replace an adjudicator who has already committed a judgment.

Every judgment also uses the current case revision as compare-and-swap input. A stale UI/process therefore cannot write against a case whose review state changed after it was read.

## Append-only corrections

A reviewer's first judgment is revision 1. A correction creates a new judgment revision and must explicitly name the current judgment that it supersedes. The old row remains immutable. Consensus and export use only each reviewer/role's latest judgment while retaining the complete audit trail.

## Resolution policy

`AdjudicationPolicy` defines:

- minimum independent reviewer count;
- the fraction required for automatic reviewer consensus;
- whether automatic resolution is allowed; and
- minimum confidence for an adjudicator decision.

Below quorum the case remains `open`. Once quorum is reached, a unique label meeting the configured consensus fraction can resolve automatically. Otherwise the case becomes `needs_adjudication`. A sufficiently confident independent adjudicator can resolve that disagreement. Resolution receipts bind the active-judgment set, schema, policy, owner, label and method by SHA-256.

## Gold export and corrections

Gold-label manifests contain only the latest round for each item. If a resolved round is reopened, the old resolution is retained historically but is removed from the **current** gold view until the new round resolves. This prevents known-stale labels from continuing to train or evaluate models while a correction is pending.

Each exported record binds the current case, item digest, evidence-set digest, label, schema digest, resolution receipt and round. `write_gold_manifest` writes this digest-only manifest atomically.

## Execution boundary

The repository now contains the complete source lifecycle for assignments, judgments, adjudication, corrections and gold-label manifests. Actual expert decisions are human/runtime work and are not fabricated by this source-only implementation sweep.