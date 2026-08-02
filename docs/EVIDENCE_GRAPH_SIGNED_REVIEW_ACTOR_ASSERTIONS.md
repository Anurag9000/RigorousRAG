# Signed evidence-graph review actor assertions

Last updated: 2026-08-02

HMAC-signed actor assertions provide a stronger reviewer-identity input than a free command-line string or static environment value. They establish that the assertion was created by an entity possessing the configured shared key, was issued by the pinned issuer, and is within its validity interval.

They do **not** by themselves establish external human identity, non-repudiation or scientific correctness.

## 1. Assertion schema

```json
{
  "schema_version": 1,
  "actor_id": "reviewer-42",
  "issuer": "review-control-plane",
  "issued_at": 1785690000.0,
  "expires_at": 1785690900.0,
  "nonce": "b4ed9f...",
  "signature": "64-lowercase-hex-HMAC-SHA256"
}
```

The signature covers canonical JSON containing every field except `signature`.

## 2. Key requirements

- raw key file between 32 and 4,096 bytes;
- key bytes are used exactly as stored;
- keep the key outside the repository;
- mount it read-only with restricted filesystem permissions;
- do not add a trailing newline unless it is intentionally part of the key;
- rotate keys through a controlled issuer/configuration transition;
- never print, log or place key bytes in command arguments.

## 3. Create an assertion

```bash
python scripts/evidence_graph_review_actor_assertion.py sign \
  --actor-id reviewer-42 \
  --issuer review-control-plane \
  --key-path /run/secrets/review-actor-hmac-key \
  --output /run/rigorousrag/reviewer-42.assertion.json \
  --lifetime-seconds 900
```

The command:

- permits lifetimes from 60 seconds through 24 hours;
- generates a cryptographically random nonce unless one is supplied;
- creates a mode-0600 temporary file in the output directory;
- writes and fsyncs the complete assertion;
- publishes it through an atomic hard link;
- refuses an existing or concurrently appearing output;
- fsyncs the containing directory;
- returns metadata but never key material or the signature.

The signing command creates new files only. It has no replace mode.

## 4. Verify an assertion

```bash
python scripts/evidence_graph_review_actor_assertion.py verify \
  --assertion-path /run/rigorousrag/reviewer-42.assertion.json \
  --key-path /run/secrets/review-actor-hmac-key \
  --expected-issuer review-control-plane
```

Verification requires:

- exact versioned schema and no extra fields;
- no duplicate JSON keys;
- bounded UTF-8 assertion bytes;
- regular non-redirecting assertion and key files;
- valid actor, issuer and nonce identifiers;
- finite non-negative timestamps;
- expiry after issuance;
- at most 24-hour lifetime;
- issuance no further in the future than the configured clock-skew ceiling;
- exact expiry at verification time, with no stale-expiry grace;
- exact pinned issuer;
- valid HMAC-SHA256 through constant-time comparison.

Verification output contains only actor/issuer/time/nonce metadata and SHA-256 digests of the canonical assertion and signature. It does not return the signature or key.

## 5. Configure the decision process

```bash
EVIDENCE_GRAPH_REVIEW_ACTOR_ASSERTION_PATH=/run/rigorousrag/reviewer-42.assertion.json
EVIDENCE_GRAPH_REVIEW_ACTOR_HMAC_KEY_PATH=/run/secrets/review-actor-hmac-key
EVIDENCE_GRAPH_REVIEW_ACTOR_EXPECTED_ISSUER=review-control-plane
```

Leave these direct modes empty:

```bash
EVIDENCE_GRAPH_REVIEW_ACTOR_ID=
EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH=
```

The loader requires exactly one of:

1. direct process environment actor ID;
2. descriptor-read actor ID file;
3. signed assertion path with key path and expected issuer.

Ambiguous or incomplete configuration fails closed.

## 6. Use in a governed decision

```bash
python -m tools.evidence_graph_relation_cli decide PROPOSAL_ID \
  --owner-id alice \
  --decision approved \
  --reviewer-id reviewer-42 \
  --reason-code independently_verified
```

The command verifies the assertion, resolves `reviewer-42` from the verified actor binding, and requires the explicit reviewer argument to match. Reviewer policy and proposer/reviewer separation are then enforced independently.

The one-time decision output contains:

- actor ID;
- `binding_method=hmac_assertion`;
- deterministic binding digest;
- verification time;
- no key material.

The actor binding digest commits the assertion digest, issuer and expiry in addition to actor ID and binding method.

## 7. Security properties

Implemented protections:

- HMAC integrity and issuer pinning;
- bounded short lifetime;
- exact expiry;
- limited future-issuance skew;
- strict schema and duplicate-key refusal;
- path redirection refusal;
- minimum key strength by byte length;
- constant-time signature comparison;
- assertion and signature digests for audit correlation;
- no output overwrite by the signing tool;
- no key disclosure in normal CLI output.

## 8. Remaining limitations

A shared HMAC key provides symmetric authentication:

- any holder of the key can create assertions for any actor ID accepted by policy;
- it does not provide public-key non-repudiation;
- it does not identify the human who accessed the key;
- it is not OIDC, SAML, directory or hardware-attested identity.

At this stage, the nonce is signed but replay prevention is not yet durable. A still-valid assertion could be reused for another policy-permitted decision by the same actor. The next governance layer must reserve each assertion/nonce to one deterministic terminal decision while permitting exact crash recovery for that same decision.

Still open after nonce reservation:

- asymmetric signatures and key IDs;
- external IAM/OIDC assertions;
- hardware-backed signing;
- issuer key rotation and overlap windows;
- remote submission transport authentication;
- multi-party/quorum approval.

Permanent non-claim: a cryptographically valid actor assertion proves possession of a configured shared key, not scientific truth or external peer review.
