# Wave 5 status addendum — signed review actor and replay governance

Last updated: 2026-08-02

This addendum extends `WAVE5_STATUS_ADDENDUM_GOVERNED_REVIEW_2026-08-02.md`. It records the stronger actor-identity and replay controls added after the governed reviewer-policy layer. It does not extend older exact-current full-suite evidence and does not claim release readiness.

## 1. Process-owned actor binding

Implemented:

- [x] Decision commands require a process-owned actor identity.
- [x] Explicit `--reviewer-id` must exactly equal the process actor.
- [x] Actual terminal decisions use the resolved actor ID, not the untrusted CLI value.
- [x] Exactly one direct actor source: environment value or descriptor-read file.
- [x] Missing or multiply configured actor sources fail closed.
- [x] Actor files are bounded UTF-8 regular files.
- [x] Symbolic-link and reparse traversal is refused.
- [x] Deterministic actor-binding digest.
- [x] Binding reconstruction detects actor/method/digest tampering.
- [x] Direct actor modes are empty by default in configuration.

Non-claim: direct process binding prevents free-form CLI reviewer impersonation but does not prove external human identity.

## 2. HMAC-signed short-lived actor assertions

Implemented:

- [x] Versioned canonical JSON assertion schema.
- [x] HMAC-SHA256 signature.
- [x] Constant-time signature comparison.
- [x] Minimum 32-byte and maximum 4,096-byte key files.
- [x] Maximum 16-KiB assertion files.
- [x] Strict duplicate-key and extra-field refusal.
- [x] Pinned issuer.
- [x] Finite issuance and expiry timestamps.
- [x] Maximum 24-hour assertion lifetime.
- [x] Maximum 300-second future-issuance skew.
- [x] Exact expiry with no stale-expiry grace.
- [x] Signed nonce.
- [x] Assertion and signature digests for audit correlation.
- [x] Assertion, key and actor path redirect refusal.
- [x] Signed binding commits assertion digest, issuer and expiry.
- [x] Direct and signed actor modes are mutually exclusive.

Still open:

- [ ] Asymmetric signatures and key IDs.
- [ ] External IAM/OIDC/SAML assertions.
- [ ] Hardware-backed signing or attestation.
- [ ] Issuer key rotation and overlap windows.
- [ ] Remote submission transport authentication.

Non-claim: a valid HMAC assertion proves possession of a configured shared key, not which human accessed it.

## 3. Assertion provisioning and verification

Implemented:

- [x] Dedicated sign command.
- [x] Dedicated verify command.
- [x] Random nonce by default.
- [x] Lifetime bounded from 60 seconds through 24 hours.
- [x] Mode-0600 temporary assertion file.
- [x] Complete write and file fsync.
- [x] Atomic hard-link publication.
- [x] Existing or concurrently appearing outputs are never overwritten.
- [x] Containing-directory fsync.
- [x] Key material and raw signatures are never returned.
- [x] Verification returns only bounded identity/time/nonce metadata and digests.

## 4. Durable one-decision assertion use

Implemented:

- [x] Separate SQLite signed actor-use journal.
- [x] One assertion digest may reserve only one deterministic decision.
- [x] Exact assertion/decision replay is idempotent.
- [x] Assertion reuse for a different proposal or decision is refused.
- [x] Reservation occurs before authorization or terminal-decision mutation.
- [x] `reserved -> committed` is the only state transition.
- [x] Reservation and commit timestamps are finite and monotonic.
- [x] Reservation identity commits assertion, decision, proposal, owner, graph-set key, decision type, actor, issuer, binding and expiry.
- [x] Reconstruction recomputes the use digest.
- [x] Row/payload tampering is refused.
- [x] Parent and database identity changes are refused.
- [x] No delete, release, replace or reuse command.
- [x] Read-only status/list audit CLI.
- [x] Audit output excludes signatures, keys and source text.

Recovery behavior:

- [x] Stable terminal-decision replay ignores a later invocation timestamp while retaining the original stored `decided_at`.
- [x] Stable replay compares every governed semantic field.
- [x] Same signed assertion may complete exact crash recovery for its deterministic decision.
- [x] A fresh assertion may recover that decision only when an older durable reservation proves signed review began before the decision existed.
- [x] Signed provenance cannot be retroactively attached to an existing terminal decision with no prior signed reservation.
- [x] All reservations for a recovered decision become committed after authorization and decision durability are verified.

## 5. Signed actor-use publication provenance

Implemented in a dedicated publication adapter:

- [x] Reads committed signed actor-use records by deterministic decision ID.
- [x] Refuses reserved, mismatched or out-of-scope use records.
- [x] Refuses an excessive actor-use count.
- [x] Preserves original proposal and decision identities.
- [x] Produces a deterministic per-relation aggregate actor-use digest and count.
- [x] Exposes no assertion body, nonce, signature or key through relation metadata.
- [x] Direct actor decisions receive a deterministic zero-use aggregate.

Not yet canonical:

- [ ] Immediate publication CLI switched to the signed-use provenance adapter.
- [ ] Durable publication/recovery CLI switched to the signed-use provenance adapter.
- [ ] Repository-native contracts for both switched operator paths.

Until those entrypoints are switched, graph publication still requires committed reviewer authorization receipts but does not yet embed the signed actor-use aggregate into reviewed edge metadata.

## 6. Focused verification performed

A reconstructed executable workspace used the exact new assertion, actor-binding and actor-use implementations with minimal stubs only for older relation/security types.

Results:

- [x] Python compilation passed for the exact overlaid modules.
- [x] 12/12 executable runtime checks passed.

Executed checks:

1. HMAC assertion signing.
2. Valid signature verification.
3. Exact-expiry refusal despite configured skew.
4. Assertion-field tamper refusal.
5. Signed actor-binding construction.
6. Explicit reviewer/actor mismatch refusal.
7. Actor-use reservation.
8. Exact reservation replay.
9. Monotonic actor-use commit.
10. Exact committed replay.
11. Assertion reuse for another decision refusal.
12. Actor-use database payload tamper refusal.

All corresponding repository-native test files are committed, including policy, stable replay, actor source, signed assertion, assertion CLI, actor-use storage, actor-use CLI and signed governed decision scenarios.

Verification limitations:

- this was not an exact-current repository checkout;
- older relation/security types were minimally stubbed in the executable harness;
- complete exact-current pytest was not run;
- coverage, Ruff, Windows, Docker/Compose and connected-provider tests were not run;
- process-kill/disk-full/fsync fault injection was not run.

The earlier 114/114 evidence-graph and complete repository pytest result remains tied to the older unchanged archive and is not extended to this head.

## 7. Remaining governance work

Highest priority:

- [ ] Make signed-use publication provenance canonical in both operator publication paths.
- [ ] Bind signed actor-use aggregate directly into authorization-receipt identity or a linked immutable publication manifest.
- [ ] Add asymmetric signing and key IDs.
- [ ] Integrate external IAM/directory identity.
- [ ] Add two-person/quorum approval.
- [ ] Add reviewer assignment, correction, appeal and escalation UI.
- [ ] Add inter-annotator agreement reports.
- [ ] Add signed append-only audit export and backup/restore drills.
- [ ] Run exact-current complete verification and multi-process/disk-failure matrices.

## 8. Permanent non-claims

- Actor authentication does not establish scientific truth.
- A signed assertion does not establish semantic entailment.
- HMAC authentication is not public-key non-repudiation.
- SQLite durability is not distributed consensus.
- Focused synthetic execution is not exact-current full-repository verification.
- Release readiness is not claimed.
