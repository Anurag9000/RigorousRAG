# Wave 5 verification addendum — governed Ed25519 custody signatures

Last updated: 2026-08-03

This addendum records the exact implementation and execution boundary for public-key signed external restore chain-of-custody evidence.

## Repository implementation

The implementation was added directly to `main` without a branch or pull request.

Primary commits:

- `0be11d33964e827f57d693791cb7e6820e328cc1` — governed Ed25519 signer-key registry, signing, verification, RFC 3161 binding, CLIs and scripts;
- `6e2e078bd258fe9a53e777c315c3ab816636cda4` — focused repository-native contracts;
- `72e40bdf1864d928432aef440ca08629ec1c5934` — public-key registry configuration;
- `6e1aff5dc1800f81d58807845f96346c096b0c50` — operator/security runbook;
- `f9cc9e7cd7f07a1d7feb120fd7def08b4ac8b02a` — current Wave 5 backlog synchronization;
- `0bdf0aedceca3d6f5f280bf4c87eee95f2e0a3a6` — correction of the focused RFC 3161 real-module test seam.

The last correction changed only a test double. The production governed timestamp verifier already used the real RFC 3161 trust-profile boundary.

## Implemented guarantees

### Governed signer public-key registry

- owner and key-ID scope;
- Ed25519-only algorithm contract;
- raw 32-byte public-key validation;
- canonical public-key fingerprint;
- explicit validity interval;
- active and retired states;
- process-owned actor provenance for registration and retirement;
- deterministic record digests;
- exact replay and scope-collision refusal;
- monotonic retirement and no reactivation;
- retained historical public-key records;
- parent-directory, symlink/reparse and database-inode defenses.

### Signing boundary

- complete pre-existing custody manifest verification before signing;
- governed active-key and validity checks;
- protected unencrypted PEM PKCS#8 Ed25519 private-key loading;
- POSIX group/other permission refusal;
- private-derived public-key fingerprint matching against the registry;
- canonical JSON signing payload;
- 64-byte Ed25519 signature encoded as canonical base64;
- deterministic envelope digest;
- atomic no-overwrite output;
- no private-key persistence, serialization, logging or output.

### Verification boundary

- offline verification using a pinned Ed25519 public key;
- expected owner and key-ID pinning;
- complete embedded custody-manifest reconstruction;
- manifest and envelope digest validation;
- public-key fingerprint validation;
- Ed25519 signature verification;
- governed current verification against an active registry record;
- generic privacy-safe CLI failures.

### RFC 3161 and retired-key history

- binding of a verified RFC 3161 receipt to the canonical signed-envelope bytes;
- deterministic timestamped-envelope binding digest;
- basic offline timestamp receipt and subject verification;
- fresh same-process governed TSA verification for historical retired-key acceptance;
- exact receipt-digest and subject matching between embedded and newly governed verification;
- signature-time validation against key activation, expiry and retirement boundaries;
- refusal to accept a retired key using current wall-clock verification alone.

## Focused execution evidence

Executed in the reconstructed focused workspace after the production modules were split into their final repository paths and after correcting the real RFC 3161 module test seam:

```text
python -m compileall -q tools scripts tests
python -m pytest -q \
  tests/unit/test_evidence_graph_set_signed_retirement_restore_custody_signature_compact.py \
  tests/unit/test_evidence_graph_set_signed_retirement_restore_custody_signature_cli.py
```

Result:

```text
8 passed in 0.13s
```

The eight focused contracts cover:

1. public-key registration replay, collision refusal, retirement and registry identity replacement;
2. governed signing, offline verification, manifest tamper refusal and governed active-key verification;
3. RFC 3161 binding and historical verification after signer-key retirement;
4. private-key permission refusal and atomic no-overwrite publication;
5. exact key confirmation before actor/registry loading;
6. path-free status/list/retirement CLI output;
7. signing CLI private-path secrecy and generic signature failure output;
8. canonical runtime registry caching.

Focused compilation passed.

## Verification boundaries still open

- complete exact-current repository pytest and coverage;
- Ruff for the exact current source tree;
- full-tree compilation from an unchanged current checkout;
- Windows ACL and reparse-point behavior;
- Docker/Compose persistence and restart behavior;
- independent-process key registration/retirement and output-path contention;
- process-kill and filesystem-failure injection during signature publication;
- HSM/KMS/PKCS#11 signing adapters;
- external certificate, directory or transparency-log distribution of signer identities;
- external RFC 3161 interoperability fixtures from real TSA providers.

Ruff is unavailable in the reconstructed execution environment and is not claimed.

## Permanent non-claims

- A valid Ed25519 signature proves possession of the corresponding private key, not the scientific correctness of the evidence.
- A repository-owned public-key registry does not independently prove a real-world signer identity.
- RFC 3161 receipt binding is not trusted historical verification unless the TSA response is freshly verified through the governed trust profile.
- Private signing keys are not managed, generated, rotated or destroyed by this implementation.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
