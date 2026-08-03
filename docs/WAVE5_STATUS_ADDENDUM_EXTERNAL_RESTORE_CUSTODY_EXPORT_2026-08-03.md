# Wave 5 status addendum — external restore chain of custody

Last updated: 2026-08-03

## Implemented

- [x] Query-only restore-intent reader.
- [x] Complete-chain-only export eligibility.
- [x] Live restored-target verification.
- [x] Live pre-restore backup/receipt verification.
- [x] Live post-restore comparison-receipt verification.
- [x] Exact post-bound custody-manifest requirement.
- [x] Completed artifact-pair requirement.
- [x] Exact live backup/pre-receipt path-digest binding.
- [x] Strict artifact/pre-bound/restore/post-bound/export chronology.
- [x] Active/inactive/not-checked legal-hold status.
- [x] Raw actor-ID reduction to SHA-256 digests.
- [x] Raw-path- and source-text-free external payload.
- [x] Deterministic chain digest and strict reconstruction.
- [x] Descriptor-safe bounded offline verification.
- [x] Atomic no-overwrite manifest publication.
- [x] Optional HMAC-SHA256 authenticated envelope.
- [x] Explicit key ID and constant-time authentication comparison.
- [x] Minimum 32-byte protected key file.
- [x] POSIX group/world permission refusal.
- [x] No import, restore, overwrite, merge, delete, or journal-mutation command.

## Operator commands

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py export ...
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py verify MANIFEST
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py authenticate MANIFEST ...
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py verify-authenticated ENVELOPE ...
```

## Focused verification

A reconstructed functional harness passed:

```text
4 focused checks passed
```

Covered:

1. complete-chain construction and privacy reduction;
2. incomplete, stale, and live-path-divergent refusal;
3. atomic export, no-overwrite behavior, and digest tamper refusal;
4. HMAC round trip, wrong-key refusal, and key-ID pinning.

Repository-native tests additionally cover weak keys, broad key permissions, duplicate JSON keys, offline CLI isolation, and path/actor/key-secret-free summaries. Those contracts have not been executed together from a complete unchanged current checkout.

The reconstructed copy used for execution was condensed. One initial local constructor error came from that condensation; the committed repository implementation already used explicit constructor fields and did not contain the local defect.

## Still open

- [ ] Complete exact-current repository pytest and coverage.
- [ ] Ruff and full-tree compilation from an unchanged current checkout.
- [ ] Independent-process export and artifact-replacement races.
- [ ] Process-kill, filesystem-full, fsync, and permission fault injection.
- [ ] Windows and container export matrices.
- [ ] Public-key asymmetric signatures.
- [ ] Trusted timestamps and signer key rotation.
- [ ] Hardware-backed key custody.
- [ ] External transparency publication.

## Permanent non-claims

- A chain digest proves structural integrity, not authorship.
- HMAC proves possession of a shared secret, not public non-repudiation.
- Legal-hold status in a manifest does not authorize release or deletion.
- External custody export cannot import or mutate repository state.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
