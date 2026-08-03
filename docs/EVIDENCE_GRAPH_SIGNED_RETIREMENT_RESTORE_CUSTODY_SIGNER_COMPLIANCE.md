# Custody signer governance compliance and enforced signing

Last updated: 2026-08-03

This runbook detects signer records that bypassed canonical direct or signed-administration governance and prevents those active records from creating new signatures through the enforcement-aware path.

Cryptographic validity and governance validity are deliberately separate.

## 1. Compliance audit

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signer_compliance.py \
  --owner-id alice \
  --registry-db-path data/evidence_graph_set_signed_retirement_custody_signers.sqlite3 \
  --admin-use-db-path data/evidence_graph_set_signed_retirement_custody_signer_admin_uses.sqlite3
```

Both databases are opened query-only. The command does not initialize a missing database.

### Registration/retirement classifications

- `direct_compliant`: process environment or descriptor-file actor binding;
- `signed_committed_compliant`: exact committed one-operation reservation;
- `signed_reserved_incomplete`: reservation exists but did not commit;
- `signed_missing_reservation`: non-direct binding has no reservation;
- `signed_scope_mismatch`: binding digest exists but owner/action/key/action digest differs;
- `not_applicable`: active key has no retirement edge.

### New-signature eligibility

An active record is eligible only when registration classification is:

```text
direct_compliant
signed_committed_compliant
```

A reserved, missing, or scope-divergent signed registration is ineligible.

### Historical governance compliance

A retired record is governance-compliant only when both registration and retirement are direct-compliant or backed by committed reservations.

This status does not change whether an Ed25519 signature verifies mathematically.

## 2. Enforcement-aware signing

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signer_enforced.py sign-compliant \
  evidence/external-chain.json \
  --owner-id alice \
  --key-id custody-ed25519-2026-02 \
  --private-key-path custody-ed25519-2026-02-private.pem \
  --output evidence/external-chain.ed25519.json \
  --registry-db-path data/evidence_graph_set_signed_retirement_custody_signers.sqlite3 \
  --admin-use-db-path data/evidence_graph_set_signed_retirement_custody_signer_admin_uses.sqlite3
```

The command requires:

1. exactly one matching registry record;
2. active state;
3. compliant registration governance;
4. manifest owner equal to registry owner;
5. private-key-derived public fingerprint equal to registry fingerprint;
6. structurally verified custody manifest.

The registry and reservation databases remain read-only. The only mutation is atomic creation of the requested signed envelope.

## 3. Governance-aware historical verification

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signer_enforced.py verify-compliance \
  evidence/external-chain.ed25519.json \
  --owner-id alice \
  --key-id custody-ed25519-2026-01 \
  --public-key-path custody-ed25519-2026-01-public.pem
```

By default, a cryptographically valid historical envelope is reported even when its signer record is governance-noncompliant. Output includes both:

```text
signature_valid
governance_compliant_for_historical_verification
```

To fail closed on governance noncompliance:

```bash
--require-governance-compliance
```

This option is appropriate for automated acceptance gates. For forensic investigation, omit it so mathematically valid but governance-divergent evidence remains inspectable.

## 4. Compatibility risk detection

Historical compatibility scripts could create registry state without the newer one-operation reservation boundary. The compliance audit makes this visible instead of silently trusting the record.

Canonical production procedures are:

```text
direct administration:
  evidence_graph_set_signed_retirement_restore_custody_signers_governed.py

signed administration:
  evidence_graph_set_signed_retirement_restore_custody_signer_admin_governed.py

new signature creation:
  evidence_graph_set_signed_retirement_restore_custody_signer_enforced.py sign-compliant
```

## 5. Privacy and mutation boundary

Reports contain:

- key ID, issuer, public fingerprint, and state;
- binding methods and binding digests;
- reservation use IDs;
- classifications, eligibility, and report digest.

Reports omit:

- actor IDs;
- assertion bodies/signatures;
- private/public key paths;
- private-key material;
- source text.

Compliance audit and historical verification do not mutate the registry or reservation journal.

## 6. Verification boundary

Repository-native contracts are committed for:

- direct and committed-signed compliance;
- missing, reserved, and scope-mismatched reservations;
- registration plus retirement compliance;
- duplicate/bounded refusal and report digest validation;
- refusal of new signatures from noncompliant active records;
- compliant private-key/manifest checks;
- historical cryptographic verification with optional governance enforcement;
- path/private-material-free output.

These newest contracts have not been executed together from a complete unchanged current checkout. Independent-process bypass attempts, process-kill recovery, full pytest, coverage, Ruff, Windows, and container matrices remain open.

Release readiness is not claimed.
