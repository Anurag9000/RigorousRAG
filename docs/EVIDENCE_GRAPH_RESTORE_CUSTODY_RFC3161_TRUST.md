# Governed external TSA trust profiles for RFC 3161 custody evidence

Last updated: 2026-08-03

The base RFC 3161 verifier accepts explicit trust-anchor, intermediate and CRL files. This control plane binds those files to an owner-scoped governance record so an operator cannot silently substitute a different certificate chain or timestamp policy during verification.

The registry stores no certificate body, private key, raw path or timestamp response. It stores SHA-256 digests and bounded governance metadata only.

## Profile scope

Each profile binds:

- owner ID;
- stable profile ID;
- required RFC 3161 policy OID;
- exact trust-anchor bundle SHA-256;
- optional intermediate-certificate bundle SHA-256;
- optional CRL bundle SHA-256;
- zero or more permitted TSA signer-certificate SHA-256 fingerprints;
- token generation-time validity start and optional end;
- process-owned registration actor and binding digest;
- active or retired lifecycle;
- process-owned retirement actor and retirement time;
- deterministic record digest.

When the signer allowlist is empty, any TSA signer certificate accepted by the pinned chain and policy may verify. Supplying one or more signer fingerprints narrows the profile to those exact certificates.

## Storage configuration

```bash
EVIDENCE_GRAPH_RESTORE_CUSTODY_RFC3161_TRUST_DB_PATH=\
data/evidence_graph_restore_custody_rfc3161_trust.sqlite3
```

A standalone example is available at:

```text
config/evidence_graph_custody_rfc3161_trust.env.example
```

Registration and retirement use the repository's existing process-owned review actor boundary. Exactly one actor source must be configured.

## 1. Vet the TSA evidence out of band

Before registration, independently obtain and review:

- the intended institutional TSA identity;
- its timestamp policy OID;
- trust-anchor certificates;
- intermediate certificates where required;
- current governed CRL evidence where required;
- intended TSA signer-certificate fingerprints;
- activation and retirement dates;
- certificate and policy documentation.

The registry does not perform institutional accreditation or discover trust anchors from the network.

## 2. Register an active profile

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust.py \
  register \
  --owner-id alice \
  --profile-id institutional-tsa-2026 \
  --confirm-profile-id institutional-tsa-2026 \
  --policy-oid 1.2.3.4.1 \
  --trust-anchor-bundle config/institutional-tsa-roots.pem \
  --untrusted-bundle config/institutional-tsa-intermediates.pem \
  --crl-bundle config/institutional-tsa-crls.pem \
  --allowed-signer-sha256 SIGNER_CERT_SHA256 \
  --valid-from 1785715200 \
  --valid-until 1817251200
```

`--valid-until` is optional. Additional signer fingerprints may be supplied by repeating `--allowed-signer-sha256`.

Registration is idempotent only for the exact same profile record. Reusing a profile ID with a changed policy, certificate digest, allowlist or validity window fails as an identity collision. A changed trust scope therefore requires a new profile ID.

The trust-anchor bundle is parsed as PEM certificates and every included trust anchor must carry CA basic constraints.

## 3. Inspect profiles

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust.py \
  status \
  --owner-id alice \
  --profile-id institutional-tsa-2026
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust.py \
  list \
  --owner-id alice \
  --state active
```

Status and list output contain digests and lifecycle metadata only. They do not load actor credentials or expose certificate paths.

## 4. Verify with a governed profile

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust.py \
  verify-response \
  --owner-id alice \
  --profile-id institutional-tsa-2026 \
  --request-bundle data/restore-custody.rfc3161-request.json \
  --response data/restore-custody.tsr \
  --trust-anchor-bundle config/institutional-tsa-roots.pem \
  --untrusted-bundle config/institutional-tsa-intermediates.pem \
  --crl-bundle config/institutional-tsa-crls.pem \
  --output-receipt data/restore-custody.rfc3161-receipt.json
```

Before invoking the cryptographic verifier, the governed boundary requires exact equality between supplied bundle digests and the registered profile. It then requires:

- request owner equal to profile owner;
- token policy equal to profile policy;
- signer certificate in the optional allowlist;
- token generation time within the profile validity window;
- token generation time no later than profile retirement when retired;
- all base RFC 3161 verification checks.

A retired profile remains usable only for historical tokens generated inside its recorded window. It does not authorize tokens generated after retirement.

## 5. Retire a profile

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust.py \
  retire \
  --owner-id alice \
  --profile-id institutional-tsa-2026 \
  --confirm-profile-id institutional-tsa-2026
```

Retirement is monotonic and records the process-owned actor binding. It does not delete the profile and does not invalidate historical tokens that fall inside the profile's recorded period.

A replacement chain, policy or signer set should be registered under a distinct profile ID with an explicit overlap window where operational policy permits.

## Rotation procedure

For a planned TSA certificate-chain rotation:

1. register the replacement profile under a new profile ID;
2. set the new profile's `valid_from` to the approved activation time;
3. retain the older profile through the governed overlap period;
4. verify new tokens against the new profile;
5. retire the older profile after the overlap closes;
6. retain both profile records for historical verification;
7. preserve external evidence explaining institutional identity and rotation approval.

No automatic profile selection or network discovery is performed.

## Failure behavior

The registry or governed verifier refuses:

- missing process-owned actor identity for mutation;
- wrong profile confirmation;
- malformed profile or policy OID;
- invalid or non-CA trust anchors;
- redirected, replaced or oversized certificate evidence;
- duplicate or excessive signer allowlists;
- reversed validity windows;
- profile-ID collisions;
- owner-scope drift;
- changed trust, intermediate or CRL bundle digest;
- wrong token policy;
- disallowed TSA signer certificate;
- token time outside the profile window;
- token time after profile retirement;
- database parent or inode replacement;
- tampered stored record JSON;
- any failure from the base RFC 3161 verifier;
- existing receipt output.

CLI failures are generic and path-free.

## Exact trust boundary

A governed profile proves that verification used the exact certificate/policy evidence previously registered by a process-owned actor.

It does not prove by itself:

- that the actor correctly identified the institution;
- that the root belongs to the intended TSA;
- that the TSA clock is accurate or independently audited;
- that revocation evidence is current unless separately governed;
- that the TSA uses hardware-backed keys or clocks;
- that a profile rotation was institutionally approved outside this registry.

Out-of-band identity vetting, public certificate transparency, institutional policy evidence, OCSP/CRL lifecycle, HSM evidence and trusted-clock attestation remain distinct controls.

## Focused verification

The focused registry slice passed:

```text
5 passed
```

It covers a real OpenSSL TSA response under a registered profile, exact trust/policy/signer enforcement, wrong-root refusal, deterministic registration replay, profile collision, retirement, database replacement, validity windows, signer allowlists, confirmation-before-runtime loading and read-only path-free CLI output.

Combined with the base RFC 3161 slice:

```text
11 focused tests passed
```

This is not a full exact-current repository or platform matrix. Ruff, full pytest, production institutional TSA onboarding, Windows, containers, OCSP and HSM-backed operation remain open.
