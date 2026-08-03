# Wave 5 governed RFC 3161 trust verification boundary

Last updated: 2026-08-03

This ledger extends `WAVE5_RFC3161_VERIFICATION_BOUNDARY_2026-08-03.md` with owner-scoped external TSA trust-profile governance.

## Code boundary

Code and documentation head before this ledger commit:

```text
cd161fb74f9bf420c26a04716801f542daa2e1ff
```

New governance modules:

- `tools/evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust.py`
- `tools/evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust_runtime.py`
- `tools/evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust_cli.py`
- `scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust.py`

Focused contracts:

- `tests/unit/test_evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust.py`
- `tests/unit/test_evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust_cli.py`

Deployable configuration:

- `config/evidence_graph_custody_rfc3161_trust.env.example`

Operator runbook:

- `docs/EVIDENCE_GRAPH_RESTORE_CUSTODY_RFC3161_TRUST.md`

## Implemented trust scope

Each durable profile binds:

- owner and profile IDs;
- required RFC 3161 policy OID;
- exact root trust-anchor bundle digest;
- optional intermediate bundle digest;
- optional CRL bundle digest;
- optional allowed TSA signer-certificate fingerprints;
- explicit token-generation validity start and optional end;
- active or retired state;
- process-owned registration and retirement actor provenance;
- deterministic record digest.

The registry validates trust anchors as PEM CA certificates and stores no certificate body, private key or raw path.

## Lifecycle behavior

- Exact active registration replay is idempotent.
- Reusing a profile ID with changed scope is refused as a collision.
- Retirement is monotonic and retains historical verification metadata.
- Active profiles accept tokens only inside the configured validity window.
- Retired profiles accept historical tokens only up to the earlier of explicit validity end and retirement time.
- New trust scope requires a new profile ID.
- Database parent and inode replacement fail closed.
- Stored JSON is strictly reconstructed with duplicate-key and non-finite-value refusal.

## Governed verification behavior

Before invoking the base RFC 3161 verifier, the profile boundary requires exact equality for:

- owner;
- timestamp policy;
- trust-anchor bundle digest;
- intermediate bundle digest;
- CRL bundle digest.

After cryptographic verification, it requires:

- TSA signer certificate in the optional allowlist;
- token generation time inside the profile window;
- token generation time no later than profile retirement.

Only then may the digest-only RFC 3161 receipt be published.

## Executed focused evidence

Executed in the reconstructed workspace containing the exact split RFC 3161 and trust-profile implementations:

```text
python -m compileall -q \
  tools/evidence_graph_set_signed_retirement_restore_custody_rfc3161*.py \
  scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161*.py \
  tests/unit/test_rfc3161*.py

python -m pytest -q \
  tests/unit/test_rfc3161.py \
  tests/unit/test_rfc3161_cli.py \
  tests/unit/test_rfc3161_trust.py \
  tests/unit/test_rfc3161_trust_cli.py
```

Result:

```text
11 passed
```

Focused compilation passed. Ruff was unavailable and is not claimed.

## Real cryptographic integration

The test matrix includes two real local OpenSSL TSA integrations:

1. the base request/response/chain verification round trip;
2. the same response verified through a registered policy/root/signer profile.

The governed test also supplies a second unrelated CA and confirms trust-bundle substitution fails before token verification.

## Contract coverage

The eleven focused checks cover:

- RFC 3161 request bundle creation and DER emission;
- real granted response verification;
- wrong nonce and wrong trust anchor;
- rejected response;
- request-bundle tampering;
- no-overwrite request and receipt boundaries;
- generic path-free base CLI failures;
- exact governed policy/root/signer enforcement;
- idempotent registration and profile collision;
- profile retirement and database replacement;
- validity-window and signer-allowlist enforcement;
- confirmation before registry/actor loading;
- read-only path-free trust CLI output.

## Exact non-claims

A governed profile establishes that verification used the exact policy and certificate evidence registered by a process-owned actor.

It does not independently establish:

- institutional identity of that actor or TSA;
- correctness of the trust-anchor selection;
- TSA clock accuracy;
- hardware-backed key or clock operation;
- current revocation without governed CRL/OCSP evidence;
- authenticity of external request transport;
- scientific correctness of the custody chain.

The RFC 3161 receipt therefore continues to record:

```text
independently_trusted_clock_proven=false
hardware_clock_proven=false
```

## Remaining verification and governance

- exact-current complete repository pytest and coverage;
- Ruff and full-tree compilation from an unchanged checkout;
- production institutional TSA onboarding;
- out-of-band trust-anchor distribution controls;
- automated external profile rotation assessment;
- OCSP and long-term revocation evidence;
- governed online TSA transport and credentials;
- Windows, container and OpenSSL version matrices;
- malformed/fuzzed ASN.1 corpus;
- SQLite, filesystem and process-kill fault matrices for the trust registry;
- HSM/KMS/PKCS#11 and independently attested clock integration.

No exact-current CI success or release readiness is claimed.
