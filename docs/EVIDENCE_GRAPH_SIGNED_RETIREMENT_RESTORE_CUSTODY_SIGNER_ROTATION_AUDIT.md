# Custody signer rotation audit

Last updated: 2026-08-03

This runbook covers read-only evaluation of the Ed25519 custody signer registry against an explicit operator rotation policy.

The audit never registers, retires, generates, loads private keys, signs manifests, or deletes records.

## 1. Run an audit

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signer_operations.py \
  --owner-id alice \
  --registry-db-path data/evidence_graph_set_signed_retirement_custody_signers.sqlite3 \
  --allowed-issuer lab-security \
  --maximum-active-keys 2 \
  --maximum-key-age-seconds 31536000 \
  --rotation-warning-seconds 2592000 \
  --minimum-overlap-seconds 604800
```

`--allowed-issuer` may be repeated. At least one issuer is required.

The policy is normalized and bound to a deterministic policy digest. The warning interval may not exceed the maximum key age.

## 2. Key classifications

Real registry records receive one classification:

- `active_current`;
- `active_rotation_due`;
- `active_expired`;
- `active_unapproved_issuer`;
- `retired`;
- `retired_unapproved_issuer`.

The audit never fabricates a key record. If the registry has no active key, the owner-wide action appears separately as:

```text
register_initial_key
```

This applies to both an empty registry and a registry containing only retired keys.

## 3. Suggested operator actions

Per-key actions may be:

- `register_successor`;
- `maintain_overlap`;
- `eligible_for_operator_retirement`;
- `reduce_active_key_count`;
- `investigate_unapproved_issuer`;
- `no_action`.

These are planning classifications only. They do not authorize or execute registration or retirement.

### Rotation due or expired newest key

When the newest active key reaches the warning or expiration threshold, the action is `register_successor`.

### Older key with a successor

An older key is `eligible_for_operator_retirement` only after the newest active key has been registered for at least the configured minimum overlap interval.

Before that interval, the action is `maintain_overlap`.

### Too many active keys

When active-key count exceeds policy, older approved keys are classified `reduce_active_key_count`. The report does not decide which key must be retired; operators must inspect issuer, age, deployment, and public-key distribution evidence.

### Unapproved issuer

An active or retired key whose issuer is absent from the explicit allowlist receives `investigate_unapproved_issuer`. The audit does not automatically retire or delete it.

## 4. Strict integrity

The rotation assessment validates:

- policy digest;
- owner scope;
- bounded query completeness;
- unique key IDs;
- exact active/retired counts;
- real-item classifications and actions;
- owner-wide global actions;
- deterministic assessment digest;
- non-mutation safety flags.

No synthetic key history is created to represent a missing active key.

## 5. Output privacy

The report includes only:

- key ID;
- issuer;
- public-key fingerprint;
- active/retired state;
- registration/retirement timestamps;
- age and overlap intervals;
- classifications and actions;
- policy and assessment digests.

It does not contain:

- private keys;
- PEM contents or paths;
- registration/retirement actor IDs;
- source text;
- signatures;
- mutation commands.

Every report states:

```text
registry_mutation_performed: false
key_material_mutation_performed: false
key_deletion_performed: false
source_text_returned: false
raw_path_returned: false
```

## 6. Relationship to signer administration

The audit is read-only. Use the governed signer-administration entrypoint for deliberate changes:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signers_governed.py register ...
python scripts/evidence_graph_set_signed_retirement_restore_custody_signers_governed.py retire ...
```

The administration path accepts only direct process environment/file actor bindings until signer administration receives a dedicated signed-assertion reservation journal.

## 7. Verification boundary

Repository-native contracts are committed for:

- deterministic policy normalization and tamper refusal;
- empty and fully retired registry global-action handling;
- age, warning, overlap, active-count, and issuer classifications;
- bounded-result refusal;
- query-only CLI output and no mutation.

These tests have not been executed together from a complete unchanged current checkout. Full pytest, coverage, Ruff, independent-process contention, Windows/container matrices, and measured rotation exercises remain open.

Release readiness is not claimed.
