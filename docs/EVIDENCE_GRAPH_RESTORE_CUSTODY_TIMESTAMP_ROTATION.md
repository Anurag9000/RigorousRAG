# Custody timestamp-authority rotation audit

Last updated: 2026-08-03

This read-only audit evaluates registered custody timestamp-authority keys against an explicit rotation policy. It reports actions but never registers, retires, signs, deletes, or modifies a key.

## Command

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_operations.py \
  --owner-id alice \
  --registry-db-path data/evidence_graph_set_signed_retirement_custody_timestamp_authorities.sqlite3 \
  --maximum-active-key-age-seconds 31536000 \
  --minimum-overlap-seconds 604800 \
  --maximum-active-keys 2
```

The registry is opened through the query-only authority view. A missing or uninitialized registry fails closed rather than being created by the audit.

## Default policy

```text
maximum active-key age: 365 days
minimum successor overlap: 7 days
maximum active keys: 2
```

The exact policy is digest-bound into every report.

## Classifications

### `initial_key_required`

No active authority key exists. This includes a new registry and a registry in which every key has been retired.

Suggested action:

```text
register_initial_timestamp_authority_key
```

### `healthy_single_active`

Exactly one active key exists and its age is below the configured maximum. No action is emitted.

### `rotation_required_no_successor`

Exactly one active key exists and it has reached or exceeded the maximum active-key age.

Suggested action:

```text
register_successor_timestamp_authority_key
```

### `overlap_window_active`

At least two keys are active, the configured maximum active-key count is not exceeded, and the newest key has not yet completed the minimum overlap period.

Suggested action:

```text
retain_oldest_key_until_overlap_completes
```

### `retire_oldest_after_overlap`

At least two keys are active and the newest key has completed the overlap period.

Suggested action:

```text
retire_oldest_timestamp_authority_key
```

This is planning information only. Retirement still requires the governed authority-administration command and exact key-ID confirmation.

### `too_many_active_keys`

The active-key count exceeds the explicit policy ceiling.

Suggested action:

```text
review_and_retire_excess_active_keys
```

The report does not select arbitrary keys for deletion. It identifies the oldest and newest active authority/key pairs and leaves the governed decision to the operator.

## Report integrity

The deterministic report binds:

- owner ID;
- generation timestamp;
- policy digest;
- classification;
- active and retired counts;
- oldest active authority and key IDs;
- newest active authority and key IDs;
- overlap age;
- every authority/key fingerprint, state, registration timestamp, retirement timestamp, and current age;
- ordered actions.

Reconstruction refuses:

- duplicate authority/key identities;
- unsupported states;
- active rows containing retirement timestamps;
- retired rows missing retirement timestamps;
- retirement before registration;
- unordered or duplicate report items;
- count, overlap, oldest/newest, action, policy, or report-digest divergence;
- bounded-result saturation.

## Privacy and mutation boundary

Output may contain public authority IDs, key IDs, public-key fingerprints, states, and timestamps.

Output does not contain:

- actor IDs;
- actor assertions;
- private-key material;
- key or registry paths;
- custody evidence or source text.

The following flags remain false:

```text
registry_mutation_performed
key_material_mutation_performed
key_deletion_performed
attestation_created
raw_path_returned
```

## Verification boundary

Focused reconstructed execution passed:

```text
5 passed
```

The rotation tests cover all six classifications, deterministic ordering, report tamper refusal, duplicate and bounded-result refusal, invalid registry chronology, privacy flags, and query-only path-free CLI output.

Combined with the timestamp attestation and authority-lifecycle tests:

```text
13 passed
```

These are focused reconstructed tests, not a complete unchanged checkout of the current repository. Full pytest, coverage, Ruff, platform, container, independent-process, and fault-injection matrices remain open.

## Non-claims

- Rotation assessment is not registration or retirement authorization.
- A recommended action does not prove an operational incident.
- Authority-key rotation does not establish that an asserted clock was accurate.
- The custom authority attestation is not an RFC 3161 timestamp token.
- Release readiness is not claimed.
