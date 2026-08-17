# Disaster-recovery rehearsal contract

`orchestration/disaster_recovery_rehearsal.py` is the source-level rehearsal layer above the repository's existing backup, custody, restore, migration and rollback machinery.

## Scope

The workflow answers one operational question without touching production routing: **can the governed recovery points for the required components be restored into an isolated target, verified, and removed while meeting declared RPO/RTO objectives?**

It does not create backups, select an ungoverned snapshot, promote a rehearsal target, or perform a production cutover. Production restoration remains owned by the existing governed restore/cutover paths.

## Immutable drill identity

A `RecoveryRehearsalSpec` binds:

- owner identity;
- simulated incident timestamp;
- maximum RPO and RTO;
- one recovery point per component;
- exact backup-manifest SHA-256;
- source-data watermark for RPO calculation;
- custody-evidence SHA-256;
- recovery-policy SHA-256.

The canonical specification is content-addressed as `drill_id`. A different specification cannot reuse that identity.

## Durable progression and crash replay

`SQLiteRecoveryRehearsalStore` journals the state machine and a monotonic fencing token. Each acquisition increments the token, including a reclaim by the same worker identity, so an older handle cannot mutate state after a newer claim.

State and fencing validation occur inside the same `BEGIN IMMEDIATE` transaction as the revision CAS. External side effects are preceded by durable request states and receive deterministic idempotency keys. A retry therefore reissues the same logical operation rather than guessing whether the previous call crossed the side-effect boundary.

The persisted flow is:

`planned -> prepare_requested -> restoring -> restore_requested -> verify_requested -> ... -> cleanup_requested -> completed`

For multi-component drills, `restore_requested` and `verify_requested` repeat in canonical component order.

## Backend boundary

`RecoveryRehearsalBackend` exposes only four capabilities:

1. prepare an isolated rehearsal target;
2. restore one governed recovery point;
3. verify one recovered component;
4. remove the rehearsal target.

There is intentionally no publication, alias swap, route update or promotion method in this interface.

## Local-file adapter

`LocalFileRecoveryRehearsalBackend` adapts `tools.disaster_recovery` for offline local drills. It:

- confines all targets below one real, non-symlink isolation root;
- rejects path-like component identifiers;
- binds registered backup assets to exact custody evidence and manifest identity;
- verifies source backup checksums before restoration;
- restores only into component subdirectories of the drill root;
- requires the recovered directory to contain exactly the manifest population, with no symlinks, directories or stale unmanifested files;
- derives a content digest over the exact recovered inventory and readiness result;
- refuses redirected cleanup targets;
- proves that the isolated drill root was removed.

## RPO/RTO evidence

For each component, RPO is the simulated incident timestamp minus the governed source watermark. The drill's observed RPO is the worst component value.

Observed RTO is the simulated incident timestamp to the latest component verification timestamp. Cleanup occurs after readiness verification and before the terminal receipt is issued.

The final `RehearsalReceipt` includes:

- drill and owner identity;
- declared objective;
- exact component verification evidence;
- worst observed RPO;
- observed RTO;
- cleanup evidence;
- objective result and closed reason codes;
- a SHA-256 over the complete unsigned receipt payload.

A persisted receipt is re-hashed on read; a digest or identity mismatch is rejected.

## Privacy and safety

Durable backend errors are represented as exception class plus SHA-256 of the message, not raw exception text. This avoids journaling local paths, credentials or provider error bodies by accident.

Rehearsal completion never implies production recoverability was measured in this source-only audit. A real drill still requires real backup artifacts, storage/KMS/provider credentials, execution of restore verification, and observed timing under the deployment environment.
