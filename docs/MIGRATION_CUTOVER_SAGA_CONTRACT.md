# Adapter-only cutover compensation saga

Last updated: 2026-08-02

## Scope

`tools/migration_cutover_saga.py` defines the exact mutation and compensation order that any future production cutover adapter must satisfy. The repository does not yet provide such an adapter, does not expose this module through a CLI, and does not connect it to the ready-operation journal.

The contract exists to make publication and rollback semantics testable before any authoritative backend can be mutated.

## Required operation

The saga accepts only a `CutoverOperation` in `ready` state. The operation already binds:

- validated migration task;
- eligible paired promotion report;
- exact preflight;
- encrypted rollback artifact;
- isolated-staging verification;
- unchanged source-generation identity.

A ready operation is still revalidated through the backend adapter while the adapter's exclusive owner/document lock is held.

## Required backend adapter methods

A future adapter must implement:

- `exclusive_lock`;
- `current_identity`;
- `write_hidden_target`;
- `validate_hidden_target`;
- `commit_visibility`;
- `validate_visible_target`;
- `discard_hidden_target`;
- `restore_rollback`;
- `validate_rollback`.

No reflection, fallback method discovery or implicit backend selection is performed. The caller must explicitly supply an adapter implementing the complete protocol.

## Success order

Inside one adapter-provided exclusive lock:

1. acquire lock;
2. revalidate the complete current source identity;
3. write the target into a hidden/non-visible namespace;
4. require the exact expected target publication identity;
5. validate the hidden target;
6. atomically commit visibility;
7. validate the visible target.

The expected target identity binds the ready operation, target artifact digest, target profile, content hash and target vector/sparse row counts.

`commit_visibility` has a strict atomic contract: it must either return after visibility has changed or raise without changing visibility. Adapters that cannot satisfy that contract require an additional durable uncertainty/reconciliation protocol before they can be registered.

## Compensation order

All compensation executes under the same exclusive lock.

Before visibility:

- discard any hidden target;
- revalidate that the original source identity remains current;
- return a bounded `aborted` result.

After visibility:

- restore the encrypted rollback artifact;
- validate the restored rollback identity;
- return a bounded `rolled_back` result only after verification.

If compensation or rollback validation fails, the saga raises `CutoverRecoveryError` containing only bounded phase and exception-type identifiers. Private backend messages are not retained.

## Result semantics

`CutoverSagaResult` supports only:

- `published`;
- `aborted`;
- `rolled_back`.

It includes ordered phase identifiers, generic failure type, publication ID and rollback-verification flag. Its deterministic trace digest contains no private exception message.

The adapter-only module does not persist results or mutate migration/cutover journals. Durable execution states must be designed only after a real adapter and crash-recovery protocol exist.

## Focused verification

The constrained local saga harness passed **10 tests** covering:

- hidden validation before visibility;
- exact source revalidation;
- source-drift abort before target write;
- hidden-target discard after validation failure;
- wrong target identity discard;
- visible-validation failure followed by verified rollback;
- injected failure immediately after visibility commit;
- bounded recovery failure;
- lock-acquisition failure without unsafe recovery outside the lock;
- refusal of non-ready operations and incomplete adapters;
- deterministic path-free trace digests;
- compensation occurring before lock release.

One review-discovered defect was fixed before publication: the raw hidden-target handle is retained before identity validation, ensuring a target with the wrong identity can still be discarded instead of being orphaned.

## Remaining before a production adapter

- exact temporary restore tests using the real vector, sparse and generation implementations;
- visibility semantics for the current authoritative architecture;
- durable execution and recovery journal phases;
- backend-specific hidden target namespaces;
- atomic visibility handoff or explicit uncertainty reconciliation;
- authenticated rollback restoration from the encrypted artifact;
- post-publication and post-rollback exact snapshot validation;
- crash, process-kill, disk-full and backend-failure injection at each phase;
- exact-head Linux, Windows and container verification.

## Permanent non-claims

- The saga contract is not a production adapter.
- Fake-adapter success does not prove backend atomicity.
- A `published` result from an injected adapter is not connected to repository operator surfaces.
- No live cutover has been enabled or performed.
