# Maintained target population reconciliation

`orchestration.target_population_reconciliation` is the fleet-level control plane above the existing per-operation blue/green population journal and cutover saga.

## Scope

For one owner at a time it compares the desired maintained target set with a physical inventory across dense, sparse, lexical, late-interaction, graph and multimodal target families. Desired state is bound to generation, retrieval/model profile, schema, source-corpus digest and expected population count.

The planner classifies exact/healthy populations, missing targets, in-flight builds, generation/profile/schema/source/count drift, failed populations, stale/missing aliases and old unselected populations. Plans are deterministic and content-addressed.

## Mutation safety

A reconciliation run may perform only three control-plane actions:

1. submit a deterministic staged population under an idempotency key;
2. bind an alias to an already-existing **exact ready** population using expected physical ID + alias revision compare-and-swap under a fencing token; or
3. record an orphan candidate after re-reading its physical observation and proving that it is not live, protected or in flight.

Population submission never changes live routing. A new population must be observed on a later reconciliation pass as exact and ready before alias cutover is possible.

The reconciler never deletes physical data. Actual collection retirement/deletion remains in the retention/legal-hold governed lifecycle and requires its own authorization/confirmation path.

## Concurrency and isolation

Snapshots are owner-scoped and reject cross-owner rows. Mutating backends must assert the caller's monotonic fencing token before each action. Alias cutover re-reads the physical target and live alias before CAS; orphan recording re-reads the target plus live aliases/protection status immediately before recording the candidate.

`PopulationReconciliationJob` adapts this logic to `orchestration.periodic_reconciliation`. Work is bounded by `max_actions`; if additional actions remain it returns a continuation signal so operators/metrics can see that convergence is incomplete. A later run recomputes state rather than trusting a stale mutation plan.

## Relationship to existing migration code

This module does not replace `tools.migration_target_population` or `tools.migration_cutover_durable_blue_green`.

- the migration journal owns durable intent, receipts, executor leases and crash recovery for an individual cutover operation;
- the durable blue/green adapter performs the actual hidden population, exact readback, sparse/generation publication, route visibility, rollback and recovery;
- maintained-target reconciliation discovers which operations/populations are needed across the currently configured target fleet.

No database, vector engine, scheduler thread or network client is started by this module.