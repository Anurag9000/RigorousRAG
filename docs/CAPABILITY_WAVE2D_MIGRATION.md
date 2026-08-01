# Capability Wave 2D — isolated profile migration and governed promotion

Last updated: 2026-08-02

## Scope

Wave 2D now implements:

- profile-drift inventory and immutable migration planning;
- a durable leased migration-task journal;
- target-profile retained-source reparse and privacy finalization;
- explicit embedding-adapter governance;
- task-isolated manifest-last vector/sparse shadow artifacts;
- exact shadow structural and source-generation validation;
- a repository-owned paired benchmark producer;
- aggregate quality/resource promotion gates;
- paired confidence-interval non-inferiority and optional practical-effect gates;
- append-only promotion-report history.

It still does **not** cut over a live document. No command replaces authoritative vector or sparse state, changes the durable current-generation pointer or performs rollback. Live cutover remains blocked until atomic publication, rollback, measured-resource and exact-head fault-injection contracts exist.

## Planning and journal

The migration control plane consists of:

- `tools/migration_types.py` — validated candidates/tasks and bounded value types;
- `tools/migration_planner.py` — profile-drift inventory and deterministic task IDs;
- `tools/migration_journal.py` — idempotent SQLite tasks, leases, retries, validation digests and generic failures;
- `tools/migration_runtime.py` — path-scoped journal factory;
- `tools/index_migration_cli.py` and `scripts/index_migrations.py` — inventory, seed, status and owner-verified cancellation.

Task identities bind owner, document, source-generation sequence and source/target profile fingerprints. Retained-source paths are not stored in the journal or returned by the operator surface.

## Explicit target-profile encoder boundary

`tools/embedding_adapters.py` defines the migration encoder contract.

Profiles with `requires_adapter=false` may use the bounded lazy SentenceTransformer-compatible adapter. It applies passage prefixes, honors declared normalization, checks output row count and profile dimensions, and rejects booleans, NaN and infinity.

Instructor, SPECTER2, BGE-M3 and any other `requires_adapter=true` profile fail closed until an explicit factory is registered for the canonical profile alias. Returned adapters must expose the exact requested profile and a compatible `encode_passages` method.

## Retained-source shadow construction

`tools/migration_shadow_builder.py`:

1. resolves the target profile and verifies its fingerprint against the task;
2. privately resolves the retained-source capability under the configured owner upload root;
3. reparses through the current ingestion pipeline;
4. revalidates owner/source-byte document identity;
5. reapplies the complete final-index privacy boundary;
6. requires the reparsed document ID to equal the task document ID;
7. builds the same deterministic sparse fields used by authoritative ingestion;
8. encodes those exact field texts under the target profile;
9. independently validates row count, dimensions and finite values;
10. creates one vector row per sparse field with matching field/page/section provenance.

Retained paths and raw source bytes are not serialized into shadow rows, manifests, journal tasks or CLI results.

## Isolated shadow artifacts

`tools/migration_shadow_store.py` writes one task directory under `MIGRATION_SHADOW_ROOT` containing:

- `vectors.json`;
- `sparse.json`;
- `manifest.json`.

The store provides manifest-last publication, staging on the same filesystem, fsync where supported, exact hashes/bytes/counts, source and target fingerprints, source sequence, content/parser fingerprints, strict JSON, bounded structures, regular-file and root-identity checks, idempotent replay of identical artifacts and refusal of changed/tampered artifacts.

`tools/migration_shadow_executor.py` checks the source generation before and after construction. The live generation must remain active/restored with the exact source sequence, source profile and content hash. Only then is the shadow validation digest recorded.

`tools/migration_shadow_runtime.py` separates build claims from future cutover claims. Build workers select only planned, retryable failed or expired-running tasks; validated tasks cannot starve unbuilt work.

## Shadow operator surface

```bash
python -m tools.migration_shadow_cli execute-one \
  --owner-id alice \
  --worker-id migration-builder-1

python -m tools.migration_shadow_cli validate <task-id>

python -m tools.migration_shadow_cli remove <task-id> \
  --confirm-task-id <same-task-id>
```

Removal is limited to failed or cancelled tasks. There is no cutover action.

## Repository-owned paired benchmark

`tools/migration_benchmark.py` accepts strict paired current/shadow fixtures containing query identifiers, relevance identifiers, ranked identifiers, aggregate support/citation observations, abstention outcomes and resource observations.

It computes:

- recall@k;
- nDCG@k;
- MRR;
- support recall;
- citation precision;
- abstention accuracy;
- conservative p95/max resource aggregates and mean estimated cost;
- repeated-run and distinct-seed counts;
- signed paired 95% delta intervals.

All runs must use the same ordered query contract. The benchmark fingerprint identifies that contract, not the current/shadow outputs or resources. Raw query text, answer text, passages, retained paths and provider responses are unsupported.

Operator commands:

```bash
python -m tools.migration_benchmark_cli inspect \
  --fixture-file paired_fixture.json

python -m tools.migration_benchmark_cli run \
  --fixture-file paired_fixture.json \
  --evidence-output promotion_evidence.json \
  --report-output benchmark_intervals.json
```

The equivalent wrapper is `scripts/migration_benchmarks.py`.

## Aggregate promotion gate

`tools/migration_promotion.py` implements the versioned `conservative-v1` gate. It requires exact task/journal/manifest/evidence/source-generation alignment, benchmark minimums, equal vector/sparse counts by default, quality floors, maximum point regressions and bounded latency/memory/storage/estimated-cost ratios.

`tools/migration_promotion_store.py` persists immutable reports by digest and atomically advances a per-task current pointer. Reports contain no paths, raw queries or passages.

## Paired statistical gate

`tools/migration_statistical_gate.py` implements `paired-noninferiority-v1`:

- minimum repeated-run, seed-count and confidence-level requirements;
- lower-bound non-inferiority for recall, nDCG, MRR, support recall, citation precision and abstention accuracy;
- optional lower-bound practical-gain thresholds;
- deterministic assessment and policy digests;
- composite promotion evidence/policy digests without changing the stored report schema.

The default non-inferiority margins are 0.01 for every metric except citation precision, which allows no regression.

## Recommended promotion workflow

```bash
python -m tools.migration_promotion_cli evaluate-fixture <task-id> \
  --fixture-file paired_fixture.json \
  --policy-file reviewed_promotion_policy.json \
  --statistical-policy-file reviewed_statistical_policy.json
```

This command generates paired evidence, applies aggregate and statistical gates, persists the final append-only report and returns the paired interval assessment in one process.

Compatibility with externally produced strict aggregate evidence remains available:

```bash
python -m tools.migration_promotion_cli evaluate <task-id> \
  --evidence-file promotion_evidence.json
```

That compatibility path cannot apply paired interval gates because it does not contain per-run deltas.

Report inspection and bounded cleanup:

```bash
python -m tools.migration_promotion_cli status <task-id>
python -m tools.migration_promotion_cli history <task-id> --limit 100
python -m tools.migration_promotion_cli remove-task <task-id> \
  --confirm-task-id <same-task-id>
```

The equivalent wrapper is `scripts/migration_promotions.py`. Report cleanup is limited to failed or cancelled migration tasks.

## Configuration

```dotenv
INDEX_MIGRATION_DB_PATH=data/index_migrations.sqlite3
MIGRATION_SHADOW_ROOT=data/migration_shadows
MIGRATION_PROMOTION_ROOT=data/migration_promotions
```

Embedding profiles are governed by `EMBEDDING_PROFILE` and `EMBEDDING_PROFILES_JSON`. Adapter-required profiles also require explicit runtime adapter registration.

## State and decision semantics

Migration task states remain:

1. `planned`;
2. `running`;
3. `validated` — isolated shadow exists; live state is unchanged;
4. `committed` — reserved for future atomic cutover;
5. `failed`;
6. `cancelled`.

Promotion reports separately use `eligible` or `blocked`. An eligible report is evidence for a future cutover precondition; it is not authorization and performs no mutation.

## Focused verification

Committed contracts cover:

- profile drift, aliases, task identities, leases, retries and cancellation;
- adapter-required refusal/registration and standard profile encoding;
- retained-source capability, path and document identity checks;
- one-to-one vector/sparse provenance;
- manifest-last publication, exact counts/digests and idempotent replay;
- tamper, non-finite, symlink/reparse and root-replacement refusal;
- source-generation races and generic failures;
- build-only claim separation and no-cutover CLI behavior;
- strict paired benchmark contracts and metric computation;
- contract-only benchmark fingerprints;
- repeated runs, seeds and signed paired intervals;
- aggregate quality/resource gates;
- paired lower-bound non-inferiority and practical-gain gates;
- deterministic composite digests;
- append-only report history and atomic current pointer;
- direct in-process fixture evaluation and privacy-safe outputs.

The constrained local promotion/benchmark/statistical harness passed **42 tests**. Earlier focused shadow, adapter and lifecycle suites also passed in their isolated harnesses. This is not the complete exact-head repository matrix.

## Remaining Wave 2D work

- Execute governed fixtures against the real current and shadow retrieval stacks rather than consuming already collected ranked identifiers.
- Measure wall-clock latency, process/device memory, artifact storage and provider billing where applicable.
- Add reviewed bootstrap/permutation procedures and multiple-comparison controls where scientifically appropriate.
- Implement an atomic vector+sparse+generation publication with no mixed-generation window.
- Persist exact rollback references and verify rollback before releasing old state.
- Add cutover/rollback leases, idempotency and crash recovery.
- Add bounded shadow/report retention and compaction.
- Add pause/resume/cancel semantics for active workers.
- Add specialized governed adapters for adapter-required profiles.
- Inject faults before and after every build, report, vector, sparse, pointer and rollback transition.
- Run clean exact-head Linux, Windows and container verification.

## Verification boundary

The execution container cannot resolve GitHub, so no clean clone or complete exact-head matrix can be run here. Live cutover and release readiness are not claimed.
