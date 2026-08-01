# Capability Wave 2D — profile migration and isolated shadow validation

Last updated: 2026-08-01

## Scope

Wave 2D now implements inventory, immutable planning, a durable resumable task journal, target-profile retained-source execution, task-isolated vector/sparse shadow artifacts, and validation digests.

It still does **not** cut over a live document. There is no command that mutates the authoritative current-generation pointer from a shadow artifact. Live cutover remains blocked until rollback, measured-quality, exact-head fault-injection and release gates are complete.

## Control-plane components

- `tools/migration_types.py`
  - validated candidates and durable tasks;
  - bounded identifiers, exact integers, SHA-256 fingerprints and finite timestamps;
  - planned, running, validated, committed, failed and cancelled states.
- `tools/migration_planner.py`
  - compares current durable generations with a target embedding profile;
  - consults retained-document capability without returning source paths;
  - classifies ready, already-current, source-unavailable, deleted and registry-inspection-failed documents;
  - derives deterministic task IDs from owner, document, source sequence and source/target fingerprints.
- `tools/migration_journal.py`
  - immutable SQLite task identity;
  - idempotent seeding;
  - bounded attempts;
  - leases and renewal;
  - validation digest before any future commit;
  - generic failure types;
  - planned/failed cancellation only;
  - symlink/reparse and database-identity checks.
- `tools/migration_runtime.py`
  - path-keyed process-local journal factory.
- `tools/index_migration_cli.py`
  - inventory, seed, status and owner-verified cancel;
  - bounded JSON without retained paths.

## Explicit target-profile encoder boundary

`tools/embedding_adapters.py` defines the migration encoder contract.

### Standard SentenceTransformer-compatible profiles

Profiles whose registry definition has `requires_adapter=false` use a bounded lazy `SentenceTransformerEncoder`:

- passage prefixes are applied from the profile;
- output row count must match input passages;
- dimensions must match the profile when declared;
- every vector value must be finite;
- normalization follows the profile contract;
- batch size and passage count are bounded.

### Adapter-required profiles

Profiles such as Instructor, SPECTER2 and BGE-M3 are not silently treated as ordinary sentence encoders. They fail closed until an explicit adapter factory is registered for the canonical profile alias.

The adapter registry is process-local and rejects duplicate registration unless replacement is explicit. Returned adapters must expose the exact requested profile and `encode_passages`.

## Retained-source shadow builder

`tools/migration_shadow_builder.py` creates shadow rows from the retained source without touching live indexes.

1. Resolve the canonical target profile and require the journal fingerprint to match the current registry fingerprint.
2. Retrieve the retained-source capability privately from `DocumentStore`.
3. Validate the retained path under the configured upload root.
4. Reparse the file through `tools.ingestion.ingest_file`.
5. Reverify owner/source-byte document identity.
6. Reapply the complete final-index redaction boundary.
7. Require the reparsed document ID to equal the migration task document ID.
8. Build the same deterministic authoritative sparse fields used by live ingestion.
9. Encode those exact field texts under the target profile.
10. Independently validate row count, dimensions and finite numeric values even for custom adapters.
11. Produce one vector row for every sparse field with matching field ID and provenance.

The retained source path is never included in vector rows, sparse rows, manifests, journal rows or CLI output.

The current first executable slice deliberately aligns vector and sparse shadows one-to-one at the authoritative field level. A future cutover implementation may choose a richer chunk layout only after comparative retrieval evidence proves it better and records a new parser/build fingerprint.

## Isolated shadow artifact store

`tools/migration_shadow_store.py` writes one directory per migration task under `MIGRATION_SHADOW_ROOT`.

Each completed directory contains:

- `vectors.json`;
- `sparse.json`;
- `manifest.json`.

The store provides:

- manifest-last publication through a same-root staging directory;
- file and directory fsync where supported;
- Windows-safe directory-fsync fallback;
- exact vector and sparse SHA-256 digests;
- exact byte and row counts;
- source and target profile fingerprints;
- source generation sequence;
- content and parser fingerprints;
- strict JSON with duplicate-key and nonstandard-number refusal;
- bounded file size, row count, nesting, items and strings;
- regular-file, symlink/reparse and root-identity checks;
- task-isolated cleanup;
- idempotent reuse of byte-identical artifacts across retry timestamps;
- refusal when an existing task directory contains different artifacts.

A manifest validation digest binds the artifact identity. Creation time is retained for audit but excluded from the idempotent artifact comparison, allowing a retry to reuse the exact first artifact.

## Lease-aware shadow executor

`tools/migration_shadow_executor.py` validates both sides of the build window.

Before building, the current generation must still match:

- owner and document;
- active/restored state;
- source sequence;
- source profile fingerprint.

After artifact publication, the generation is read again and must still match the same source identity and content hash. A generation change during parsing/encoding fails the task rather than validating stale output.

The executor then records the artifact validation digest through the task journal. Failures persist only a generic failure class.

A validated task may replay only if its journal digest and existing artifact digest still match. The builder does not run again.

## Build-worker claim separation

The journal's generic claim path includes validated tasks because future cutover workers need them. Shadow-build workers require a different queue.

`tools/migration_shadow_runtime.py` provides an atomic shadow-build claim that selects only:

- planned tasks;
- failed tasks below the attempt ceiling;
- expired running tasks below the attempt ceiling.

Validated tasks are excluded, preventing them from starving unbuilt tasks.

`execute_next_shadow_build` claims one task, constructs the retained-source builder, uses the current generation store, writes/validates isolated artifacts and leaves successful tasks in `validated` state.

## No-cutover operator CLI

```bash
python -m tools.migration_shadow_cli execute-one \
  --owner-id alice \
  --worker-id migration-builder-1 \
  --lease-seconds 300 \
  --max-attempts 3

python -m tools.migration_shadow_cli validate <task-id>

python -m tools.migration_shadow_cli remove <task-id> \
  --confirm-task-id <same-task-id>
```

The CLI intentionally has no cutover command.

- `execute-one` claims only buildable work.
- `validate` compares one existing artifact with the journal task and digest.
- `remove` is allowed only for failed or cancelled tasks under exact confirmation.
- Output contains task IDs, states, digests, counts and fingerprints, never retained paths or source text.

## Configuration

```dotenv
MIGRATION_JOURNAL_DB_PATH=data/index_migrations.sqlite3
MIGRATION_SHADOW_ROOT=data/migration_shadows
```

Embedding profiles are governed by `EMBEDDING_PROFILE` and `EMBEDDING_PROFILES_JSON`. Adapter-required target profiles additionally require explicit process-local adapter registration.

## State machine

1. `planned` — immutable task seeded from a current source generation.
2. `running` — one shadow or future cutover worker owns an unexpired lease.
3. `validated` — isolated shadow output exists and its digest is recorded; no live state has changed.
4. `committed` — reserved for a future atomic cutover implementation.
5. `failed` — generic failure type recorded and retryable within the attempt policy.
6. `cancelled` — operator-cancelled before active execution.

Expired running tasks may be reclaimed. Validated tasks preserve their digest for a future cutover worker.

## Focused contracts

Committed contracts cover:

- profile drift, aliases and stable task IDs;
- journal idempotency, leases, retries and cancellation;
- explicit adapter-required profile refusal and registration;
- profile passage-prefix and normalization settings;
- row-count, dimension and finite-value checks;
- retained-source capability/path/document-identity validation;
- one-to-one vector/sparse field provenance;
- absence of retained paths from artifacts;
- manifest-last publication and exact digest/count checks;
- retry reuse across different creation times;
- changed-artifact and tamper refusal;
- non-finite, empty, redirected and replaced-root refusal;
- source generation checks before and after build;
- worker lease ownership;
- validated artifact replay without rebuilding;
- generic failure recording;
- shadow-build claiming that excludes validated tasks;
- attempt ceilings and expired-running recovery;
- no-cutover CLI validation and removal restrictions.

## Remaining Wave 2D work

- Run the new store, builder, adapter, runtime and CLI contracts on a clean exact head.
- Add retrieval-quality and provenance benchmark gates comparing current and shadow profiles.
- Measure build latency, memory, artifact size and monetary/token cost where applicable.
- Persist experiment/promotion metadata associated with each validated artifact.
- Implement an atomic current-generation cutover that can publish shadow vector and sparse rows without a mixed generation window.
- Retain exact rollback references to the prior vector, sparse and generation state.
- Add bounded shadow retention and cleanup policy.
- Inject crashes before/after every artifact, journal, vector, sparse and pointer transition.
- Add pause/resume/cancel semantics for actively leased build workers.
- Add specialized governed adapters for adapter-required profiles only after their model/runtime contracts are tested.

## Verification boundary

Source and focused contracts are committed. The execution container cannot resolve GitHub, so no clean clone or complete exact-head Linux/Windows/container matrix can be run here. Live cutover and release readiness are not claimed.
