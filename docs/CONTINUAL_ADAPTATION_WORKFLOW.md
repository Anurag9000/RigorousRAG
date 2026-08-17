# Continual drift-to-promotion workflow

`orchestration.continual_adaptation` binds existing RigorousRAG drift, training-lineage, benchmark and promotion primitives into a durable workflow. It does not download data or models and does not start a trainer, benchmark process or scheduler on import.

## State machine

A workflow is content-addressed by owner, baseline/candidate versions, drift evidence, adaptation policy, drift decision, exact training request, feedback batch, benchmark contract and continual-promotion policy.

The durable stages are:

`detected -> build_requested -> build_ready -> benchmark_requested -> benchmark_ready -> decision_ready -> promoted|held`

A `stable` drift decision terminates as `stable_held` without issuing a build. A promoted workflow may later append an independent `rolled_back` transition. Authority/evidence mismatches terminate as `failed` with a bounded failure type and terminal receipt digest.

## Crash and retry semantics

Side-effecting work is always preceded by a durable request stage. The build backend sees the stable workflow ID after `build_requested`; benchmark execution sees the same ID after `benchmark_requested`. A process crash therefore leaves an unambiguous request to retry. Implementations of the injected build and benchmark protocols must make repeated calls with the same workflow ID idempotent.

Unexpected backend errors are not silently converted into a permanent failure. The requested state remains durable so another executor can retry. Permanent identity/evidence mismatches fail closed.

## Evidence binding

A build result becomes a `TrainingLineage` over the exact parent artifact, dataset digest, code revision, seed and training configuration. Benchmark evidence must name that exact candidate artifact and bind a verified dataset manifest, experiment digest, benchmark receipt and base promotion decision.

The base promotion decision must match the workflow owner, feedback batch, baseline version and candidate version. Continual safeguards then add drift, forgetting, forward-transfer, replay-privacy and rollback-readiness gates.

Promotion is allowed only for an eligible continual decision. The publication backend must implement expected-baseline compare-and-swap semantics and return a receipt naming the exact candidate artifact and governed decision. Rollback is a separate expected-current operation and must restore the governed baseline while referencing the original promotion publication.

## Concurrency and privacy

The SQLite journal uses expiring executor leases with monotonic fencing tokens and revision compare-and-swap. Injected mutation backends receive the fencing token so stale processes can be rejected at external side-effect boundaries as well.

The journal stores only identifiers, digests, bounded metrics/decision payloads, workflow state and receipts. It does not store raw prompts, documents, training examples or model tensors.

## Relationship to other modules

- `tools.index_drift` decides whether adaptation is needed.
- `tools.training_lineage` binds candidate build provenance and replay identity.
- verified benchmark dataset/evaluation modules provide benchmark evidence.
- `tools.feedback_promotion` supplies base quality/latency/cost gates.
- `tools.continual_promotion` adds forgetting/transfer/replay/rollback safeguards.
- `tools.promotion_journal` remains the tamper-evident model-promotion audit journal.
- maintained-target population reconciliation can converge physical retrieval targets after a promoted retrieval profile/index becomes desired state.

Actual training, dataset acquisition, model execution and benchmark execution remain explicit runtime operations outside this source-only implementation sweep.