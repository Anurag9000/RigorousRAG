# RigorousRAG strict source-implementation audit — 2026-08-17

## Authority and boundary

This document is the stricter follow-up to `docs/SOURCE_IMPLEMENTATION_AUDIT_2026-08-17.md`.

The implementation baseline audited here is commit `33d0b2f9e0e8ec9f7fd6ccfdddbd44719239b282`. This document itself is a documentation-only commit after that source baseline.

The audit boundary requested by the user is intentionally unusual and must be stated precisely:

**Included:** all source code needed for architectures, algorithms, methodology, model/training design, losses, data/collation, hard-negative mining, checkpoints/resume/stage saves, evaluation/statistics, routing/planning, extraction, citation/evidence semantics, multimodal support, governance, persistence, reconciliation, promotion, rollback, disaster recovery, observability and operator workflows.

**Excluded:** actually downloading datasets/models, installing/running optional model stacks, executing tests/benchmarks, training models, mining real hard negatives with real indexes/models, running inference, calibrating on real data, provisioning external infrastructure/KMS/databases/object stores, executing real DR drills, and claiming measured performance.

No test suite, model training run, dataset download, model download or benchmark execution was performed as part of this strict source audit.

## Repository/process state

At the end of the implementation sweep:

- development was committed directly to `main`;
- no new PR was opened;
- branch enumeration returned only `main`;
- open-PR enumeration returned none;
- historical PR/commit history was not rewritten;
- targeted indexed searches returned no hits for `TODO`, `FIXME` or `NotImplementedError` in the current repository.

## Strict conclusion

Under the source-only boundary above, **no material implementation family remains unimplemented on `main` that can be concretely specified without selecting or executing an external model, dataset, provider or deployment environment**.

This is stronger than the earlier source audit. The earlier document was intentionally conservative around learned training/architecture and several operational seams. Those seams now have executable source implementations.

The result does **not** mean the system has been experimentally validated or deployed. It means the repository contains the implementation surface needed to perform the excluded execution work later.

## Capability-by-capability strict status

### 1. Owner/tenant-scoped ingestion and authoritative lifecycle — implemented

The repository contains source for:

- owner/tenant scoping;
- immutable source/generation identity;
- metadata/object/vector/lexical/graph lifecycle coordination;
- durable intent, outboxes, sagas, retries and adoption;
- reconciliation, reindex and migration/cutover planning;
- legal hold/retention/deletion/retirement controls;
- crash recovery and idempotent operation records;
- maintained physical-target population reconciliation.

The last missing fleet-level seam was closed by:

- `8e37433dc2d3a44e7b207e12a1f3159b83b534c4` — maintained target population reconciliation;
- `2a84c3766f1dc522ec2bec24d1951117597edba4` — source tests for population reconciliation;
- `adfa9491f48f44f9d0af9732db5e38a052b4de4f` — reconciliation contract documentation.

The reconciler distinguishes maintained logical targets from physical collections, suppresses duplicate builds, requires exact-ready observation before alias CAS, records orphan candidates conservatively, and never converts orphan detection into implicit deletion.

### 2. Distributed cutover/blue-green coordination — implemented

Current source contains:

- routing/cutover control;
- same-dimension and dimension-changing migrations;
- durable population intent;
- crash-resumable hidden population;
- monotonic cross-process fencing;
- exact readback validation;
- generation/route CAS;
- rollback/compensation/recovery.

The previously suspected “distributed blue/green coordinator” gap was stale: later durable cutover/population commits already supplied cross-process fencing and recovery semantics.

### 3. Retrieval architecture matrix — implemented

Source exists for:

- BM25/lexical retrieval;
- dense retrieval;
- hybrid retrieval;
- learned sparse retrieval, including SPLADE/uniCOIL-style representations and scoring;
- ColBERT-style late interaction/MaxSim;
- cross-encoder and listwise reranking;
- independent-corpus retrieval and weighted reciprocal-rank fusion;
- source/document caps and retrieval filters;
- bounded reranking cascades;
- hierarchical/contextual retrieval;
- multi-query, HyDE and step-back paths;
- temporal/entity/acronym normalization;
- graph/GraphRAG and cross-document evidence retrieval;
- multi-hop/agentic planning;
- multilingual/scientific/multi-vector profile governance;
- multimodal page/table/chart/figure retrieval;
- calibration/uncertainty/abstention.

Raw scores from incompatible retrieval profiles are not assumed to be directly comparable; cross-corpus/profile paths use governed fusion/ranking abstractions.

### 4. Executable learned model architectures — implemented

`training/model_architectures.py` contains executable optional PyTorch/Hugging Face source for:

- dense encoders and tied/untied bi-encoders;
- mean/CLS/max/last pooling;
- projection and L2 normalization;
- SPLADE encoders;
- uniCOIL term weighting;
- ColBERT token projection and MaxSim;
- cross-encoder rerankers;
- listwise rerankers;
- local-only pretrained loading with remote code disabled.

Static corrections include the ColBERT all-pairs einsum fix and DDP-safe forward paths.

### 5. Losses/objectives/distillation — implemented

`training/torch_losses.py`, `training/distilled_steps.py` and related planning source contain:

- InfoNCE/in-batch contrastive objectives;
- optional cross-rank negatives;
- sparse L1/FLOPS regularization;
- sparse retrieval objectives;
- pairwise softplus and margin ranking;
- ListNet/ListMLE-style listwise losses;
- KL teacher distillation;
- margin distillation;
- symmetric contrastive loss;
- Matryoshka/nested-dimension training;
- masked distillation over only finite teacher-scored candidates.

### 6. Training data/collation/hard-negative mining — implemented

Source contains:

- manifest-bound local JSONL datasets;
- byte SHA/count verification;
- deterministic positive/negative selection;
- known-positive false-negative masks;
- resumable deterministic sampling;
- bi-encoder and cross-encoder collators;
- deterministic hard-negative generations;
- rank-hard/teacher-hard/semi-hard strategies;
- known-positive filtering and deduplication;
- immutable hard-negative generation manifests;
- refresh hooks for staged training.

Actual corpus/model execution to mine real negatives remains intentionally excluded.

### 7. Training engine/checkpointing/resume/stage saves — implemented

Source contains:

- AdamW optimizer construction;
- warmup/linear/cosine scheduling;
- fp32/bf16/fp16 execution modes;
- gradient accumulation and clipping;
- optional DDP;
- evaluation and early stopping hooks;
- stage boundaries and stage checkpointing;
- hard-negative refresh hooks;
- exact resume inspection before tensor restore;
- safetensors model checkpoints;
- optimizer/scheduler/scaler state;
- Python/PyTorch/CUDA/NumPy RNG state;
- sampler/collator state;
- trainer cursor/state;
- artifact SHA verification;
- latest/best/stage pointers;
- retention pruning;
- exact 40/64-character Git source identity support.

The content-addressed self-reference issue for “best” was corrected by moving that pointer outside the checkpoint payload.

### 8. Learned query/domain routing — implemented

Source contains:

- a trainable linear-softmax domain classifier;
- cross-entropy fitting;
- learned plan candidates and rankers;
- pairwise/listwise plan-ranking objectives;
- deterministic minibatch SGD;
- L2 and early stopping;
- exact epoch/batch cursor resume;
- permutation/RNG state;
- best/current weights and validation state;
- immutable training-manifest/config binding;
- conservative alias and temporal normalization.

### 9. Governed model/provider adapters — implemented at the provider-neutral boundary

Source contains revision/artifact/tokenizer-governed model profiles and verified-local adapters for:

- dense embedding;
- SPLADE;
- ColBERT;
- cross-encoder reranking;
- local document-layout/table/formula/OCR model paths;
- multimodal image+text entailment.

Governed profiles cover multilingual, scientific, instruction-tuned and multi-vector purposes without hardcoding unreviewed mutable model names as authoritative production choices.

A specific cloud/model registry choice can add a provider adapter later, but that is selection-dependent integration rather than a missing algorithmic/source family.

### 10. Citation/provenance/text semantic support — implemented

Source contains:

- generation/page/chunk/block/region anchors;
- claim-to-evidence provenance;
- entailment/neutral/contradiction probabilities;
- citation support metrics;
- semantic coverage/accuracy/recall;
- contradiction false-negative metrics;
- multiclass Brier score and ECE;
- contradiction-first quality gates;
- immutable correction/review lineage.

### 11. Multimodal entailment/support — implemented

This residual was closed by:

- `271b3935266834044dfab4f89d9bd15c51cfcd86` — multimodal evidence support semantics;
- `70e26ff04a2280231c557814b4f97f83c4dcca3f` — verified-local multimodal entailment adapter;
- `0b4c238961dc1d9c3c2f9edc4b998ad432283c99` — source tests;
- `7a00ea5ac9a32661ee77489e08c65be432207378` — input/label validation hardening.

The durable result binds textual claims to immutable document/generation/page/region anchors and image digests rather than storing raw image bytes.

### 12. Scientific evidence semantics/extraction/document structure — implemented

Source contains:

- PICO/PECO/PICOS/freeform research questions;
- methods/population/intervention/exposure/comparator/outcome/limitation fields;
- effect estimates and uncertainty;
- risk-of-bias/certainty fields;
- immutable review/correction lineage;
- reading-order DAGs;
- normalized geometry;
- table topology and merged cells;
- formula representations;
- figures/panels/captions/cross-references;
- structure-quality accept/review/block gates;
- extraction pipelines bound to authoritative structured-document regions;
- concrete local OCR/layout/table/formula adapters.

### 13. Graph/cross-document/evidence-graph lifecycle and recovery — implemented

The repository contains GraphRAG/cross-document evidence graph source together with the evidence-graph signed-retirement snapshot, custody, mutation, restore, artifact-journal and recovery-executor stack.

The older “evidence-graph-specific backup/restore is absent” residual was therefore too coarse. The current tree contains evidence-graph-specific restore/custody implementation, including durable recovery executors and signed snapshot/retirement handling. Generic `tools/disaster_recovery.py` supplies checksum-verified local backup/restore primitives as an additional lower-level path.

The source family that actually remained was **DR rehearsal evidence**, closed in this final sweep.

### 14. Disaster-recovery rehearsal / RTO-RPO evidence — implemented in the final sweep

Commits:

- `b5bfb3cbdad388cb9987133c05cfa7d19fd911a2` — fenced DR rehearsal orchestration;
- `6879080aed843b275920c769304c54e02d237f11` — isolation/custody/exact-population/privacy hardening;
- `b7b96159c68f3ab70dd78aa8ebc9b61850fc9b24` — focused source invariants;
- `33d0b2f9e0e8ec9f7fd6ccfdddbd44719239b282` — DR rehearsal contract documentation.

`orchestration/disaster_recovery_rehearsal.py` now provides:

- content-addressed drill specifications;
- exact owner/objective/recovery-point/custody/policy binding;
- monotonic SQLite fencing;
- transaction-bound lease and revision CAS;
- durable request states before external side effects;
- deterministic idempotency keys;
- multi-component restore/verify sequencing;
- isolated-only backend protocol with no production promote/cutover method;
- concrete local-file adapter over existing backup manifests;
- checksum verification before restore;
- path traversal and symlink/redirect rejection;
- exact recovered-file population verification;
- privacy-safe durable error digests;
- worst-component RPO calculation;
- verified-readiness RTO calculation;
- cleanup proof;
- content-addressed final rehearsal receipt with digest verification on reload.

### 15. Dataset/benchmark governance and acquisition boundary — implemented

Source contains:

- immutable dataset/version/source/checksum/license manifests;
- reviewed license evidence;
- transformation/loader identity;
- exact split digests;
- leakage checks;
- verified operator-supplied local artifact binding;
- safe file/path/link/reparse/inode/size/SHA verification;
- benchmark proposals kept separate from promotable verified manifests.

Actual dataset downloading remains explicitly excluded; implementing an ungoverned automatic downloader would weaken rather than complete this boundary.

### 16. Evaluation/statistics/reproducibility — implemented

Source contains:

- current-vs-shadow paired evaluation;
- fixed seeds/query contracts;
- repeated runs;
- ablations;
- historical baselines;
- retrieval/citation/semantic/calibration metrics;
- paired bootstrap;
- paired permutation/randomization;
- Cohen's dz with defined zero-variance behavior;
- Holm correction;
- BH/FDR correction;
- resource measurements;
- statistical and practical promotion gates;
- finite-sample conformal thresholds and abstention;
- calibration-manifest binding.

### 17. Unified quality observability — implemented

The older “unified retrieval-quality dashboard aggregation” residual is closed by:

- `b198b15bc24e0a41cd931ac18f9090ea022f5e08` — `evaluation/quality_observability.py`;
- `572879265f8e35941dd9caecb37cc74b9dc6273f` — source tests;
- `083bf93636adf31c65c77801f47747804f551b45` — contract documentation.

That module normalizes retrieval, citation, semantic, selective-risk, latency/resource and drift measurements; binds immutable provenance; enforces privacy-safe dimensions; evaluates SLOs; compares like-for-like snapshots; and writes canonical machine-readable observability artifacts. It does not fabricate measurements.

### 18. Periodic leadership/reconciliation — implemented and corrected

Source contains fenced periodic leadership and one-shot reconciliation wiring suitable for invocation by an external scheduler.

The first-run starvation defect was fixed by:

- `dc0b9dc958ee11e925453024d990162fbacca53d`;
- `d568b8597f8ac918b5c28bb97839483eebdba9ec`.

A never-run job is now immediately due; deterministic jitter applies to subsequent interval anchors instead of moving the first due time forward on every poll.

### 19. Continual drift -> rebuild -> benchmark -> promotion/rollback — implemented

Commits:

- `c856fd2e5ef435df710f2360abdf5b1e66e67e2a` — durable continual adaptation coordinator;
- `915d7284e36b54689c7a119ca25439fe873731db` — source tests;
- `86499c87c15bbdbb6c0099bd273ee525e413460b` — contract documentation;
- `a9cbfa284c2ba30c4cdc64c4a6c2bbbf4bcc57c3` — transaction-bound fencing hardening.

The workflow binds drift evidence to exact training request/output, benchmark receipt, continual-learning evidence and baseline-CAS promotion/hold/rollback decisions. Build and benchmark request states are persisted before external calls for safe idempotent replay.

### 20. Durable expert adjudication/gold labels — implemented

Commits:

- `927d914b6992bfc22b8ef281d0ef7cc918293a75` — durable adjudication/gold-label workflow;
- `00efbdfbd8f6471e16c226b7e88220b5f6475d99` — source tests;
- `8e42331e584027936476d19056e754f3183010b0` — claim-race/role-independence hardening;
- `2bdca1117616886c55beb6356d62342be987eb27` — transaction-bound fencing;
- `7375c40a6fac27bcfe2ae3dd4d0c04fd93b3b3c1` — workflow documentation;
- `e03b30e2d85d20a568da7de4b7ffd0db2900e9f1` and `044d4c4d698b0e1b1fde547be74917ead70d43ee` — self-contained hardening regressions.

Source now includes immutable case identity, reviewer/adjudicator role separation, expiring fenced claims, append-only judgments/corrections, quorum/conflict escalation, adjudicator resolution, correction rounds, immutable resolution receipts and current-round gold export.

### 21. Security/key management/governance — implemented at the architecture/provider boundary

Source contains:

- SSRF controls;
- parser isolation and malware scanning adapters;
- auth/owner boundaries;
- resource/rate controls;
- KMS/HSM envelope-encryption provider contracts;
- key references/rotation/rewrap records;
- legal-hold-aware rotation planning;
- privacy-safe operator audit export;
- retention/compaction;
- pause/resume/cancel worker control;
- durable leases/fencing/reconciliation;
- review/disclosure/attestation controls.

Cloud-provider-specific credentials/provisioning are deployment selections and remain outside the source-only audit unless a concrete provider is selected.

### 22. Hydrology/geospatial/research lineage — implemented

The pre-existing repository contains hydrology/geospatial lineage, scenario/replay/research capsule/workspace and distributed recompute source. No new hydrology algorithmic gap was found in this strict continuation.

## Static defects corrected during the strict sweep

The continuation also corrected source-level issues rather than merely adding features:

- ColBERT all-pairs einsum labels;
- DDP paths that bypassed `forward()`;
- checkpoint resume ordering;
- best-pointer self-reference in content-addressed checkpoints;
- Git commit identity incorrectly treated as fixed SHA-256 data identity;
- partial teacher-score NaN handling;
- router exact batch/epoch resume;
- corpus-fusion representative update ordering;
- conformal unsupported-alpha fail-closed behavior;
- zero-variance paired standardized effect handling;
- periodic first-run starvation;
- multimodal label/input validation;
- continual/adjudication SQLite fencing outside the mutation transaction;
- adjudicator role/claim races;
- DR rehearsal component path traversal, redirect safety, exact recovered population and privacy-safe error persistence.

## What still remains — execution/artifact work only

The remaining work is not missing source implementation under the requested boundary. It is the work required to produce real artifacts and empirical evidence:

1. Install optional training/model dependencies in a selected environment.
2. Select exact promotable model revisions and obtain their governed local artifacts.
3. Review exact licenses for selected models and datasets.
4. Acquire/download benchmark and training datasets and verify their checksums/manifests.
5. Build real lexical/vector/sparse/late-interaction indexes.
6. Run real hard-negative mining.
7. Train dense/sparse/late-interaction/reranker/router models.
8. Produce real checkpoints and model artifacts with the implemented checkpoint pipeline.
9. Run inference and calibrate confidence/conformal thresholds on real calibration sets.
10. Execute benchmark suites, ablations, shadow runs, statistical tests and promotion gates.
11. Execute the source tests and CI suites.
12. Provision production metadata/object/vector/lexical/graph stores and distributed workers.
13. Configure concrete cloud KMS/HSM/object-store/search/vector-provider credentials/adapters where selected.
14. Run fault-injection, crash-recovery and stale-worker exercises.
15. Execute real DR rehearsal drills and measure deployment-specific RTO/RPO.
16. Collect real quality/latency/resource/cost observations and render/serve them through the desired operations UI.
17. Perform deployment/canary/promotion/rollback in the target infrastructure.

None of those actions should be inferred from the existence of source code alone.

## Final source-level result

Against implementation baseline `33d0b2f9e0e8ec9f7fd6ccfdddbd44719239b282`, the strict audit finds **zero material source-implementation gaps within the requested non-execution boundary**.

Any future source addition should therefore be driven by a newly selected external provider/model/dataset/deployment requirement, an observed test/runtime defect, or a genuinely new research capability—not by stale unchecked boxes in historical planning documents.
