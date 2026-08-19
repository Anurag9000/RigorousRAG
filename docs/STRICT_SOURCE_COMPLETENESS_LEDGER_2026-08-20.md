# RigorousRAG strict source-completeness ledger — 2026-08-20

This document supersedes earlier continuation/readiness notes for the question:

> After local datasets and admitted local model/tokenizer artifacts exist, is any additional
> source code still required to construct the intended architectures/objectives, prepare
> supervision, train in stages, checkpoint/resume, evaluate, export, qualify and hand the
> resulting components into the existing runtime authority?

The answer at this source revision is **no for the canonical paths described below**. Remaining
work is empirical/execution work: dependency/runtime availability, dataset acquisition,
materializing supervision with actual local models, actual training/inference, benchmark runs,
compile/import/test execution, performance tuning and empirical promotion/calibration choices.
Those exclusions are not treated as source implementation gaps.

This is a source audit, not a claim that the code has been executed successfully in the current
environment. No model training, dataset download or benchmark execution was performed as part
of this audit, and the local environment used for the audit could not resolve GitHub for an
independent clone/compile pass.

---

## 1. Canonical end-to-end source paths

### 1.1 Grounded generator

Canonical flow:

1. already-local reviewed annotations;
2. governed grounded import with exact source SHA-256/version/license/transformation identity;
3. authoritative grounded records with evidence, claim spans, stanced support/contradiction,
   abstention, reflection, unsupported spans and optional preference labels;
4. canonical grounded publication with globally isolated example IDs, exact split/evidence ID
   digests and deterministic cache keys;
5. final `DatasetManifest` publication;
6. optional teacher/reference/document-utility supervision materialization against that final
   manifest;
7. read-only sealing of exact supervision-cache contents;
8. exact cache contracts embedded into the training JSON configuration;
9. causal-LM or seq2seq grounded model composition;
10. staged curriculum and stage-local trainability;
11. multi-positive citation, support, contradiction, abstention, reflection, unsupported-token,
    preference, teacher-distillation and retriever-coupling objectives;
12. deterministic sampler/collator plus stage-aware validation and early stopping;
13. content-addressed checkpointing/resume;
14. verified best/latest checkpoint resolution;
15. inference-only artifact export;
16. evaluation receipt, qualification, promotion evidence and final artifact admission;
17. typed bridge into the existing runtime-stack authority.

### 1.2 Dynamic RAG policy

Canonical flow:

1. recorded legal runtime decisions with server-owned action legality;
2. collision-resistant canonical `(episode_id, step_id)` identity;
3. reviewed information-need labels and deterministic hidden-state cache keys planned **without**
   writing a pre-final-manifest cache;
4. optional measured retrieval-gain binding;
5. value/GAE/counterfactual target materialization with legal-action semantics;
6. whole-episode deterministic train/validation publication;
7. final `DatasetManifest` publication;
8. generator hidden-state materialization into a cache whose identity binds that final manifest;
9. read-only sealing of exact hidden-state cache contents;
10. exact hidden-cache contract embedded into the training JSON configuration;
11. imitation, need-selection, value, off-policy, cost-aware and joint curriculum stages;
12. legal-action-masked policy objective and validation;
13. content-addressed checkpointing/resume;
14. policy artifact export/qualification;
15. typed bridge into the existing runtime-stack authority.

The older one-pass dynamic preparation helper is retained only as an explicitly opt-in,
**non-promotable compatibility/research path**. It is not the final-training authority because
its original ordering cannot prove that the hidden cache identity binds the final published
training manifest.

---

## 2. Architecture and objective completeness

The source contains executable model/objective definitions for the advanced families added in
this continuation and the previously existing retrieval/ranking families. No additional model
family was identified as necessary to meet the stated project mission.

Grounded generation includes:

- injected causal-LM and encoder-decoder/seq2seq backbones;
- claim/evidence attribution projections;
- support and contradiction heads;
- abstention head;
- reflection/action head;
- supervised token NLL;
- multi-positive claim-to-evidence citation objective;
- support/contradiction BCE;
- abstention and reflection objectives;
- unsupported-token unlikelihood;
- DPO-style grounded preference optimization;
- reference-policy sequence-log-probability supervision;
- teacher-logit distillation;
- generator-scored document utility and retriever coupling.

Dynamic retrieval policy includes:

- bounded policy/action controller;
- state-value head;
- token-level information-need selector;
- explicit legal action sets;
- imitation CE;
- information-need token objective;
- value regression to explicit return targets;
- clipped logged-behavior/current-policy off-policy objective;
- action cost regularization;
- entropy regularization;
- joint objective.

The broader repository already contains the classic/dense/sparse/late-interaction/reranking/query-
planning/fusion/scientific/evidence/security source families audited in prior ledgers. This
continuation did not find a missing major retrieval architecture, loss family, optimizer,
scheduler or serving architecture.

---

## 3. Data and supervision source completeness

### 3.1 Governed local benchmark/data import

The repository now has local-only governed import/conversion rather than requiring operators to
write dataset glue after downloading data. It supports:

- exact input SHA-256 binding;
- JSON/JSONL normalization;
- existing named benchmark adapters plus declarative mappings;
- real `DatasetManifest`/`SplitManifest` publication;
- license/promotability metadata;
- leakage qualification;
- self-verifying receipts;
- canonical benchmark JSONL iteration;
- retrieval benchmark corpus identity and query/corpus bundle identity;
- auxiliary exact artifacts such as qrels;
- benchmark-result receipts and promotion-evidence handoff;
- declarative grounded annotation conversion.

Dataset acquisition/download remains intentionally outside the implementation boundary.

### 3.2 Grounded authoritative dataset

The authoritative grounded dataset source covers:

- prompt/instruction and answer;
- evidence IDs/text/source IDs;
- claim character spans;
- supporting and contradicting evidence IDs;
- supported/contradicted labels;
- abstention;
- reflection action;
- unsupported answer spans;
- chosen/rejected answers;
- optional materialized reference log probabilities;
- teacher/retriever cache keys;
- bounded metadata.

### 3.3 Dynamic authoritative dataset

The authoritative dynamic dataset source covers:

- episode and step identity;
- exact generation context;
- complete dynamic feature vector;
- logged action;
- legal-action set;
- behavior action probability;
- realized retrieval gain;
- advantage;
- explicit return/value target;
- information-need spans;
- hidden-state cache key;
- terminal utility;
- bounded metadata.

### 3.4 Lazy content-stable authoritative JSONL access

`ManifestBoundAuthoritativeJsonlDataset` no longer retains the full parsed corpus in RAM.
Construction performs one streaming validation pass that:

- computes the exact whole-file SHA-256;
- strict-parses every non-empty UTF-8 JSON line;
- applies the authoritative record parser;
- enforces line/record safety bounds;
- detects duplicate logical record identities through a temporary disk-backed SQLite index;
- retains compact byte offsets, byte lengths and per-line SHA-256 digests.

Each `__getitem__` seek rechecks the exact indexed line digest before reparsing, so a later source
rewrite cannot silently alter a consumed training sample.

### 3.5 Grounded corpus-scale identity governance

Canonical grounded publication no longer accumulates all example/evidence IDs in Python
lists/sets. A temporary SQLite ledger:

- rejects any repeated example ID, including cross-split leakage;
- deduplicates evidence IDs per split;
- streams sorted record/evidence ID digests;
- is removed after publication.

---

## 4. Supervision cache authority

`AuthoritativeSafetensorSupervisionCache` now provides a two-phase lifecycle:

### Materialization phase

- strict root/path authority;
- closed `<sha>.json` + `<sha>.safetensors` pairing;
- manifest identity checks;
- exact tensor SHA/name checks;
- verified membership lookup;
- writes allowed only while unsealed.

### Sealed read phase

`seal()` freezes the exact logical key → tensor SHA/tensor-name map and returns the complete
cache content contract. After sealing:

- `put()` is rejected;
- `contains()` and `get()` verify the requested pair against the frozen snapshot;
- newly added keys are rejected;
- replaced/removed/mutated entries are rejected;
- `provider_identity_sha256()` captures the sealed contract rather than a mutable cache root.

Canonical grounded and dynamic materializers seal caches before their final receipts are
published.

Training configs now pin **both** the immutable cache identity and the exact sealed
`contract_sha256`. Parsing/building the same config later reopens, seals and compares the cache;
an internally valid replacement under the same root/producer therefore cannot silently change
the effective training input.

Semantic cache roles are enforced at config parsing, runner construction and checkpoint binding:

- `teacher_logits`;
- `reference_policy_log_probs`;
- `document_lm_utility`;
- `generator_hidden_states`.

All curriculum-required keys are preflighted across both training and validation datasets before
optimizer work begins.

---

## 5. Dynamic identity safety

Delimiter-concatenated authority keys such as `episode:step` were removed from persisted dynamic
supervision paths because valid identifiers may themselves contain `:`.

Canonical dynamic identity now uses SHA-256 of the canonical JSON pair `[episode_id, step_id]`.
This is used for:

- hidden-state cache keys;
- dynamic step identities;
- information-need sidecar lookups (exact tuple internally);
- realized retrieval-gain sidecar lookups (exact tuple internally);
- logged counterfactual utilities (exact tuple or canonical `dynamic-step:<sha256>` only).

The external annotation/gain JSON schemas keep separate `episode_id` and `step_id` fields and do
not need a breaking delimiter encoding.

---

## 6. Canonical path-safe publication filenames

Logical dataset/split identifiers are semantic strings, not filesystem components. A centralized
`logical_filename()` derives on-disk filenames as:

`<sha256(logical-name)><fixed-extension>`

while preserving the exact logical split name in manifests/receipts. Canonical grounded
publication uses this policy and binds it in the transformation identity. This prevents path
traversal or separator ambiguity without changing semantic split identity.

Other lower-level import/publication modules remain subject to their own path authority and are
not part of the final canonical grounded filename contract described above.

---

## 7. Training engine and curriculum

The generic PyTorch engine remains the shared implementation for:

- device placement;
- fp32/fp16/bf16 AMP;
- optional DDP;
- AdamW;
- constant/linear/cosine schedules;
- gradient accumulation;
- clipping;
- deterministic seeding controls;
- staged curricula;
- evaluation cadence;
- early stopping;
- checkpoint cadence and stage-boundary checkpoints.

Advanced source reuses this engine rather than duplicating optimizer/training loops.

Authoritative advanced runners additionally enforce:

- tokenizer contract;
- cache role/producer/tokenizer/dataset/source bindings;
- complete cache-key coverage;
- causal-vs-seq2seq path binding;
- legal action masks;
- final multi-positive citation semantics;
- exact input identity;
- advanced closed-directory checkpoint authority.

Validation is aligned to training semantics:

- grounded citation accuracy accepts any supported positive evidence;
- dynamic action accuracy applies the legal-action mask;
- dynamic value MAE uses explicit value/return targets when available;
- information-need metrics use the valid-token mask.

---

## 8. Checkpoint/restart authority

Generic checkpoint payloads already persist:

- model weights;
- optimizer;
- scheduler;
- AMP scaler;
- trainer stage/cursor;
- Python RNG;
- Torch CPU/CUDA RNG;
- NumPy RNG where present;
- sampler state;
- collator state;
- source revision;
- training config digest;
- dataset manifest digest;
- model architecture identity;
- metric snapshot;
- parent checkpoint lineage.

Advanced checkpoint authority adds:

- non-symlink root authority;
- closed manifest schema;
- closed directory contents;
- immediate strict post-save verification;
- rollback of convenience pointers if strict verification fails;
- verified best/latest/stage pointer resolution;
- direct checkpoint verification restricted to `AdvancedCheckpointManager`.

### Exact-best self-reference repair

A content-addressed checkpoint cannot include its own future digest in
`TrainerState.best_checkpoint_digest`. The authoritative advanced engine therefore reconstructs
best state on resume from verified checkpoint ancestry:

1. keep serialized best only if it is in ancestry and its metric equals serialized best metric;
2. if the resumed checkpoint itself carries the best metric, repair best to the resume digest;
3. otherwise use external `best.json` only if that pointed checkpoint is in the resumed ancestry
   and its metric matches;
4. reject unreconstructable best state.

The generic engine remains reusable and unchanged by this advanced-only repair.

---

## 9. Evaluation and promotion

Advanced evaluation source covers trainer validation and persisted promotion evidence.

Benchmark/promotion source supports:

- repeated run records;
- exact dataset/checkpoint/evaluator identities;
- detailed result receipts;
- aggregate metrics;
- leakage qualification;
- promotion thresholds;
- robustness/poisoning evaluation;
- dynamic policy metrics;
- citation/support/contradiction/abstention metrics.

`AdvancedArtifactPromotionReceipt` is self-verifying and binds:

- artifact SHA;
- policy SHA;
- evaluation receipt SHA;
- exact metrics SHA;
- promoted/blocked decision;
- reason codes.

The higher-level promotion-evidence wrapper independently reconstructs and verifies the nested
primitive receipt.

---

## 10. Inference artifact authority

Advanced export requires a verified advanced checkpoint binding and an
`AdvancedCheckpointManager`.

Inference artifacts are closed two-file directories:

- `manifest.json`;
- `model.safetensors`.

The manifest binds:

- checkpoint digest;
- plan SHA;
- training-input SHA;
- training-config SHA;
- source revision;
- dataset manifest;
- architecture;
- base model/generator family/tokenizer;
- retriever/budget identity where relevant;
- runtime config;
- weights SHA/size;
- included parameter prefixes;
- evaluation receipt where supplied;
- artifact self-digest.

Existing content-addressed export destinations are reverified instead of trusted. New exports
are reverified after publication. Final admission re-hashes the persisted directory immediately
before handing it to an admission sink.

Runtime loading again verifies manifest/weights/local base/tokenizer/retriever identities before
constructing executable models.

---

## 11. Runtime-stack handoff

`orchestration.advanced_rag_runtime_stack_bridge` removes the former manual handoff between
advanced artifact qualification and the existing runtime-stack authority.

It:

- re-verifies persisted advanced artifact bytes;
- requires promoted self-verifying advanced promotion evidence;
- builds a typed runtime component contract bound to artifact/training/runtime lineage;
- maps grounded artifacts to the existing `generator` component slot;
- maps learned dynamic retrieval policy to the existing policy-routing (`query_router`) slot;
- builds runtime stacks using the mature runtime-stack authority;
- produces aggregate exact-stack-bound `offline_quality` evidence.

The existing runtime-stack authority continues to own fencing, component compatibility,
promotion evidence, cutover and rollback semantics.

---

## 12. Compatibility/legacy paths

Compatibility primitives remain only where they have legitimate research value.

Important rule:

- `dynamic_trajectory_pipeline.prepare_and_materialize_dynamic_trajectories()` is **not** a final
  training authority. It refuses execution unless `allow_noncanonical=True` and every receipt it
  returns is explicitly `promotable=False`.
- `advanced_rag_cli.py` is no longer an independent implementation; it is only a compatibility
  shim to the hardened `advanced_rag_operator`.

Thus there is no alternate weaker operator path that silently bypasses the canonical advanced
checkpoint/data/cache authority.

---

## 13. What is source-complete now

Under the stated boundary—local data/model artifacts already exist, but no actual model/data
execution is required—the repository has source for:

- architecture construction;
- causal and seq2seq grounded paths;
- dynamic policy/value/need-selection architecture;
- all intended advanced losses/objectives;
- deterministic staged curricula;
- local governed dataset conversion;
- benchmark conversion/bundle identities;
- grounded authoritative label parsing;
- dynamic episode parsing;
- supervision materialization providers;
- two-phase manifest-bound dynamic hidden-state supervision;
- final-manifest-bound grounded supervision caches;
- collision-safe dynamic step identity;
- deterministic data access/collation;
- sampler/collator checkpoint state;
- optimizer/scheduler/AMP/DDP/accumulation/clipping;
- stage checkpointing;
- exact resume;
- best/latest resolution;
- trainer validation and early stopping;
- evaluation receipts;
- robustness evaluation contracts;
- best-checkpoint export source;
- inference-only artifact creation;
- artifact verification/admission;
- runtime-stack integration;
- existing retrieval/search/scientific/evidence/security/DR/ops source audited previously.

No new architecture, loss, training-loop, checkpoint/recovery or serving-source family is
currently identified as necessary to satisfy the original mission.

---

## 14. What remains, explicitly outside source implementation

The remaining work is empirical/execution work:

1. acquire/download the real datasets under their licenses;
2. make the intended local pretrained models/tokenizers/retrievers available;
3. install/resolve the runtime dependency stack in the execution environment;
4. execute local dataset conversion/materialization commands;
5. actually compute teacher/reference/document-utility/hidden-state caches;
6. actually train grounded and dynamic models;
7. run checkpoint/resume in the target hardware environment;
8. run validation/benchmarks/robustness suites;
9. choose/tune empirical thresholds, temperatures, costs, budgets and promotion criteria;
10. inspect learning curves and failure cases;
11. perform ablations and repeated seeds;
12. run inference/runtime integration tests;
13. run compile/import/unit/integration/security/performance tests;
14. measure latency, VRAM/RAM/disk/throughput and tune operational limits;
15. perform real release/canary/cutover/rollback exercises.

These are intentionally not represented here as missing source.

---

## 15. Audit caveat

This ledger records a **source-level** conclusion. It does not claim empirical correctness,
performance or successful execution. In particular, no model/dataset execution and no full local
clone/compile/test pass was performed in this audit environment. Any defects discovered by later
real execution should be fixed as execution defects; they should not be confused with a known
missing architecture/methodology/source path at the time of this ledger.
