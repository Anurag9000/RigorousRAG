# RigorousRAG strict source-completeness ledger — 2026-08-20 continuation

This document supersedes the **status conclusions** of the earlier strict source-completeness
ledgers while preserving their historical record. It records the final source-only state after
the continuation sweeps that hardened governed data import, canonical training data, exact
resume, benchmark/evaluation evidence, promotion policy semantics, artifact admission, and
runtime-stack handoff.

The repository workflow for this continuation remained direct-to-`main`: no implementation
branch or replacement pull request was created, no force-push/history rewrite was used, and the
final GitHub state was re-checked for a single live `main` branch and no open PRs.

---

## 1. Scope and stopping criterion

The governing user requirement is stricter than “the architecture is sketched” but explicitly
excludes empirical execution. For this ledger, **source-complete** means:

> once the required local governed datasets, local model/tokenizer weights and dependencies are
> available, no new architecture, methodology, loss, data/collation, training orchestration,
> checkpoint/resume, evaluation-evidence, export, promotion, admission or runtime-handoff source
> must be written in order to execute the intended system.

The following remain intentionally outside the claim:

- dependency/environment installation;
- dataset downloading/acquisition;
- remote model downloading;
- actual teacher/reference/base-model execution during cache materialization;
- actual model training or inference;
- benchmark execution;
- test execution;
- compile/import smoke execution;
- empirical hyperparameter or threshold tuning;
- repeated-seed/ablation experiments;
- measured latency, VRAM, RAM, disk or throughput;
- canary/deployment/rollback exercises against a live mutable runtime authority.

Therefore this is a **source completeness and static authority audit**, not an empirical success
claim.

---

## 2. Canonical end-to-end source chain

The authoritative production source chain is now:

```text
local governed source bytes
    -> strict import / immutable manifests
    -> canonical grounded or dynamic training records
    -> deterministic supervision labels and strict sealed tensor caches
    -> immutable training recipe / exact local artifact bindings
    -> authoritative staged trainer
    -> content-addressed exact-resume checkpoints
    -> verified best/latest checkpoint
    -> inference-only advanced artifact export
    -> governed benchmark authority
    -> evaluator contract authority
    -> evaluator-bound exact sample cohort
    -> streaming result artifact + exact sample-universe proof
    -> evaluator-bound advanced evaluation evidence
    -> strict directional promotion policy qualification
    -> exact artifact admission / runtime-component binding
    -> restart-verifiable advanced runtime-stack bundle
    -> existing fenced mutable runtime-stack authority
```

No production stage in this chain relies on a caller merely copying an opaque identity from a
mutable file without an independent content/lineage re-verification path.

---

## 3. Grounded-generation training source status

### 3.1 Model and objective source

Implemented source includes:

- injected local base language model;
- grounded auxiliary heads;
- claim/evidence representation flow;
- citation pointer/multi-positive evidence supervision;
- support and contradiction prediction;
- abstention;
- reflection/action prediction;
- unsupported-token unlikelihood;
- supervised token NLL;
- sequence preference/DPO objective;
- teacher-logit distillation;
- LM-supervised retriever coupling;
- joint weighted objective composition.

The staged plan supports:

1. supervised generation;
2. attribution/citation;
3. grounding;
4. reflection;
5. retriever coupling;
6. preference/distillation; and
7. joint optimization.

### 3.2 Governed training records

The grounded data source includes:

- `GroundedGenerationExample`-family schemas;
- evidence IDs/text/source IDs;
- claim spans;
- claim→evidence supervision;
- explicit supporting vs contradicting evidence;
- unsupported spans;
- abstention labels;
- reflection labels;
- chosen/rejected answers;
- teacher/reference/retriever cache keys;
- manifest/source/import lineage metadata.

Governed import/canonical publication now uses:

- path-safe local source authority;
- SHA-bound local files;
- strict JSON parsing;
- canonical record reparsing before publication;
- deterministic content-derived output filenames;
- disk-backed duplicate/leakage identity tracking;
- immutable split SHA/record/evidence digests;
- one-shot staged/atomic publication;
- closed final receipts/manifests.

### 3.3 Canonical supervision caches

Teacher, reference-policy and document-utility caches are:

- final-dataset-manifest bound;
- tokenizer bound;
- producer/model bound;
- source-commit bound;
- content-contract bound;
- safetensor-only;
- closed-directory verified;
- explicitly sealed read-only before canonical receipt publication.

The training config includes the **exact sealed cache content contract**, not merely a cache
root or producer identity.

### 3.4 Collation and label alignment

The authoritative collator path includes:

- causal-LM and seq2seq-LM alignment contracts;
- character span→token alignment;
- variable claim/evidence padding;
- claim/evidence masks;
- multi-positive citation targets;
- support/contradiction matrices;
- unsupported-token masks;
- chosen/rejected preference tensors;
- teacher token logits;
- retriever-document utility vectors;
- deterministic resumable collator state.

### 3.5 Validation parity

Validation is aligned with the final training objective semantics:

- legal/masked citation alternatives are recognized;
- multi-positive citations are evaluated as multi-positive rather than one legacy pointer;
- support/contradiction/abstention signals are retained;
- preference/distillation diagnostics are available;
- trainer early stopping consumes the same semantic outputs trained by the authoritative step.

**Grounded-generation source status: complete under the stated source-only boundary.**

---

## 4. Dynamic-RAG policy training source status

### 4.1 Model and objective source

Implemented source includes:

- policy head;
- state-value head;
- information-need selector;
- legal action masking;
- actions: continue/retrieve/verify/abstain/stop;
- imitation loss;
- information-need token loss;
- value/return loss;
- GAE-style return/advantage targets;
- clipped off-policy policy-gradient correction;
- behavior-policy probability use;
- action-cost regularization;
- entropy regularization;
- joint objective composition.

### 4.2 Runtime-recorded episode source

The durable episode/step representation supports:

- episode and step identity;
- generation context/prefix;
- dynamic feature vector;
- chosen/logged action;
- legal action set;
- behavior action probability;
- realized retrieval gain;
- information-need spans;
- generator hidden-state cache key;
- terminal utility;
- explicit value target;
- lineage metadata.

### 4.3 Collision-resistant dynamic identities

All authority-bearing dynamic supervision paths now use a canonical collision-resistant
`(episode_id, step_id)` identity rather than ambiguous string concatenation such as
`episode:step`.

This covers:

- generator hidden-state cache keys;
- information-need sidecars;
- realized-retrieval-gain sidecars;
- logged counterfactual utilities;
- persisted trajectory identities used by the authoritative path.

### 4.4 Two-phase final-manifest-bound preparation

The promotable dynamic source path is intentionally two phase:

1. prepare deterministic episode/step records and final dataset publication;
2. materialize generator hidden states into a cache whose identity binds that final manifest.

The older one-pass trajectory pipeline is explicitly fenced as non-promotable/research-only and
requires an explicit opt-in rather than being an alternate production authority.

### 4.5 Legal-action/off-policy parity

The installed/authoritative runner uses the final legal-action path:

- illegal logits are masked before action objectives;
- value learning uses explicit return/value targets;
- off-policy ratios are derived from current-vs-behavior probabilities rather than a fabricated
  vector of ones;
- variable-length hidden-state cache collation is supported;
- validation uses the same legal-action mask and value targets as training.

**Dynamic-RAG policy source status: complete under the stated source-only boundary.**

---

## 5. Training engine and exact-resume authority

The generic staged PyTorch engine already owns:

- device placement;
- mixed precision;
- optional DDP;
- AdamW;
- scheduler state;
- gradient accumulation;
- gradient clipping;
- deterministic seed policy;
- evaluation intervals;
- early stopping;
- staged curricula;
- optional hard-negative refresh;
- final training summaries.

The advanced authoritative runners additionally enforce:

- local-only immutable model/tokenizer identities;
- exact dataset split content SHA;
- exact sealed cache contracts;
- curriculum field/cache coverage before optimizer work;
- trainability identity;
- advanced closed checkpoint manager;
- lineage-aware exact resume.

### 5.1 Checkpoint contents

Checkpoint source captures:

- model parameters;
- optimizer;
- scheduler;
- scaler;
- trainer/stage cursor;
- Python RNG;
- Torch CPU RNG;
- Torch CUDA RNG;
- NumPy RNG when present;
- sampler state;
- collator state;
- source commit;
- training config digest;
- dataset manifest identity;
- model architecture identity.

### 5.2 Advanced checkpoint hardening

Authoritative advanced checkpoints now additionally have:

- closed manifest schema;
- closed directory membership;
- no symlinked child acceptance;
- verified artifact hashes;
- post-save verification before pointer advancement;
- pointer rollback if verification fails;
- verified restart-time `best`/`latest` pointer resolution;
- lineage-aware repair of the self-referential “new best checkpoint” resume edge.

**Checkpoint/resume source status: complete.**

---

## 6. Canonical recipes and no-new-source execution readiness

The repository contains:

- canonical grounded and dynamic data builders;
- concrete local-only supervision providers;
- manifest-bound cache materializers;
- canonical training-data bundle/restart descriptors;
- governed/canonical recipe bridges;
- exact cache-content-pinned training configs;
- turnkey authoritative training runners.

A separate duplicate CLI for every library composition was deliberately **not** added where the
existing canonical builder + recipe bridge already forms a complete source API. This is an
operator ergonomics choice, not a missing methodology/source dependency: no new model, label,
loss, collator, trainer, checkpoint or export source must be written to use the existing local
artifacts.

---

## 7. Governed benchmark authority

### 7.1 Query benchmark import

Promotion-grade benchmark import includes:

- strict local source SHA binding;
- path-safe filenames;
- bounded/streaming production input policy;
- canonical row validation;
- immutable dataset manifests;
- disk-backed cross-split leakage qualification;
- independently re-readable/recomputed leakage receipts.

### 7.2 Qrels

Qrels authority is disk-backed and receipt-bound. It verifies:

- query/document pair identity;
- pair count;
- query count;
- document count;
- relevant-pair digest;
- local receipt/source integrity.

### 7.3 Corpus publication

The authoritative corpus path is closed and content-bound, with deterministic document IDs and
source identity.

### 7.4 Retrieval benchmark v3

The retrieval benchmark contract now proves before evaluation:

- every qrels document exists in the authoritative corpus;
- every governed benchmark query has qrels;
- no extra qrels query exists outside the governed benchmark;
- governed query IDs and qrels query IDs are **exactly equal as sets**;
- the sorted query universe has an immutable SHA;
- qrels/corpus coverage has an immutable SHA.

At result publication, the exact inverse proof is enforced:

- exactly one result row for every authorized query;
- no missing result query;
- no extra result query;
- no duplicate result query;
- result-universe SHA equals the benchmark query-universe SHA.

**Governed retrieval benchmark source status: complete.**

---

## 8. Evaluator contract authority

A promotion-grade evaluator is no longer represented by an opaque caller-supplied 64-hex
string.

The evaluator receipt now binds:

- evaluator ID;
- implementation ID;
- full source commit;
- exact local evaluator config content SHA;
- declared metric schema;
- metric family;
- metric direction;
- metric scope;
- metric definition;
- sample semantics;
- aggregation semantics.

The semantic evaluator SHA is **path-independent**: moving identical admitted config bytes does
not change the evaluator identity, while the local receipt still records the current path for
restart-time byte re-verification.

### 8.1 Strict production evaluator semantics

The reusable evaluator library permits broader research descriptions. The installed production
publisher and production promotion path require exactly:

```text
sample_semantics      = one_result_row_per_authorized_sample
aggregation_semantics = arithmetic_mean_over_exact_cohort
metric.scope           = both    # for every promotion-grade metric
```

This matches what the streaming result artifact can actually encode and recompute.

A production evaluator cannot claim:

- median/percentile sample aggregation when the materializer computes a mean;
- per-sample-only metrics absent from aggregate evidence;
- aggregate-only metrics with no per-row representation.

**Evaluator identity/semantics source status: complete.**

---

## 9. Evaluator-bound cohort authority

The production cohort is layered deliberately.

### 9.1 Base authoritative cohort

The base cohort binds:

- authoritative benchmark authority;
- selected governed splits (for generic benchmark v2) or exact retrieval query universe;
- benchmark manifest SHA;
- benchmark contract SHA;
- exact sample count;
- exact sorted sample-ID universe SHA;
- base evaluator contract SHA;
- derived evaluator contract SHA;
- component authority receipt file SHAs.

### 9.2 Evaluator-bound cohort

The production wrapper additionally binds:

- exact base cohort receipt bytes/contract;
- exact evaluator receipt bytes/contract.

Restart verification reconstructs both and proves the evaluator receipt is the evaluator whose
SHA the cohort contains.

**Sample/evaluator cohort source status: complete.**

---

## 10. Streaming benchmark result evidence

The installed result command does **not** accept one giant in-memory rows array. It consumes an
already-produced local JSONL result stream whose complete bytes are bound by an operator-supplied
SHA-256.

The promotion-grade materializer:

- validates a strict header;
- validates each canonical row;
- uses SQLite for duplicate-ID and metric accumulation state;
- recomputes aggregate metrics from rows;
- compares the supplied footer aggregate to independent recomputation;
- writes a closed `result.jsonl` + `result_receipt.json` directory;
- re-verifies the result receipt;
- proves the exact evaluator-bound cohort sample universe;
- removes the just-created output if final verification fails.

### 10.1 Homogeneous row metric schemas

Every row must use the same:

- retrieval metric key set; and
- generation metric key set.

This prevents an aggregate metric from silently using fewer samples because some rows omitted a
metric.

### 10.2 Evaluator metric schema equality

The final aggregate metric names must equal the evaluator contract metric names exactly:

- no undeclared metric;
- no missing declared metric.

**Result evidence source status: complete.**

---

## 11. Advanced evaluation evidence v3

Historical v1/v2 evaluation evidence remains readable from its original modules for research
reproducibility.

The **production verifier** accepts only evaluator-bound v3 evidence. It binds and reconstructs:

- exact checkpoint/run identity;
- exact evaluator-bound cohort;
- underlying cohort-bound evaluation evidence;
- advanced evaluation receipt;
- each strict v2 result receipt/artifact;
- exact sample universe;
- evaluator receipt/config/source identity;
- homogeneous row metric schema;
- evaluator metric schema equality.

Promotion therefore cannot accept a caller-invented evaluator SHA or a cherry-picked sample
subset.

**Advanced evaluation evidence source status: complete.**

---

## 12. Strict promotion policy semantics

The primitive advanced promotion evidence already self-verifies:

- artifact SHA;
- evaluation evidence SHA;
- evaluation receipt SHA;
- metrics SHA;
- policy SHA;
- recomputed qualification decision/reason codes.

Production adds an exact evaluator-direction policy contract:

- every `direction=maximize` metric requires one `minimum` threshold;
- every `direction=minimize` metric requires one `maximum` threshold;
- at least one directional metric must exist;
- descriptive metrics cannot be thresholded;
- minimize metrics cannot receive minimum thresholds;
- maximize metrics cannot receive maximum thresholds;
- policy keys outside the evaluator metric schema are rejected.

This prevents a mathematically self-consistent but empty/selectively incomplete/wrong-direction
policy from qualifying an artifact.

The same strict assertion is used by:

- installed release qualification;
- installed promotion verification;
- production artifact admission;
- direct runtime-component binding;
- runtime-bundle reconstruction.

**Promotion-policy source status: complete.**

---

## 13. Artifact export and admission

Advanced artifact export requires the advanced closed checkpoint authority.

Exported artifacts bind:

- checkpoint digest;
- plan SHA;
- training input SHA;
- training config SHA;
- source commit;
- dataset manifest SHA;
- architecture SHA;
- base model/generator identity;
- tokenizer identity;
- optional retrieval stack identity;
- runtime/budget identities;
- optional evaluation receipt identity;
- exact safetensor bytes/size/SHA.

Existing content-addressed destinations are re-verified rather than trusted.

Production artifact admission re-hashes the exact persisted artifact directory immediately
before the sink handoff and re-runs strict evaluator/promotion evidence verification.

**Artifact export/admission source status: complete.**

---

## 14. Runtime-stack handoff

The repository already contains a mature mutable runtime-stack authority with:

- typed runtime components;
- stack content identity;
- promotion evidence;
- policy enforcement;
- fencing;
- monotonic state transition semantics;
- rollback/current-stack authority.

This continuation intentionally did **not** create a second mutable runtime database/controller.

### 14.1 Strict advanced component binding

The advanced bridge now requires:

- exact artifact-directory verification;
- evaluator-bound v3 evaluation evidence;
- strict directional promotion-policy coverage;
- exact promoted artifact identity.

The runtime component contract records this production evidence requirement.

### 14.2 Restart-verifiable runtime bundle

A new immutable pre-activation bundle closes the source handoff into the mutable runtime
authority.

The closed directory contains exactly:

- `stack.json`;
- `offline_quality.json`;
- `bindings.json`;
- `bundle_receipt.json`.

The bundle binds:

- promoted advanced artifact locations and binding SHAs;
- other runtime components;
- retrieval/generation/compatibility contracts;
- source revision;
- valid-from/expiry;
- exact reconstructed `RuntimeStackArtifact` SHA;
- exact advanced offline-quality evidence SHA;
- file SHAs for every persisted bundle artifact.

Restart verification:

1. rejects symlinked/non-canonical bundle children;
2. re-verifies each source artifact and promotion evidence;
3. rebuilds every advanced runtime component binding;
4. rebuilds the runtime stack;
5. rebuilds offline-quality evidence;
6. compares semantic SHAs;
7. compares persisted JSON through canonical JSON normalization so tuple/list serialization does
   not create false mismatches.

Mutable activation/rollback remains delegated to the existing fenced runtime authority.

**Runtime pre-activation source handoff: complete.**

---

## 15. Installed production command surfaces

The package intentionally separates training and release authority.

### Training

`rigorousrag-advanced-training`

Advertises only:

- config validation;
- training;
- checkpoint verification;
- artifact export;
- artifact verification.

Historical research evaluation/qualification helpers remain importable but are not exposed by
this installed command.

### Release/evaluation

`rigorousrag-advanced-release`

Uses the strict evaluator-bound production operator and exposes:

- governed cohort creation;
- retrieval cohort creation;
- evaluator-bound advanced evaluation evidence;
- strict qualification;
- strict promotion verification.

### Governed data/evaluation commands

Installed commands include:

- `rigorousrag-grounded-import`;
- `rigorousrag-dynamic-publish`;
- `rigorousrag-benchmark-import`;
- `rigorousrag-benchmark-qualify`;
- `rigorousrag-corpus-import`;
- `rigorousrag-qrels`;
- `rigorousrag-retrieval-benchmark`;
- `rigorousrag-evaluator-contract`;
- `rigorousrag-benchmark-result`;
- `rigorousrag-runtime-bundle`.

High-volume import CLIs route through a format/size guard: very large monolithic JSON is rejected
before whole-document parsing and must be represented by a streaming format such as JSONL/TREC.
The underlying semantic adapters remain reusable library primitives.

---

## 16. Compatibility paths deliberately retained

The repository retains historical/research modules for reproducibility rather than deleting
source history. They are not equal production authorities.

Examples include:

- historical advanced-evaluation v1/v2 envelopes;
- older release-operator versions;
- generic primitive promotion evidence;
- the fenced one-pass dynamic trajectory preparation path;
- reusable non-strict evaluator-contract library entry points;
- lower-level generic checkpoint manager/research runner primitives.

The installed package scripts, artifact admission path, strict promotion assertion, runtime
bridge and runtime bundle do not use those weaker semantics as production authority.

---

## 17. Scale-honesty hardening

The continuation removed/avoided multiple cases where advertised high record ceilings would
have conflicted with in-memory implementation:

- authoritative advanced training JSONL datasets use compact byte-offset/line-digest indices
  rather than retaining every parsed record object;
- grounded canonical duplicate/cross-split identity accounting uses SQLite rather than giant
  Python sets;
- benchmark leakage qualification is disk-backed;
- qrels authority is disk-backed;
- exact query/result universes are proved with SQLite;
- result metric aggregation is disk-backed and streaming;
- installed large-data import surfaces reject oversized monolithic JSON;
- the accidental whole-JSON result CLI draft was collapsed into the streaming authority shim.

---

## 18. Filesystem and TOCTOU hardening

Source added or strengthened during the continuation includes:

- path-safe content-derived logical filenames;
- symlink/reparse/path authority checks;
- closed directory membership for caches/checkpoints/artifacts/runtime bundles;
- exact file-size/SHA checks;
- read-after-write/reconstruction verification;
- cache sealing before canonical receipt publication;
- frozen cache contracts during authoritative training;
- artifact re-hash at final admission;
- strict runtime bundle child verification;
- no trust in an existing content-addressed destination without re-verification.

---

## 19. Static convergence findings

The final source sweep checked the production chain for:

- `NotImplementedError`;
- `TODO`;
- `FIXME`;
- stale advertised release/evaluation paths;
- weak promotion assertions in production sinks;
- duplicate installed evaluation authorities;
- cache-content identity asymmetry;
- query/qrels/result universe asymmetry;
- evaluator metric-schema asymmetry;
- unsupported evaluator aggregation/scope semantics;
- incomplete/wrong-direction promotion policies;
- runtime bundle serialization/symlink issues.

No new major source family remained after the fixes above.

This static audit does **not** substitute for a Python import/compile/test run, which remains an
explicitly excluded empirical activity in this workstream.

---

## 20. What is no longer considered missing

Under the requested source-only definition, the following are implemented and are **not** on the
remaining-work list:

- major retrieval architecture families;
- grounded generator architecture;
- dynamic retrieval policy architecture;
- grounded losses;
- dynamic policy/value/off-policy losses;
- staged curricula;
- deterministic samplers/collators;
- governed local data schemas;
- final-manifest-bound supervision caches;
- concrete local supervision-provider adapters;
- cache sealing/content contracts;
- generic trainer execution source;
- authoritative training runners;
- optimizer/scheduler/AMP/DDP/accumulation/clipping source;
- validation evaluators;
- early-stopping metrics;
- exact checkpoint/resume source;
- best/latest source;
- checkpoint→inference-artifact export;
- artifact verification/admission;
- governed benchmark/query/corpus/qrels source;
- retrieval benchmark authority;
- evaluator contract authority;
- exact evaluation sample cohorts;
- streaming result evidence;
- result/evaluator metric-schema proof;
- advanced evaluation receipts/evidence;
- promotion policy qualification;
- poisoning/robustness evaluation contracts;
- authoritative runtime component bridge;
- restart-verifiable runtime-stack bundle;
- existing mutable runtime-stack fencing/rollback authority;
- package-level production command separation.

---

## 21. Remaining work — empirical/execution only

The remaining work is now execution and evidence generation rather than missing methodology
source:

1. install the actual dependency environment;
2. acquire/provide the real governed datasets locally;
3. provide the intended local model/tokenizer/retriever/teacher/reference artifacts;
4. execute canonical cache materialization;
5. execute actual training;
6. execute actual model inference;
7. execute governed benchmark suites;
8. produce real result JSONL artifacts;
9. run repeated seeds/ablations;
10. select promotion thresholds from actual empirical evidence;
11. measure latency/VRAM/RAM/disk/throughput;
12. run poisoning/robustness experiments;
13. execute live runtime-stack promotion/canary/rollback exercises;
14. run compile/import/static-type/lint/test suites if/when the testing exclusion is lifted.

Those activities can reveal empirical bugs, numerical issues or environment incompatibilities;
this ledger deliberately does not claim otherwise.

---

## 22. Final source-only conclusion

At the end of these continuation sweeps, the audit has **no known major missing source-level
architecture, model, objective, trainer, checkpoint/resume path, governed data path, evaluation
authority, promotion path, artifact-admission path or runtime-stack handoff** under the user's
stated exclusions.

The repository can therefore be called **source-complete under the explicit non-execution
boundary**, with the remaining uncertainty concentrated where it belongs: actual dependencies,
real data, real model execution, real training, real benchmark measurements and live deployment
exercise.
