# RigorousRAG strict source implementation audit — continuation

Date: 2026-08-18

This document continues and supersedes the *status conclusions* (not the historical record)
of:

- `docs/EXHAUSTIVE_MISSION_AUDIT_2026-08-01.md`
- `docs/SOURCE_IMPLEMENTATION_AUDIT_2026-08-17.md`
- `docs/STRICT_SOURCE_IMPLEMENTATION_AUDIT_2026-08-18.md`

The implementation baseline inspected at the beginning of this continuation was:

- `69e881aada2e585cf3ef55f345129f36f60e45e7` — `feat: authorize legal hold and key lifecycle mutations`

The source-expansion head before this document commit is:

- `ab0b680362fc7512d2c396c25f9f7a51656e597d` — `feat: add bounded dynamic RAG inference orchestration`

GitHub compare reports that head as **11 commits ahead and 0 behind** the baseline.  The
changes are intentionally concentrated in ten new/expanded source files.  No feature branch
or pull request was created for this continuation.

---

## 1. Scope and interpretation

The user explicitly asked for the strongest possible implementation audit while excluding:

- dataset downloading/acquisition;
- actual model training;
- actual model inference/execution;
- dependency installation/execution;
- test execution; and
- runtime benchmark execution.

For this audit, **source-complete** therefore means that the repository contains the code and
contracts required to express the architecture, algorithm, loss functions, training stages,
checkpoint/resume identity, evaluation methodology and serving/governance boundaries.  It
does **not** mean that a model has been trained, a benchmark has been downloaded, a cloud
service has been exercised, or an empirical metric has been observed.

The audit deliberately refuses to infer runtime success from source structure.  Runtime,
dependency, test, model and dataset claims remain unmade until those excluded activities are
actually performed.

---

## 2. Reconstructed original mission — exhaustive requirement families

The dated audit ledgers plus the live source establish the following durable mission.

### 2.1 Repository/process authority

The requested repository policy is:

1. preserve valid work from earlier PRs/branches;
2. consolidate the working project onto `main`;
3. commit future implementation directly to `main`;
4. do not create replacement PRs or feature branches;
5. do not force-push or rewrite history;
6. keep code, configuration, documentation, experiment specifications and lifecycle contracts
   consistent;
7. remove obsolete live branches only after their useful history has been preserved; and
8. verify the exact current `main` state before making a release/completeness claim.

### 2.2 Forensic engineering audit

The requested audit is not a README review.  It covers:

- every material source package, class, function, method, script, configuration surface and
  durable state transition;
- product intent and research intent;
- trust boundaries and authority ownership;
- data flow, control flow and provenance flow;
- persistence, retries, cancellation, recovery, replay and reconciliation;
- dead/incomplete/duplicated/unsafe source;
- mathematical correctness of ranking/fusion/calibration/evaluation/statistics;
- parser/OCR/upload/evidence/visual handling;
- concurrency, queues, timeouts, leases, fencing, resource ceilings and cleanup;
- agent/API/CLI/browser/container/deployment/release paths;
- privacy, tenant isolation, network boundaries, filesystem boundaries and disclosure;
- source-only fixes rather than merely documenting defects; and
- an updated ledger that distinguishes implemented source from unexecuted evidence.

### 2.3 Product/search mission

The product is expected to support both classic academic search and uploaded-document RAG,
including:

- query parsing/normalization;
- classic external/search-provider result handling;
- uploaded private corpora;
- durable ingestion and document lifecycle;
- semantic/hierarchical evidence retrieval;
- agentic research workflows;
- citation authority and BibTeX/export paths;
- comparison/evidence matrices;
- contradiction analysis;
- scientific evidence synthesis;
- reproducible research workspaces/capsules; and
- service/browser/CLI/deployment operation.

### 2.4 Ingestion, parsing, OCR and retained evidence

Required source families include:

- strict upload limits and owner-scoped identities;
- safe filename/path handling;
- symlink/reparse/path escape rejection;
- bounded PDF/DOCX/text/archive parsing;
- page/section/chunk/block/region provenance;
- OCR with bounded rasterization/pixels/output;
- retained visual evidence;
- table/chart/formula/figure/reference structure;
- immutable parser/model identities;
- malware/parser isolation hooks;
- durable queued/processing/finalizing/success/failed state;
- retries/backoff/scheduler/replay/reconciliation; and
- privacy masking and bounded public error state.

### 2.5 Lifecycle, storage and authority

The source must express:

- source/version authority;
- ingest attempt identity;
- staging and atomic publication;
- immutable/current heads;
- delete/tombstone/retention/legal-hold behavior;
- reindex/adoption/reconciliation;
- cache invalidation;
- provenance and audit events;
- snapshot/restore;
- backup/DR/multiregion contracts;
- residency/transfer policy;
- cutover, rollback and compensation; and
- fenced mutable authority.

### 2.6 Retrieval/ranking matrix

The requested retrieval matrix includes:

- lexical/BM25;
- dense bi-encoder retrieval;
- sparse learned retrieval, including SPLADE/uniCOIL-style source;
- hybrid retrieval;
- independent-corpus retrieval and fusion;
- weighted/programmable RRF;
- score calibration;
- ColBERT/late-interaction MaxSim;
- page-native/multimodal late interaction;
- cross-encoder and listwise reranking;
- staged/cascaded reranking;
- metadata filters and hard caps;
- diversity/deduplication/source balancing;
- contextual/hierarchical/parent-document/sentence-window retrieval;
- claim/evidence retrieval;
- contradiction-aware retrieval;
- GraphRAG/evidence-graph traversal;
- multi-hop/agentic retrieval;
- multilingual/domain-specific profiles;
- uncertainty/calibration/abstention; and
- cost/latency/risk-aware routing.

### 2.7 Query transformation and adaptive planning

The mission includes:

- multi-query expansion;
- HyDE-style transformation;
- step-back transformation;
- decomposition/multi-hop plans;
- acronym/entity/time normalization;
- multilingual transformations;
- temporal/freshness handling;
- learned domain/query classification;
- learned retrieval-plan selection;
- candidate-plan ranking/repair;
- adaptive budgets; and
- server-owned routing authority rather than free-form model tool authority.

### 2.8 Learned retrieval/adaptation/training

Required training source includes:

- dense contrastive learning;
- hard-negative mining/refresh;
- sparse retrieval objectives and sparsity/FLOPS regularization;
- ColBERT/late-interaction learning;
- pointwise/pairwise/listwise reranking;
- distillation;
- learned domain/query classification;
- learned query-plan selection;
- learned cross-profile fusion;
- staged training;
- deterministic data/sampler/collator identities;
- exact optimizer/scheduler/AMP state;
- Python/PyTorch/CUDA RNG state;
- source commit, dataset manifest and training-config identities;
- stage/cursor-aware resume;
- content-addressed checkpoints; and
- best/latest pointer semantics.

### 2.9 Evidence, citation and semantic authority

The source must separate structural provenance from semantic support and include:

- server-owned evidence/citation ids;
- claim-to-evidence linkage;
- page/chunk/block/region lineage;
- support/entailment scoring;
- contradiction scoring;
- citation support/completeness diagnostics;
- source trust/retraction governance;
- review/adjudication lineage;
- DLP/release boundaries;
- prompt-injection/retrieved-content trust; and
- fail-closed model-output schemas.

### 2.10 Scientific/document intelligence

The mission includes:

- deterministic scientific document structure;
- reading order and geometry;
- section/reference relationships;
- table topology;
- figure/caption relationships;
- formulas/equations;
- chart/table native evidence;
- scientific NER/linking adapters;
- PICO/PECO/PICOS extraction;
- effect size, confidence interval, standard error, sample/event normalization;
- risk-of-bias/certainty semantics;
- numerical consistency checks;
- causal DAG semantics;
- fixed/random-effects synthesis; and
- multi-document scientific evidence synthesis.

### 2.11 Evidence graph and domain extensions

The repository mission also includes:

- document-to-evidence graph construction;
- cross-document claims/entities/relations;
- graph traversal/reranking;
- evidence matrices;
- contradiction groups;
- citation chasing;
- scientific research capsules/workspaces; and
- hydrology/geospatial lineage, including CHIRPS and HEC-HMS/HEC-RAS-oriented source
  contracts without coupling the core to one execution backend.

### 2.12 Evaluation, reproducibility and statistics

Required source includes:

- exact dataset/version/split manifests;
- licensing decisions before promotion;
- leakage checks;
- repeated current-vs-shadow seeds;
- ablations;
- retrieval metrics;
- citation metrics;
- semantic support/contradiction metrics;
- calibration/selective-risk metrics;
- conformal retrieval/abstention;
- paired bootstrap/permutation tests;
- effect size;
- Holm/BH multiple-comparison correction;
- latency/resource reporting;
- interleaving/online experiment contracts;
- immutable result receipts; and
- promotion gates.

### 2.13 Human review and active learning

The requested source includes:

- uncertainty/disagreement queues;
- adjudication;
- reviewer identity/lineage;
- review activation;
- invalidation/recompute;
- outbox/reconciliation; and
- production reachability rather than orphan review utilities.

### 2.14 Security, governance and operations

The audit mission includes:

- tenant/owner authority derived server-side;
- API authorization;
- operator authorization for sensitive mutations;
- strict input/output bounds;
- SSRF/network controls;
- prompt-injection/retrieved-content distrust;
- DLP/model-input release;
- KMS/HSM-compatible envelope/key lifecycle;
- legal hold;
- data residency;
- leases/fencing;
- DLQ/reconciliation;
- worker control;
- runtime-stack promotion/rollback;
- artifact/supply-chain admission;
- local executable/tree/model/OCR identity binding;
- disclosure/attestation;
- durable operator audit/export;
- non-root/container/release constraints; and
- exact publication authority.

### 2.15 API/browser/CLI/release mission

The source mission includes:

- bounded service interfaces;
- no browser innerHTML/third-party-script authority leaks;
- terminal-safe CLI output;
- safe readiness behavior;
- reproducible/pinned dependency and CI metadata;
- immutable deployment/runtime identities; and
- release verification against the exact head.

---

## 3. What the prior PR/audit work had already implemented

The earlier hardening and audit series already made the repository substantially more than a
prototype RAG application.  The following are present in the existing source and were **not
reimplemented in this continuation** merely to create churn.

### 3.1 Identity/API/bounded execution

Earlier work already established server-owned tenant identity, request-scoped agents,
bounded executors, request/model/id/evidence/citation/metadata/response limits, strict HTTP
framing and capacity semantics for timed-out work.

### 3.2 Upload/source/durable ingestion

Earlier work already established owner-scoped randomized upload handling, descriptor-relative
path safety, symlink/reparse rejection, exact byte caps, durable SQLite ingestion states,
atomic claims, attempts/backoff/scheduling/replay, bounded public errors and keyset/high-water
scans.

### 3.3 Parsing/OCR/privacy/evidence

Earlier work already established bounded PDF/DOCX/text/OCR/page/pixel/archive/section handling,
immutable parser snapshots, content/owner document identities, privacy masking and owner-scoped
visual evidence with byte/pixel/payload validation.

### 3.4 Retrieval/provenance/search

Earlier work already established owner-scoped vector operations, canonical uploaded citation
metadata, strict adapters, URL hardening, stable classic search reload, BM25/dense/hybrid,
learned sparse retrieval, ColBERT/MaxSim, cross/listwise reranking, contextual/hierarchical
retrieval, query transforms, GraphRAG, multihop, multimodal/scientific/temporal retrieval,
learned planning and programmable fusion.

### 3.5 Scientific/evidence intelligence

Earlier work already established scientific document IR, table/formula/figure/reference
linking, document/evidence graphs, semantic support/contradiction, PICO/PECO/PICOS,
effect-size normalization, evidence synthesis, risk-of-bias/certainty/causal semantics and
hydrology/geospatial research lineage.

### 3.6 Training/checkpoint source

Earlier work already provided hard-negative pipelines, dense/sparse/late-interaction/reranking
losses, distillation, staged PyTorch training, deterministic seeding, CPU/CUDA/MPS selection,
AMP, gradient accumulation/clipping, AdamW, schedulers, optional DDP, evaluation/early
stopping, hard-negative refresh, exact content-addressed checkpoints and exact resume of
model/optimizer/scheduler/scaler/RNG/sampler/collator/trainer cursor identities.

### 3.7 Evaluation/governance/operations

Earlier work already provided governed dataset manifests/leakage checks, repeated experiment
orchestration, semantic-support metrics, calibration/conformal logic, paired statistical
inference, active learning/adjudication, source-trust activation/invalidation/reconciliation,
backup/DR/multiregion/residency, runtime-stack authority, DLP, operator authorization and
supply-chain/artifact admission.

### 3.8 Post-strict-audit source that existed before this continuation

The previous strict audit document was not the final live source state.  After it, `main`
added additional authority around:

- content-bound generation release;
- authoritative generation composition;
- admitted local scientific models;
- admitted local executables;
- admitted local artifact trees;
- admitted Tesseract OCR;
- atomic fenced generation publication; and
- authorization for legal-hold/key-lifecycle mutations.

This is why the continuation used the live head rather than treating an older `TODO.md` or
an older “complete” audit sentence as authoritative.

---

## 4. New source gaps identified in this continuation

After eliminating stale TODOs and duplicate ideas, four material research/source families
remained useful and non-overlapping with the existing implementation.

### 4.1 Grounded generator-side RAG learning

The repository had extensive retriever/reranker/planner/fusion training, but did not yet
contain a first-class generator-side objective suite for jointly learning:

- token generation;
- claim-to-evidence citation attribution;
- support/contradiction discrimination;
- abstention/reflection actions;
- unsupported-content unlikelihood;
- grounded preference optimization;
- teacher distillation; and
- language-model-supervised retriever coupling.

### 4.2 Generation-time dynamic retrieval learning and inference

The repository already had query/plan-level adaptation, but not a token/step-generation
closed-action policy for **continue vs retrieve vs verify vs abstain vs stop**, nor an
information-need selector and bounded iterative runtime.

### 4.3 Citation refinement after generation

The repository validated server-owned citations, but did not yet contain a deterministic
post-generation refinement layer for removing weak/redundant citations, adding stronger
allowlisted evidence, enforcing independent-source support, or refusing unresolved claims
without changing answer text.

### 4.4 RAG poisoning/robustness evaluation

The repository treated retrieved content as untrusted data and defended prompt/tool
authority, but did not yet expose a matched clean/attacked evaluation contract for corpus
poisoning, duplicate/source flooding, citation spoofing, contradiction injection, stale
content and related evidence-manipulation failure modes.

---

## 5. Research grounding used for the expansion

The expansion was grounded in primary paper sources rather than invented method names.
Planning references include:

- Self-RAG — arXiv:2310.11511 — retrieval on demand plus self-reflection signals;
- REPLUG — arXiv:2301.12652 — retrieval-augmented black-box LM and LM-supervised retriever
  adaptation;
- DRAGIN — arXiv:2403.10081 — generation-time decisions about when/what to retrieve;
- BRIGHT — arXiv:2407.12883 — reasoning-intensive retrieval benchmark;
- RAGTruth — arXiv:2401.00396 — annotated hallucination/grounded-generation corpus;
- `On the Capacity of Citation Generation by Large Language Models` — arXiv:2410.11217 —
  citation generation/evaluation and generate-then-refine analysis; and
- `Benchmarking Poisoning Attacks against Retrieval-Augmented Generation` —
  arXiv:2505.18543 — matched poisoning/defense benchmark methodology.

These names are planning references only.  They do not imply that any dataset/model has been
downloaded, that any external license has been approved, or that the repository reproduces a
paper result.

---

## 6. Implementation added directly to `main`

### 6.1 `training/grounded_generation.py`

Commit: `b55afe928235fcade4dfcd4ba83da6a782076872`

Implemented:

- immutable `GroundedGenerationArchitectureConfig`;
- explicit training stage kinds (`supervised`, `attribution`, `grounding`,
  `retriever_coupling`, `preference`, `reflection`, `joint`);
- immutable stage/objective/plan digests;
- base-model, tokenizer, dataset-manifest, source-commit, optional retriever-stack and
  teacher-model bindings;
- checkpoint-stage/cursor identity contracts;
- executable PyTorch auxiliary heads for citation attribution, support, contradiction,
  abstention and reflection;
- masked token NLL;
- sequence log-probabilities;
- citation pointer cross-entropy;
- binary support/contradiction/abstention supervision;
- unsupported-token probability-mass unlikelihood;
- DPO-style grounded preference loss;
- teacher-token KL distillation;
- LM-supervised retriever KL; and
- backend artifact/tokenizer-plan matching.

### 6.2 `training/dynamic_retrieval_policy.py`

Commit: `fdf022c349277bc5d61e969a6b0aa7bba5837efc`

Implemented:

- closed actions: `continue`, `retrieve`, `verify`, `abstain`, `stop`;
- server-owned bounded policy features for entropy/margin/evidence sufficiency/support/
  contradiction/citation coverage/novelty/unresolved entities/temporal uncertainty/budgets;
- hard generation/retrieval/verification/consecutive-retrieval budgets;
- immutable architecture/objective/stage/training-plan identities;
- PyTorch policy/value controller;
- PyTorch information-need token selector;
- action imitation loss;
- information-need BCE;
- retrieval-value Huber regression;
- bounded off-policy policy-gradient loss with importance clipping;
- retrieval/verification/abstention expected costs;
- entropy bonus;
- combined dynamic-policy objective;
- deterministic server-owned action masks;
- runtime state transitions; and
- deterministic bounded information-need span selection.

### 6.3 `evaluation/rag_robustness.py`

Initial commit: `e73daa1497ad9d1ca9fdafdffdae4ac3e2e4ce56`

Correction commit: `61bbc809d363ce7e649f2ab00ca7759334c01b24`

Implemented:

- attack taxonomy for corpus poisoning, contradiction injection, flooding, duplicate
  amplification, source impersonation, citation spoofing, indirect prompt injection, stale
  evidence, cross-context contamination, multimodal poisoning and agent-evidence poisoning;
- digest-bound matched clean/attacked cases;
- clean/attacked retrieval success;
- clean/attacked supported-answer rates;
- attacked target retrieval/citation rates;
- answer attack success rate;
- robust-or-abstain rate;
- clean/attacked abstention;
- support retention;
- contradiction increase;
- suspicious top-k, duplicate-cluster and source-concentration diagnostics;
- independent-support diversity;
- per-attack slices;
- candidate source-trust/provenance/injection/contradiction risk signals;
- conservative allow/review/block defense policy; and
- robustness promotion gates.

The correction makes zero-support clean baselines non-degrading: when clean support is zero,
there is no positive clean support to lose, so the retention ratio is one rather than being
misreported as total degradation.

### 6.4 `evaluation/advanced_dataset_proposals.py`

Commit: `a55e751fa5a19fbba97a9b6753ae1d80382bc501`

Added planning-only proposal families for:

- BRIGHT;
- RAGTruth;
- ASQA/ELI5-style long-form attribution;
- KILT;
- FreshQA-style temporal/freshness QA;
- QASC/StrategyQA reasoning;
- repository-owned matched clean/poisoned RAG cases;
- repository-owned dynamic-RAG episodes; and
- repository-owned grounded preference data.

These remain `DatasetProposal` objects, not promotable `DatasetManifest` claims.  Exact
version, bytes/checksum, licensing, split identities, loader/transformation identities and
leakage evidence are intentionally deferred until real acquisition, which is excluded here.

### 6.5 `evaluation/dynamic_rag_metrics.py`

Commit: `f7afdd7c336a98c7aa3b1ecf9dd874bcf7f8dbde`

Implemented episode/step evaluation for:

- action accuracy;
- retrieve precision/recall;
- retrievals per episode;
- useful/unnecessary retrieval rate;
- retrieval gain/cost;
- step latency;
- generation token usage;
- final supported rate;
- contradiction rate;
- abstention rate;
- answer utility;
- oracle regret; and
- selected-action calibration diagnostic.

### 6.6 `tools/citation_refinement.py`

Commit: `cb807f868d472296b04a251abc4b2679709bcc78`

Implemented a deterministic server-owned generate-then-refine citation algorithm that:

- binds answer/claim/evidence identities by digest;
- rejects evidence outside the server-owned allowlist;
- uses externally governed support/contradiction/quality scores;
- retains useful original citations when possible;
- greedily adds evidence for support and independent-source requirements;
- penalizes contradictory candidates;
- caps citations per claim;
- records added/removed citations;
- separates supported/partially-supported/contradicted/unresolved claims;
- emits explicit review/abstain actions; and
- never rewrites answer text or invents evidence.

### 6.7 `training/advanced_rag_steps.py`

Commit: `02fb87da3535a60ca261b9c6c5333dfa0c6ca6cf`

This closes the “losses exist but are not trainable through the repository engine” seam.
It adds:

- `GroundedGenerationStep`;
- `DynamicRetrievalPolicyStep`;
- task-specific batch/output contracts;
- per-component metrics;
- DPO/reference-log-prob wiring;
- teacher/retriever coupling wiring;
- policy-gradient logged-action wiring;
- action-cost/entropy wiring; and
- `GroundedTrainingPlan` / `DynamicPolicyTrainingPlan` → generic `TrainerConfig` adapters.

The existing `TorchTrainingEngine` therefore supplies optimizer, scheduler, AMP, DDP,
accumulation, clipping, evaluation, early stopping, hard-negative hooks and exact checkpoint
resume instead of duplicating those mechanisms.

### 6.8 `training/advanced_rag_models.py`

Commit: `481dd29e29cee24a483666d13d17e590c9ad9ef5`

Implemented:

- an injected base-LM training wrapper;
- differentiable claim hidden-state gathering;
- differentiable evidence encoding/pooling;
- grounding/citation auxiliary-head composition;
- optional chosen/rejected preference forwards;
- optional injected retriever coupling; and
- a composed dynamic policy/value + information-need selector module.

The wrapper deliberately does not select or download a pretrained model revision.

### 6.9 `orchestration/refined_generation_publication.py`

Commit: `1c55e9184a08f458181af12a75d95fdf7b0f9b9f`

Implemented a fail-closed composition boundary so citation refinement is not merely an
orphan utility:

- binds the exact authoritative generation receipt;
- binds answer digest;
- binds the exact server-owned evidence-id universe;
- verifies claim citations were actually emitted by the grounded model output;
- binds the refinement receipt;
- allows publication only when no claim requires review/abstention; and
- atomically re-checks runtime stack/fence in the same SQLite authority database before a
  digest-only refined publication record is inserted.

### 6.10 `orchestration/dynamic_rag_runtime.py`

Commit: `ab0b680362fc7512d2c396c25f9f7a51656e597d`

Implemented the missing bounded inference/control loop for the dynamic policy:

- feature-provider contract;
- closed-action policy provider;
- generation-chunk provider;
- information-need query builder;
- retrieval-query release boundary;
- retrieval provider;
- evidence-admission boundary;
- verification provider;
- immutable provider contract digest;
- deterministic server-side action selection from the allowed set;
- hard iteration/token/retrieval/verification/evidence/character limits;
- retrieval-query blocking → abstention;
- evidence identity collision checks;
- cumulative evidence caps;
- deterministic trace digests; and
- terminal stop/abstain result receipts.

The runtime explicitly does **not** grant the learned policy direct retrieval/tool authority,
and production publication must still pass the repository's authoritative generation/DLP/
closed-schema/fence/publication path.

---

## 7. Training methodology now expressible in source

### 7.1 Grounded generator curriculum

A complete staged source curriculum can now be represented as:

1. **Supervised generation** — token NLL on evidence-conditioned answers.
2. **Attribution** — claim-to-evidence citation pointer supervision.
3. **Grounding** — support, contradiction, abstention and unsupported-token unlikelihood.
4. **Reflection** — closed reflection/action classification.
5. **Retriever coupling** — distill generator document utility into retriever logits.
6. **Preference alignment** — DPO-style grounded chosen/rejected response pairs.
7. **Teacher distillation** — token-distribution distillation when an admitted teacher is
   available.
8. **Joint stage** — weighted combination under an immutable objective digest.

Every stage carries max optimizer steps, checkpoint cadence, learning rate and objective
identity.  The plan binds the exact base model, tokenizer, dataset manifest, source commit
and optional retriever/teacher identities.

### 7.2 Dynamic retrieval curriculum

A complete source curriculum can now be represented as:

1. **Imitation** — learn retrieve/continue/verify/abstain/stop from labeled/logged actions.
2. **Information-need selection** — token-level BCE over the portion of generated context
   that expresses the unresolved need.
3. **Retrieval value learning** — predict realized counterfactual retrieval gain.
4. **Off-policy refinement** — bounded importance-weighted policy-gradient updates from
   logged episodes.
5. **Cost regularization** — expected retrieval/verification/abstention cost.
6. **Entropy regularization** — exploration without granting action authority.
7. **Joint stage** — combine imitation/value/need/RL/cost under one immutable objective.

The runtime hard budget remains authoritative even if the learned policy scores a forbidden
action highly.

### 7.3 Exact checkpoint/resume path

The new plan adapters feed the existing `TorchTrainingEngine`, whose checkpoint manager
already persists/restores:

- model weights;
- optimizer state;
- scheduler state;
- AMP scaler;
- trainer/stage cursor;
- Python RNG;
- PyTorch RNG;
- CUDA RNG where relevant;
- sampler state;
- collator state;
- source commit;
- exact trainer-config digest;
- dataset-manifest digest;
- model-architecture/plan identity;
- parent checkpoint identity;
- stage boundary; and
- metric snapshot.

Resume validates those identities rather than silently accepting a checkpoint from another
source/configuration/dataset.

---

## 8. Experiment matrix now supported in source

The following source-level experiment families can be configured without adding another
algorithm implementation:

### Retrieval/planning ablations

- BM25 vs dense vs sparse vs hybrid;
- RRF/fusion variants;
- ColBERT/late interaction;
- cross/listwise reranking;
- contextual/hierarchical retrieval;
- GraphRAG;
- multi-query/HyDE/step-back/decomposition;
- static query-level routing vs learned routing;
- static retrieval cadence vs dynamic generation-time retrieval; and
- dynamic policy feature/action/cost ablations.

### Generator/grounding ablations

- token NLL only;
- + citation attribution;
- + support/contradiction;
- + abstention/reflection;
- + unsupported unlikelihood;
- + teacher distillation;
- + LM-supervised retriever coupling;
- + grounded DPO; and
- joint combinations.

### Citation ablations

- model citations only;
- citation refinement;
- refinement without diversity bonus;
- independent-source thresholds;
- contradiction thresholds;
- citation caps; and
- review/abstention policies.

### Robustness ablations

- clean vs matched attacked cases;
- source-trust policy;
- provenance-integrity gates;
- injection-risk gates;
- contradiction-risk gates;
- duplicate concentration limits;
- source concentration limits; and
- independent-source requirements.

### Reporting dimensions

- retrieval effectiveness;
- citation correctness/completeness;
- semantic support/contradiction;
- abstention/selective risk;
- dynamic action quality;
- retrieval utility/cost/latency;
- robustness/attack success;
- support retention;
- calibration;
- resource usage; and
- paired statistical significance/effect-size/correction.

---

## 9. Dataset families — implemented planning versus excluded acquisition

### Existing governed proposal coverage

The canonical dataset-governance source already includes planning families for:

- BEIR;
- MS MARCO;
- Natural Questions / TriviaQA;
- LoTTE / MIRACL / Mr.TyDi / mMARCO;
- SciFact / SCIDOCS / TREC-COVID / NFCorpus;
- HotpotQA / 2WikiMultiHopQA / MuSiQue;
- DocVQA / InfographicVQA / PubTables-1M / PubLayNet / ChartQA; and
- repository-owned adversarial corpora.

### New planning coverage

The continuation adds BRIGHT, RAGTruth, long-form attribution, KILT, temporal QA,
reasoning/multihop, matched poisoning, dynamic-policy episodes and grounded preference data.

### Deliberately not claimed

No proposal is a real experiment until an exact dataset manifest binds:

- exact version/revision;
- immutable content digest;
- exact split digests and record ids;
- loader/transformation identities;
- verified-allowed licensing evidence; and
- leakage findings.

Those actions require real dataset acquisition/review and are explicitly outside this task.

---

## 10. What remains after this continuation

### 10.1 Material source implementation gaps found by this audit

**None remain that this continuation identified inside the user's defined source-only scope.**

That statement is intentionally narrower than “the system has been proven correct.”  It
means that every material requirement recovered from the dated mission audits has source
coverage, and the additional generator/dynamic/citation/robustness research families added in
this continuation have architecture, algorithms, losses, training/evaluation contracts,
checkpoint integration and bounded authority surfaces.

### 10.2 Work that remains only because the user explicitly excluded execution

The following are still real project work, but they are **not missing source implementation**
under this request:

1. acquire/download selected datasets;
2. review real licenses and construct exact `DatasetManifest` objects;
3. tokenize/materialize real training records;
4. install/resolve training/inference dependencies;
5. load admitted pretrained model/tokenizer artifacts;
6. execute training stages;
7. save real checkpoint artifacts produced by those runs;
8. resume real runs and verify numerical continuation;
9. run inference;
10. run unit/integration/system/security tests;
11. run benchmark/evaluation suites;
12. collect empirical latency/memory/throughput/cost metrics;
13. run repeated seeds/ablations/statistical comparisons;
14. execute poisoning/robustness benchmark cases;
15. calibrate thresholds from held-out data;
16. train/freeze/promote learned routing/generator/dynamic-policy artifacts;
17. exercise real KMS/HSM/cloud/multiregion backends;
18. perform real disaster-recovery drills; and
19. verify the exact deployment/release artifact produced from the chosen commit.

No numeric performance claim should be made before those activities occur.

### 10.3 Deployment-specific adapter work

Some integrations are intentionally represented as typed injected protocols because the
concrete backend is deployment-specific: external model runtime, retriever, KMS/HSM,
multiregion object store, dynamic feature provider, retrieval-query release provider,
evidence admission provider and verifier.  Providing one arbitrary vendor implementation is
not necessary for source completeness and would weaken backend neutrality.  A deployment
must instantiate those interfaces with admitted implementations before execution.

---

## 11. Verification performed in this continuation

Within the user-requested non-execution boundary, this continuation verified:

- the live repository rather than stale conversation claims;
- the prior dated audit ledgers;
- the post-audit commit sequence;
- direct-to-main ancestry;
- a baseline→head compare showing 11 commits ahead / 0 behind before this document;
- no `NotImplementedError` source hits in the repository search used for this pass;
- the generic trainer's checkpoint path directly binds `TrainerConfig.digest`, source commit,
  dataset manifest and model architecture and restores those exact expected identities;
- authoritative generation already rechecks runtime stack/fence after model invocation;
- the existing publication ledger serializes publication with runtime-stack authority;
- the new refined publication path performs the same fail-closed stack/fence recheck; and
- the new dynamic runtime keeps action selection, query release and evidence admission under
  server-owned boundaries.

A public raw-file fetch path was unavailable from the execution environment for a local AST
compile pass.  Therefore this document does **not** claim that Python modules were imported,
compiled, unit-tested or executed.  That is consistent with the user's explicit exclusion of
test/dependency/model execution, but it is recorded rather than hidden.

---

## 12. Repository-state rule

At the beginning of this continuation, GitHub reported:

- `main` as the only live branch; and
- PRs #1–#4 closed/merged.

This continuation created no branch and no PR.  Final branch/PR/head state must still be
re-read after this document commit before the final user-facing status is asserted.

---

## 13. Final source-only conclusion

Relative to the recovered RigorousRAG mission, the repository was already source-complete
across the classic retrieval, lifecycle, scientific, provenance, security, governance,
statistics and production-authority families covered by the August 18 strict audit.  The
continuation found that generator-side grounded learning, generation-time dynamic retrieval,
post-generation citation refinement and matched poisoning robustness were worthwhile
non-duplicate extensions.  Those extensions are now implemented on `main` with executable
source-level algorithms and explicit authority/checkpoint/evaluation contracts.

The remaining uncertainty is empirical, not a known missing algorithm/source family:
actual dependencies, real datasets, real model artifacts, training, inference, tests,
benchmarks, calibration, production backend exercises and release verification have not been
run in this source-only task and must not be represented as completed.
