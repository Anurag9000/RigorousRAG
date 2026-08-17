# RigorousRAG source-implementation audit — 2026-08-17

## Status of this document

This is a dated source-level audit of the live `main` branch performed after re-reading the historical audit material, the repository capability/TODO ledgers, later August commit history, and the current source tree. It supersedes unchecked boxes in older ledgers **for source-implementation status only** where later commits or the commits listed below demonstrably implemented the item.

It does **not** claim that models were trained, datasets were downloaded, benchmarks/tests were run, infrastructure was deployed, or numerical performance claims were measured during this audit. Those actions were explicitly excluded from the requested scope.

The repository was changing concurrently during this audit. All work here was therefore additive or narrowly corrective on fresh `main`; no history was rewritten and no historical PR progress was discarded.

## User mission reconstructed

The original RigorousRAG mission is broader than “make a RAG demo.” The requested end state is a research-grade and production-grade retrieval/evidence platform with all of the following source-level capabilities.

### A. Repository/process authority

- direct-to-`main` development only for this continuation;
- preserve all useful earlier PR work and history;
- no throwaway feature branches or new PRs;
- keep implementation, contracts, governance and documentation coherent at one authoritative head;
- distinguish source completeness from execution/testing/model-training completeness.

### B. Forensic engineering audit

Audit the repository rather than only the happy path, including:

- persistence, crash recovery, idempotency and stale-state recovery;
- dead/incomplete/unsafe code and mathematical correctness;
- retrieval, ranking, reranking, calibration and evaluation semantics;
- tenant/owner scope, provenance, citations, filesystem/network boundaries and secrets;
- upload/parser/OCR/malware/prompt-injection boundaries;
- queues, leases, fencing, retries, timeouts, cancellation and distributed work;
- API/CLI/browser/container/dependency/deployment/release behavior;
- exact-head evidence rather than documentation assertions.

### C. Authoritative ingestion/lifecycle

- owner/tenant-scoped ingestion;
- immutable generations and source identity;
- authoritative multi-store lifecycle/reconciliation;
- durable intent/outbox/replay/repair semantics;
- retention, adoption, reindex and migration/cutover workflows;
- legal hold, secure retirement/deletion and auditable lineage.

### D. Retrieval architecture matrix

- lexical/sparse/dense/hybrid retrieval;
- learned sparse retrieval such as SPLADE/uniCOIL-style scoring;
- late interaction such as ColBERT-style MaxSim;
- cross-encoder and listwise reranking;
- independent-corpus retrieval and rank fusion;
- source/document caps, filters and budgeted reranking cascades;
- contextual/parent-child/hierarchical retrieval;
- multi-query, HyDE, step-back, acronym/entity, multilingual and temporal transforms;
- graph/GraphRAG retrieval;
- multi-hop and agentic retrieval;
- multimodal text/table/chart/figure/page retrieval;
- uncertainty, calibration and abstention rather than forced answers.

### E. Learned adaptation/training

- domain/query classification;
- learned query-plan/retrieval strategy selection;
- hard-negative mining and curricula;
- sparse/late-interaction/reranker losses;
- optional teacher distillation;
- staged training plans;
- resumable checkpoint state including model, optimizer, scheduler, RNG and data cursor identity;
- immutable data/model/config/source-commit binding for resume and promotion.

### F. Citation/provenance/evidence authority

- claim-to-evidence provenance;
- generation/page/chunk/block/region anchors;
- citation validation and support scoring;
- contradiction detection;
- source/evidence lineage and immutable corrections;
- human-review gates where automated extraction is not sufficient.

### G. Scientific/document intelligence

- scientific document structure, reading order and geometry;
- table topology and merged cells;
- formula representations;
- figures/panels/captions/cross-references;
- PICO/PECO/PICOS-style structured questions;
- methods/populations/interventions/exposures/comparators/outcomes/limitations;
- effect estimates, confidence intervals/SE/sample/event information and synthesis scales;
- risk-of-bias/certainty fields with evidence and review lineage;
- multi-document scientific synthesis/contradiction/method/effect/citation tasks;
- quality gates so weak OCR/layout extraction cannot silently become authoritative evidence.

### H. Hydrology/geospatial research workflows

- hydrology/geospatial evidence and lineage;
- CHIRPS/rainfall and basin evidence where configured;
- HEC-HMS/HEC-RAS typed/scenario/replay/provenance workflows;
- reproducible research capsules/workspaces/artifact lineage;
- backend-neutral execution and distributed recompute contracts.

### I. Evaluation/reproducibility/statistics

- exact dataset/version/license/checksum/split manifests;
- leakage checks;
- current-vs-shadow repeated runs with fixed seeds and query ordering;
- ablations and historical baselines;
- retrieval/citation/semantic/calibration metrics;
- paired bootstrap and paired randomisation/permutation tests;
- multiplicity control (Holm and BH/FDR);
- resource observation hooks;
- promotion gates that distinguish statistical and practical improvement;
- portable/replayable experiment evidence.

### J. Security/governance/operations

- SSRF-safe network ingestion;
- isolated parser/malware boundaries;
- authentication/authorization and tenant isolation;
- rate/resource limits;
- durable leases/fencing/dead-letter/reconciliation;
- migration/cutover fencing/rollback/compensation;
- key management/envelope-encryption contracts backed by real KMS/HSM providers;
- privacy-safe operator audit exports;
- pause/resume/cancel semantics for active workers;
- review/disclosure/attestation governance;
- backup/restore/retention/object-storage/DR/multi-region controls.

## Why the old TODO is no longer authoritative by itself

`docs/TODO.md` and the older capability ledger correctly captured gaps when they were written, but later August work filled many items without synchronizing every checkbox. Examples verified in commit history include:

- `9e1e17d2362b27dd1eb91893c1344e05b8a938e9` — unified store reconciliation, adoption and reindex plans;
- `061bce7337d6ef542d6669877fcaf9c7e4352455` — backend-neutral migration cutover coordination;
- `8f263ce82fb2f08fc4c1ab1b364f717082c6eaeb` — dimension-changing blue/green cutovers;
- `d40f4f838310cebedc9e82c4b567b8768c52ea72` — same-dimension local cutovers;
- later runtime-calibrator, retention, object-storage, PostgreSQL persistence, review-attestation, distributed hydrology, GraphRAG and multimodal commits that post-date the earlier ledger.

Therefore an unchecked old box was treated as a **hypothesis**, then searched against current code/commit history before new code was added.

## Source-level work implemented in this continuation

### 1. Learned retriever/reranker training and resumability

Commit `b223fb98d2ca9c2234248cd4f4602cb01284e354` added `tools/learned_retriever_training.py`:

- SPLADE/uniCOIL-style contrastive retrieval objectives;
- SPLADE L1/FLOPS regularisation;
- ColBERT/dense in-batch contrastive loss;
- pairwise softplus and listwise cross-entropy reranker losses;
- temperature-scaled teacher distillation;
- hard-negative curriculum metadata;
- staged immutable training plans;
- content-addressed checkpoint manifests;
- model/optimizer/scheduler/RNG/data-cursor digests;
- strict resume compatibility with model/data/config/code identity;
- injected backend protocol so importing the module does not train or download anything.

### 2. Scientific evidence semantics

Commit `5db280a530a93a72443f6d0e0fdb9cff8e6f1b2b` added `scientific/evidence_semantics.py`:

- PICO, PECO, PICOS and freeform question schemas;
- precise evidence spans and confidence-bearing extracted values;
- methods/population/intervention/exposure/comparator/outcome/limitation fields;
- effect estimates for ratios/differences/SMD/correlation/proportion/generic effects;
- CI, SE, sample size and event metadata;
- log-ratio and Fisher-z synthesis transforms;
- risk-of-bias domains, certainty and human-review state;
- support/refute/qualify/limit/conflict/derived evidence relations;
- immutable human correction lineage keyed to prior-value digests.

### 3. Paired statistics and governed promotion

Commit `d58bcb9ec4efb77ce68b8e4e2f6b1d551e99cbf3` added paired bootstrap/permutation, paired standardized effects, Holm correction, Benjamini-Hochberg correction and multi-metric promotion gates in `evaluation/statistical_experiments.py`.

Static review found a zero-variance paired-effect edge case; commit `fd731962d8f0ff3b3d64a6d797b6f0a32c508b01` changed undefined Cohen's dz to an explicit `None` instead of an invalid infinity.

### 4. Exact dataset/split governance

Commit `3946b29efef4c7b8fbe1f1f231e543a34855c914` added `evaluation/dataset_governance.py`:

- exact dataset artifact SHA-256;
- exact version/source locator;
- reviewed license status/evidence;
- loader/transformation identity;
- split content/record/query/document/source-group digests;
- intended/forbidden uses, PII/safety/limitations/source citation;
- exact split-leakage detection and blocking findings;
- promotion rejection for moving placeholders such as `latest`, `main`, `TBD` or unknown licenses;
- planning-only benchmark proposals kept separate from promotable manifests.

The proposal catalog covers retrieval, QA, multilingual, scientific, multi-hop, document/table/chart/multimodal and repository-owned adversarial families without inventing mutable checksums or license decisions.

### 5. Learned query planning, domain routing, entity and time normalization

Commit `b0344499c692dadb7079d07b8027fc19d8441385` added `tools/learned_query_planning.py`:

- immutable linear-softmax domain classifier artifacts and cross-entropy objective;
- confidence fallback behavior;
- plan candidates for direct/hybrid/multi-query/HyDE/step-back/multi-hop/graph/multimodal/scientific/temporal routes;
- learned linear plan ranker with latency/cost/risk penalties;
- pairwise and listwise plan-ranking losses;
- exact alias resolution that surfaces ambiguity instead of silently linking entities;
- conservative temporal normalization with explicit reference dates for relative phrases.

### 6. Semantic entailment/contradiction/citation support

Commit `08e310d5330b685f9b1ce4998e31c4f7bdc62e2c` added `evaluation/semantic_support.py`:

- provider-neutral NLI model identity and probability contracts;
- claim/evidence/citation anchors;
- entailment/neutral/contradiction evaluation;
- semantic coverage/accuracy/recall/contradiction-FNR;
- multiclass Brier score and expected calibration error;
- claim citation-support, contradiction and unsupported rates;
- contradiction-first promotion gates.

### 7. Independent-corpus fusion and reranking cascades

Commit `4a60f274d74f47efbf65e6c0ed26b1954febf60f` added `tools/corpus_fusion.py`:

- independently identified corpus/retriever candidates;
- metadata/date/language/MIME/corpus filters;
- weighted reciprocal-rank fusion;
- per-input, fused, per-document and per-source caps;
- contribution tracing;
- deterministic ordering;
- cost/latency-bounded reranking cascade planning.

Static review found representative-selection order could use an already-updated best rank; commit `4580080edd1b021fa0c56e687f5576808a5d6769` fixed it by comparing against the previous best state before updating.

### 8. Scientific document structure IR and quality gates

Commit `199f62e5fc82bcde06317f51fba715401897d17b` added `scientific/document_structure.py`:

- normalized bounding boxes and source anchors;
- title/heading/paragraph/list/table/figure/caption/formula/etc. regions;
- validated reading-order DAGs;
- table row/column spans and overlap checks;
- formula LaTeX/MathML/plain representations;
- figure panel/caption/cross-reference structure;
- generation-scoped structured-document digest.

Commit `1176043999e517a866c2cac32d2516654198df6a` added `scientific/structure_quality.py` with accept/review/block gates for reading-order coverage, confidence availability, low-confidence text, table structure/cell occupancy, formula representation, caption linkage, dangling references and extraction errors.

### 9. Repeated benchmark, shadow, ablation and historical orchestration

Commit `d63b5c8098170a8382d010d9ddabd75c7422a024` added `evaluation/experiment_orchestration.py`:

- exact query-contract binding;
- current/shadow paired runs;
- seeds and repeated runs;
- injected benchmark runners;
- measured-resource hooks;
- ablation variants;
- historical regression baselines.

Static review found that Git commit identity had incorrectly been validated as a 64-character SHA-256 data digest. Commit `5f711fb469b62839a712b713e01fed1a5fd33649` now accepts native 40- or 64-character Git object IDs while keeping data/model manifests strictly SHA-256.

### 10. Fenced periodic leadership

Commit `0f61b6983527b9369dc57ab8f07960fa5eb61c6f` added `orchestration/periodic_leadership.py`:

- periodic job specs;
- deterministic jitter;
- due-time calculation;
- leadership lease/fencing-token contracts;
- acquisition/completion records;
- injected durable lease/job stores;
- no import-time scheduler/thread.

### 11. Revision-pinned embedding profiles

Commit `6ab7bb046c0e2996eb2c358958ea03661f30ff4f` added `models/governed_embedding_profiles.py`:

- exact provider/model/revision profile identity;
- artifact/tokenizer digests;
- output dimension/pooling/normalization/token limits;
- explicit query/document instruction templates;
- reviewed license decision;
- provider output cardinality/dimension/normalization/input-order validation;
- planning-only SPECTER2, BGE-M3 and INSTRUCTOR family proposals.

No unverified mutable model revision, dimension, license or download behavior is hardcoded.

### 12. Conformal retrieval uncertainty and abstention

Commit `b09d47a1f0e8200df261411a8ae8f0b2df4312f1` added `evaluation/conformal_retrieval.py`:

- calibration identity bound to dataset split, retrieval stack, scoring contract and domain;
- probability/margin nonconformity functions;
- finite-sample split-conformal thresholds;
- calibrated support sets and forced abstention;
- selective coverage/error metrics.

Static mathematical review found that clipping a requested rank of `n+1` to the largest finite calibration score could overstate the nominal guarantee. Commit `a3a1872039873da5bf28db50dc9b52ebf781c848` now fails closed when the requested alpha is unsupported by the calibration sample size.

### 13. KMS/HSM envelope-key and rotation contracts

Commit `988c4e599b7b1623215efcee5be94924caf6ca78` added `security/key_management.py`:

- key purpose/lifecycle/version identity;
- tenant/owner/object/generation-associated encryption context;
- wrapped-data-key descriptors containing no plaintext key material;
- encrypted-artifact descriptors;
- injected KMS/HSM provider protocol;
- rewrap/rotation plans and immutable completion records;
- legal-hold check requirement;
- complete rotation coverage validation.

There is deliberately no insecure local production cryptography fallback.

### 14. Active-worker pause/resume/cancel semantics

Commit `23a09788399ff1472434e795b745ba546a8cc8aa` added `orchestration/worker_control.py`:

- durable control states;
- monotonic revisions and fencing token preservation;
- pause/resume/cancel transition graph;
- resumable safe points;
- requirement that side effects be committed before pause/cancel acknowledgement;
- compare-and-swap persistence contract;
- stale/concurrent control rejection.

### 15. Privacy-safe operator audit export

Commit `2f179e0c227e5364e917713d21214ca2d2fcf63f` added `audit/operator_export.py`:

- closed lifecycle/job/migration/reindex/retention/security event kinds;
- operation/job/document/generation/trace correlation;
- HMAC pseudonymization with explicit key identity;
- default omission of trace/generation identity according to policy;
- allowlisted public metadata only;
- no source/query/document/prompt/credential/arbitrary exception-text fields;
- deterministic export ordering and bundle digest.

### 16. Governed scientific multi-document benchmark adapter

Commit `ffa5d15ade8c19b4b33ad42188b10698b6a9c420` added `evaluation/scientific_multidoc_adapter.py`:

- scientific synthesis, contradiction, method-comparison, effect-extraction, citation-support and multi-hop task types;
- exact document/generation/content/source-group identities;
- expected evidence relations and locators;
- exact governed dataset manifest/split binding;
- strict local JSONL loader with digest verification, duplicate-key rejection, finite/closed schema and non-symlink regular-file checks;
- no dataset downloading.

## Important source capabilities already present before these additions

The live repository contains far more than the older audit snapshot. Current commit history/source includes, among other things:

- GraphRAG/evidence selection and graph governance;
- hard-negative mining;
- learned sparse and late-interaction retrieval primitives;
- runtime score calibration;
- multimodal page/region/generation retrieval and multimodal providers;
- hydrology replay/workspaces/artifact lineage/research capsules;
- HEC typed IR/scenario/datum/topology/planning infrastructure;
- PostgreSQL and backend-neutral persistent stores;
- durable leases, retries, fencing and recompute infrastructure;
- lifecycle intent/outbox/replay and reconciliation;
- retention/object-store/legal/governance work;
- parser sandboxing, malware/provider contracts, SSRF controls and rate/resource controls;
- migration/cutover compensation, fencing and blue-green execution;
- review attestations/disclosure/trust infrastructure;
- multi-region and portable research-bundle work.

Those systems were not duplicated simply because an older TODO checkbox remained unchecked.

## Old TODO items reclassified after the live audit

### Source-complete now

The following older categories are source-complete either through later existing commits or the additions above:

- reconciliation/adoption/reindex planning;
- periodic scheduling and fenced leadership contract;
- independent-corpus filters/RRF/caps/rerank-cascade planning;
- learned sparse/late-interaction/reranker training mathematics and checkpoint plans;
- domain classifier and learned plan ranking;
- entity resolution and temporal normalization;
- semantic entailment/contradiction/citation support evaluator;
- scientific PICO/PECO/effect/risk/correction semantics;
- scientific multimodal/document structure IR and quality gates;
- exact dataset-card/license/checksum/split/leakage governance;
- paired bootstrap/permutation/multiplicity/promotion statistics;
- repeated current/shadow/ablation/historical experiment orchestration;
- model-profile revision/license/digest governance;
- KMS/HSM/envelope-key and rotation contracts;
- active-worker pause/resume/cancel state semantics;
- privacy-safe job/lifecycle correlation export;
- migration/cutover source workflows;
- selective/conformal retrieval abstention;
- scientific multi-document benchmark adapter contract.

### Intentionally execution-dependent, not source-algorithm gaps

The following work remains because the requested scope explicitly excludes execution or because the exact external artifact/provider must first be chosen. It should **not** be represented as a fabricated completed experiment:

1. **Dataset acquisition and factual manifest population**
   - download/obtain the exact benchmark artifacts;
   - verify actual license terms for the intended use;
   - compute real artifact/split/query/document/source-group checksums;
   - populate exact record counts and run leakage reports.

2. **Actual model selection/artifacts**
   - choose exact revisions for SPECTER2/BGE-M3/INSTRUCTOR or alternatives;
   - verify licenses and deployment constraints;
   - obtain real artifact/tokenizer digests and measured output properties;
   - configure provider-specific runtime adapters.

3. **Training/fitting**
   - train/fine-tune sparse, dense, late-interaction, reranker, domain or plan models;
   - refresh hard negatives using real corpora/teachers;
   - fit runtime/calibration/conformal artifacts from real held-out data;
   - materialize the checkpoints whose schemas are already defined.

4. **Real current-vs-shadow/ablation execution**
   - run fixed-seed repeated experiments;
   - run the benchmark matrix across retrieval/reranking/router/model profiles;
   - measure statistical intervals/p-values/effects using produced observations;
   - evaluate practical promotion thresholds.

5. **Measured resource/performance data**
   - CPU/GPU/RAM/device-memory/storage/latency/throughput/provider-cost measurement;
   - stress/concurrency/restart/crash measurements;
   - exact environment manifests.

6. **Provider-specific live infrastructure wiring**
   - select and wire the real KMS/HSM implementation behind the source contract;
   - select the actual database/queue/object-store/cloud topology;
   - provision credentials/keys/policies/retention classes and deployment identities;
   - execute cutover/DR/restore/failover/security drills.

7. **Tests and runtime verification**
   - unit/integration/e2e/security/fault-injection tests;
   - import/compile/static tooling if counted as test execution;
   - exact-head CI and live-stack verification.

These are deliberately not marked as “done” because no observations were invented during a source-only audit.

## Additional research directions added beyond the original baseline

The continuation also expands the research surface in ways that were not merely stale TODO cleanup:

- finite-sample **conformal retrieval support sets** and abstention with explicit domain/stack calibration identity;
- **selective risk** reporting rather than answer-forcing accuracy alone;
- jointly governed **query-plan ranking under accuracy/latency/cost/risk** budgets;
- **contradiction-first semantic promotion** for evidence systems;
- immutable **human correction lineage** for scientific extraction;
- evidence-aware **cross-corpus RRF contribution tracing**;
- strict **scientific multi-document contradiction/method/effect** benchmark records;
- promotion-safe **dataset leakage governance** using source-group as well as record/query/document identities;
- explicit **model-family proposal vs promotable artifact** separation;
- KMS/HSM **wrapped-key rotation lineage** rather than treating “encryption at rest” as a configuration checkbox;
- **privacy-safe operational correlation exports** that preserve debugging utility without exporting raw research/user content.

## Recommended experiment matrix once execution is allowed

The source can now support a disciplined matrix rather than ad-hoc experiments:

- retrieval: BM25/lexical, dense, learned sparse, ColBERT/late interaction, hybrid and cross-corpus RRF;
- reranking: none, pairwise cross-encoder, listwise reranker, semantic-support cascade;
- routing: deterministic baseline vs learned domain/plan ranker;
- transforms: direct vs multi-query vs HyDE vs step-back vs multi-hop/graph/temporal;
- context: flat chunks vs parent-child/hierarchical/contextual structures;
- scientific: text-only vs document-structure-aware vs graph/multi-document evidence;
- multimodal: text-only vs page/region/table/chart/figure-aware retrieval;
- uncertainty: no abstention vs score calibration vs semantic support vs conformal support-set abstention;
- training: in-batch-only vs hard-negative curriculum vs teacher distillation;
- model profile: exact pinned general/multilingual/scientific/instruction-tuned families;
- ablations: one component disabled per governed variant plus interaction ablations for fusion/reranker/router/semantic gate;
- reporting: retrieval/citation/semantic/calibration/selective-risk metrics plus latency/memory/storage/cost and paired uncertainty/statistical tests.

## Static verification performed in this continuation

The work deliberately used static/source reasoning rather than prohibited execution. The audit included:

- checking the live branch/PR state;
- reading historical audit/TODO/capability documents;
- searching later commits before treating old TODO boxes as real gaps;
- reading existing retrieval architecture source before adding training losses;
- inspecting new modules after creation;
- identifying and correcting:
  - undefined zero-variance paired standardized effects;
  - Git commit ID vs SHA-256 data-digest conflation;
  - RRF representative selection update order;
  - unsupported finite-sample conformal alpha/rank clipping.

No test suite, model, dataset download, training loop or benchmark was executed as part of this source-only pass.

## Source-level conclusion

Within the stated scope, the remaining work is **not another missing family of core source algorithms from the reconstructed mission**. The core contracts/algorithms/methodologies for learned retrieval training/checkpointing, advanced retrieval/fusion/routing, scientific/multimodal evidence, statistics/evaluation, dataset/model governance, distributed control, security key management and operational auditing are represented in source.

The remaining meaningful steps require real external artifacts or execution: datasets, exact model revisions/weights, training, calibration, benchmarks, measured resources, provider-specific infrastructure selection and tests/deployment drills. Those should be executed later against this source rather than simulated or claimed in documentation.

This conclusion is a **source-completeness statement**, not a performance, correctness-under-execution, security-certification or production-readiness result.
