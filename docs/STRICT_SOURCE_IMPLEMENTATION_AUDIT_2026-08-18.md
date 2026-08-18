# RigorousRAG strict source-implementation audit — 2026-08-18

## Status and scope

This document is the strict source-level continuation audit of the live `main` branch after the 2026-08-17 audit and the subsequent adversarial implementation sweeps. It **supersedes older unchecked TODO/capability boxes for source-implementation status only** when later code/history demonstrably implements the capability.

This is **not** a release-readiness, benchmark, model-quality, deployment, or test-pass declaration. During this continuation, no dependency installation, model/tokenizer download, dataset download, model training, hard-negative mining run, model inference benchmark, full test suite, CI matrix, fault injection campaign, production deployment, cloud failover, human-review campaign, DLP/IdP/KMS/HSM call, cryptographic attestation verification, or real SLO collection was executed. Source tests were authored to pin contracts, but they were not run as part of this source-only sweep.

The source-completeness boundary used here is:

1. executable/reference source exists for the requested architecture, algorithms, losses, training/checkpoint paths, adapters and orchestration;
2. durable state/identity/fencing/rollback/reconciliation exists where a production authority boundary requires it;
3. security/governance boundaries have explicit fail-closed contracts rather than being left as prose;
4. optional external systems are injected behind typed interfaces and are not falsely claimed to exist because an interface exists;
5. model/dataset/artifact facts that require real acquisition/verification are represented as governed manifests/proposals rather than fabricated values;
6. no source family is counted complete merely because an old TODO says so, and no old unchecked box is counted missing until the current tree/history is checked.

The exact `main` head immediately before this audit document was created was `1de26f2bf0dbe6e832962e028a3063e98c47f560` (`test: cover admitted local model artifact binding`). The audit-document commit itself advances `main` once more.

## Audit method

The sweep used the current tree, recent commit history, the dated source/capability ledgers, and targeted adversarial searches. Old TODO entries were treated as hypotheses. The final static probes found no current code-searchable `NotImplementedError` or TODO/FIXME/future-work marker that exposed a new material source family. Protocol/interface boundaries were then judged against concrete adapters, durable authority layers and explicit external-runtime boundaries rather than assuming that every injected provider must have an in-repository production service implementation.

The repository was audited in eight mission families:

1. ingestion/lifecycle/security;
2. retrieval/query planning/training;
3. evidence/citation/semantic authority;
4. scientific/multimodal intelligence;
5. evaluation/experimentation/observability;
6. adaptation/human review;
7. orchestration/recovery/DR;
8. deployment/governance/supply chain.

## Final source-level capability matrix

### 1. Ingestion, lifecycle and storage authority

Source exists for the owner/tenant-scoped multi-store lifecycle and its recovery paths, including:

- retained-source ingestion and privacy-finalized document handling;
- immutable generation history/current pointers;
- dense/vector, lexical/sparse, metadata/registry and derived-graph lifecycle coordination;
- deterministic operation identities, durable outboxes/intents and replay;
- leases, fencing, retries, dead letters, adoption and reindex flows;
- reconciliation, retention/compaction and privacy-safe operator correlation/export;
- same-dimension and dimension-changing migration/cutover orchestration;
- rollback capture/restoration, compensation and source-generation revalidation;
- object-storage contracts and SQL/durable persistence paths;
- worker pause/resume/cancel safe points;
- periodic leadership/reconciliation primitives.

Older TODO boxes that still describe these as absent are stale for source status.

### 2. Retrieval, ranking, query planning and learned training

Source now covers the requested retrieval/modeling matrix:

- BM25/lexical, dense, hybrid and independent-corpus retrieval;
- weighted RRF with filters, corpus/retriever weights, source/document caps and reranking cascades;
- SPLADE/uniCOIL-style sparse scoring;
- ColBERT-style late interaction/MaxSim;
- dense bi-encoder, cross-encoder and listwise reranker architectures;
- multilingual/scientific governed profile contracts and verified local Hugging Face adapters;
- multi-query, HyDE, step-back, decomposition, multi-hop, graph, multimodal, scientific and temporal plan families;
- learned domain classification and query-plan ranking;
- exact alias/entity and relative/ISO temporal normalization;
- query/cost/latency/risk-aware routing and stopping/abstention;
- hard-negative mining and refresh hooks;
- pointwise/pairwise/listwise retrieval/reranking losses, sparse regularization and teacher distillation;
- resumable training engines/checkpoints with optimizer/scheduler/scaler/RNG/data-cursor identity;
- pointwise and query-grouped listwise cross-profile fusion-weight learning.

The pointwise cross-profile optimizer was corrected so empirical gradients are normalized **before** L2 is added; regularization no longer changes strength when example/sample weights change.

### 3. Cross-profile score calibration and heterogeneous fusion lifecycle

The earlier audit incorrectly treated rank fusion alone as sufficient for heterogeneous score spaces. This adversarial sweep closed the full lifecycle:

- rank-only RRF remains the safe fallback for incomparable score spaces;
- score-to-relevance calibration artifacts are bound to profile/model/scoring/dataset/split/candidate-universe contracts;
- calibrator held-out qualification gates sample support, class support and calibration quality;
- calibrated probability fusion never averages raw BM25/dense/SPLADE/ColBERT scores;
- learned pointwise and ListNet-style query-grouped fusion weights are reproducible/resumable;
- held-out promotion compares learned policies against uniform baselines with practical-quality and anti-collapse gates;
- canonical persistence covers calibrators, qualifications, training state, learned weights and promotion receipts;
- calibration-drift/currentness checks use qualification age, score-distribution shift and labeled Brier/ECE when available;
- the highest-authority runtime path requires both offline promotion and current per-profile drift evidence;
- fusion metrics feed the shared privacy-safe quality dashboard.

### 4. Evidence, citation and semantic authority

Source covers:

- immutable document/generation/page/chunk/block/region anchors;
- evidence/citation conversion with server-owned identity;
- claim-to-evidence support and contradiction semantics;
- text NLI provider contracts and local/adaptable model identity;
- coverage, support, contradiction, Brier and ECE metrics;
- contradiction-first promotion gates;
- conformal retrieval support sets and abstention;
- counterfactual retrieval change audits;
- server-owned citation-set validation;
- closed-schema model-output authority that rejects duplicate JSON keys, non-standard numbers, extra fields and reserved role/tool/function authority fields;
- grounded model outputs that may cite only the server-owned allowed evidence set.

### 5. Scientific, document-structure and multimodal intelligence

Source now covers the scientific/document mission beyond generic OCR:

- PICO/PECO/PICOS/free-form research questions;
- population/method/intervention/exposure/comparator/outcome/result/limitation extraction;
- effect estimates, uncertainty, sample/event metadata and synthesis transforms;
- risk-of-bias/certainty fields and immutable correction/review lineage;
- normalized page geometry, reading-order DAGs, table topology/merged cells, formulas, figures, panels and captions;
- verified-local layout/table/OCR/formula adapters;
- structured scientific extraction receipts and review queues;
- structure-quality accept/review/block gates;
- multimodal visual support semantics and verified local multimodal entailment adapter;
- provenance-aware chart IR with typed axes/series/points/uncertainty/units;
- verified-local chart-to-structure adapter with closed JSON/tabular decoding contracts;
- conservative typed table-cell numeric extraction;
- table/chart native entailment with interval-aware comparison/equality/aggregate/trend semantics;
- exact table/chart authority gates for lineage, extraction confidence, units and uncertainty width;
- review-required structured evidence can feed the governed active-learning loop instead of silently becoming authoritative;
- structured table/chart support feeds the existing semantic calibration/quality machinery.

### 6. Evidence graph, cross-document and hydrology lineage

Current source includes:

- typed evidence graph generations and provenance-preserving graph identities;
- graph retrieval/path explanations, scientific temporal/retraction semantics and GraphRAG paths;
- cross-document evidence graph sets without collapsing owner/document/generation lineage;
- evidence-graph reconciliation/integrity/retention and restore-custody infrastructure;
- deterministic evidence-graph snapshot/export/verify/staged restore/cutover/rollback receipts implemented during the strict source sweep;
- hydrology/geospatial lineage/replay and backend-neutral recompute contracts;
- scientific/cross-document evidence relationships and review lineage.

No source claim is made that a production graph database or external hydrology engine was actually run during this audit.

### 7. Evaluation, reproducibility and experimentation

Source covers:

- exact dataset/version/artifact/license/loader/transformation/split manifests;
- leakage checks and governed benchmark proposals without fabricated checksums/licenses;
- BEIR/local benchmark adapters where appropriate;
- retrieval/citation/generation/semantic/calibration metrics;
- current-vs-shadow repeated runs and fixed query/seed contracts;
- ablations and historical regression baselines;
- paired bootstrap and paired permutation/randomization tests;
- Cohen paired effects and Holm/BH multiplicity controls;
- measured-resource hooks and cost/latency/storage observations;
- promotion gates that distinguish statistical and practical change;
- a unified privacy-safe quality dashboard/SLO comparison layer;
- randomized team-draft interleaving for online retrieval-policy comparison;
- held-out interleaving promotion gates using sample support, confidence/sign-test evidence and tie/decisiveness limits;
- sealed interleaving evidence receipts;
- owner-scoped durable interleaving journals;
- sticky eligibility/traffic assignment and mutual-exclusion groups to prevent cross-experiment contamination;
- aggregate interleaving/fusion/active-learning/control-plane observations for the shared dashboard.

### 8. Adaptation, active learning and expert review

The previous tree had calibration, abstention and durable adjudication, but no governed selection loop. This continuation added:

- digest-only active-learning candidates with uncertainty, model disagreement, abstention, drift, novelty, expected impact and labeling cost;
- deterministic acquisition scoring under total-cost, item-count, per-task and per-diversity-group caps;
- idempotent materialization into the existing owner-scoped expert-adjudication store;
- durable batch/case mapping and crash-safe replay;
- immutable resolved-gold manifests bound to case/resolution/round/revision identity;
- explicit label mappings before converting adjudicated gold into binary calibration/training examples;
- standardized semantic-ensemble entropy/JS-disagreement, structured-abstention and calibration-drift acquisition adapters;
- structured table/chart review decisions routed into active learning;
- privacy-safe active-learning observability;
- resumable paged one-shot acquisition cycles suitable for the existing fenced periodic executor.

Human labels were **not actually collected** in this source audit.

### 9. Backup/restore, DR and multi-region authority

Source covers both data restoration and traffic/write authority:

- manifest-bound backup/snapshot/restore and evidence-graph staged restore/cutover;
- restore verification and rollback identities;
- DR rehearsal/failover policy primitives;
- region health/readiness/replication-lag/RPO evidence;
- one write-authoritative region per owner/service with CAS revisions and monotonic fencing;
- explicit failover/failback and append-only authority decision history;
- stale/future health evidence rejection;
- data-plane write-authority assertion so stale regions cannot continue writes after takeover.

This continuation additionally added **data residency as a hard failover constraint**:

- region/provider/country/jurisdiction descriptors;
- data-class-specific residency rules for source content, derived indexes, metadata, audit, backups, key material and model inputs;
- CAS-promoted owner/service residency-policy history;
- residency-aware failover that treats healthy-but-forbidden regions as unavailable;
- a residency-ineligible current region is evacuated to a healthy eligible region when possible;
- all-ineligible regions yield a deterministic hold rather than unsafe failover.

No real cloud/DNS/database failover was executed.

### 10. Runtime model/retrieval-stack authority

A source gap found late in the adversarial sweep was that already-built compatible stacks lacked one serving authority. It is now closed with:

- immutable stack artifacts covering dense/sparse/late-interaction/reranker/generator/semantic/router/plan/calibrator/fusion/tokenizer/document-model components;
- exact component artifact and contract digests;
- typed promotion evidence (offline quality, semantic/citation quality, calibration qualification/currentness, interleaving, resources, security/license/compatibility/dataset/operator review);
- required evidence classes and compatibility checks;
- bounded promotion-decision TTLs so an old eligible decision cannot be replayed indefinitely;
- immutable stack registry;
- owner/service/domain-scoped authority revisions and fencing tokens;
- promotion CAS;
- serving-time exact stack+fence assertion;
- monotonic rollback that creates a new revision/fence rather than rewinding history;
- rollback refusal when the historical stack is incompatible with the current serving environment.

### 11. Retrieved-content trust and indirect prompt-injection boundaries

Security is architectural rather than based on a claim that prompt-injection text can be perfectly classified:

- retrieved evidence has immutable content/provenance/generation identity and an explicit trust class;
- built-in narrow instruction/tool/secret-exfiltration signals are advisory risk/review signals;
- built-in inspection cannot be suppressed; external detectors can only add findings;
- the generation renderer re-binds packed evidence to the original generation-aware candidate and verifies the materialized content/trust decision;
- review/quarantine evidence cannot enter the generator context;
- generation input has fixed system/user roles and evidence is JSON-escaped quoted data, so evidence cannot manufacture message/tool fields;
- a system instruction explicitly states that retrieved evidence is data, not command authority;
- evidence action/tool suggestions are provenance only;
- executable tool authorization requires an independent trusted-planner decision, current allowlisted tool/schema contract and exact runtime argument digest;
- serving re-checks the tool policy and arguments before execution authority is accepted.

### 12. Data-loss prevention and model-input release

A separate governed release boundary now exists before text is sent to a model/provider/export destination:

- native minimum scanning for email, IP, secret assignments and Luhn-valid payment-card candidates;
- native findings cannot be suppressed;
- external DLP/NER scanners can provide content-bound attestations;
- policies may require named detector attestations;
- destination-specific category rules can allow, redact or block;
- release receipts contain content/finding/scan/policy/output digests rather than raw sensitive substrings;
- generation system text, user query and each evidence block pass through the model-input release policy;
- if evidence is redacted, the original evidence/provenance SHA remains citation authority while a separate released-content SHA identifies the exact text seen by the model.

This is a source safety net, not a claim that the native detector finds all PII. A production enterprise DLP/NER provider can be required by policy and was not run here.

### 13. Operator authorization and high-risk control mutations

The repo now has a repository-owned authorization layer rather than assuming authentication alone:

- external IdP verification is represented by a bounded `VerifiedPrincipalAssertion`;
- only a digest of the principal is used in authorization state;
- principal-to-role bindings are owner/domain scoped;
- role permissions bind actions and resource classes;
- high-risk permissions can require MFA and an operator-reason digest;
- authorization receipts are short-lived and bound to the exact action/resource request and current policy digest;
- operator-authorization policies have immutable CAS-promoted history;
- authorized control-plane entrypoints exist for runtime promotion/rollback, residency-policy promotion and region failover/failback.

No real IdP login/authentication was performed.

### 14. Supply-chain attestation and local model admission

The previous tree verified local artifact bytes but did not prove who built/admitted them. That seam is now closed source-wise:

- exact artifact subjects and SHA-256 identities;
- attestation statements bind artifact, source revision, build config, dependency lock, SBOM and typed predicates;
- signature verification is injected via a trusted verifier protocol—there is no home-grown signing algorithm;
- verified attestations bind statement/subject/key/verifier/version/evidence/time identity;
- admission policy requires trusted builders, signing keys, verifiers, predicates and freshness;
- admission verifies expected artifact digest, source revision and dependency-lock digest;
- local model/tokenizer tree bindings can be paired with admitted artifact proofs;
- admitted local bindings re-hash the actual local file trees before use;
- first-class admitted factories construct the existing dense, SPLADE, ColBERT and cross-encoder local providers only after re-verification;
- sparse/late/reranker runtime artifact digests must equal the admitted model-tree digest;
- adapter construction remains lazy and performs no network/model load merely because the source is imported.

Actual signature verification, SBOM generation or vulnerability scanning was not run in this audit.

## Newly closed adversarial residuals after the 2026-08-17 audit

The most important lesson from this continuation is that source completeness required checking the seams **between** already-strong components. The following gaps were found only after the earlier ledger looked nearly complete:

1. heterogeneous raw score spaces lacked governed cross-profile calibration/fusion;
2. calibration artifacts lacked held-out qualification and staleness/currentness enforcement;
3. learned cross-profile fusion weights lacked pointwise/listwise training, promotion and runtime lineage;
4. retrieval-policy comparison lacked randomized interleaving, traffic eligibility and contamination control;
5. uncertainty/abstention/expert review lacked an acquisition/materialization/gold-feedback active-learning loop;
6. generic multimodal vision support did not equal chart/table-native structured entailment;
7. chart/table structured evidence lacked claim-specific authority/review gates;
8. periodic/multi-region recovery lacked one fenced region write-authority controller;
9. region failover lacked data-residency constraints;
10. retrieved evidence lacked a generation-role and tool-authority boundary against indirect prompt injection;
11. compatible runtime model/retrieval stacks lacked one monotonic promotion/rollback serving authority;
12. high-risk control mutations lacked repository-owned scoped operator authorization;
13. model-input/export boundaries lacked a destination-aware DLP release gate;
14. model outputs lacked closed-schema publication/citation authority;
15. digest-correct local artifacts lacked supply-chain attestation/admission binding;
16. supply-chain admission initially remained separate from local adapter construction, so admitted provider factories were added;
17. pointwise learned-fusion L2 normalization, multi-region failback/freshness and residency-store idempotency each had static hardening issues corrected in place.

## Strict residual conclusion

After the final current-tree/history sweep and targeted static residual searches, **no additional material source-level capability gap was found within the bounded RigorousRAG mission described above**.

This statement means:

- the requested source architectures, algorithms, losses, checkpoint/resume paths, persistence, adapters, authority state machines, security/governance contracts and recovery workflows are represented in source;
- optional production systems that inherently require external infrastructure are explicitly injected rather than fabricated;
- older TODOs that describe later-implemented source as missing should not be used as the source-status authority.

It does **not** mean:

- the code is proven bug-free;
- all authored tests pass;
- dependencies install cleanly;
- every optional external integration is configured;
- model/dataset/license/signature facts have been collected;
- real quality, latency, cost, reliability, security or recovery targets have been met;
- the repository is release-ready.

Any future source work should be triggered by a newly demonstrated concrete gap, a static defect, an integration failure, new mission scope, or execution evidence—not by stale unchecked boxes alone.

## Work intentionally still remaining outside this source-only scope

The following execution/artifact work remains real and should not be blurred into source completeness:

### Dependencies and exact-head verification

- install dependencies in clean supported environments;
- run `pip check`/lock verification;
- compile/static-check/lint the final unchanged head;
- run the full pytest/coverage matrix on supported Python versions and Windows where required;
- validate Compose/container builds and clean-clone CLI/API/browser workflows;
- run final unchanged-head CI and branch/protection verification.

### Real model/tokenizer artifacts

- select exact model/tokenizer revisions;
- obtain local artifacts;
- calculate/record real tree and file digests;
- conduct actual license/model-card/security review;
- produce/verify real SBOM/build-provenance/signature attestations;
- configure the real trusted attestation verifier/key roots;
- run inference compatibility/dimensionality checks.

### Real datasets

- acquire benchmark/training corpora;
- calculate actual artifact/split checksums;
- review real licenses/terms/PII/safety limitations;
- materialize governed split manifests and leakage evidence.

### Actual training and adaptation

- build real training examples;
- mine real hard negatives;
- train dense/SPLADE/uniCOIL/ColBERT/reranker/domain/query-plan/fusion models;
- generate real checkpoints and resume them;
- fit/qualify/calibrate real probability mappings;
- collect human expert labels through the active-learning/adjudication loop.

### Actual retrieval/inference/evaluation

- build real lexical/vector/sparse/late-interaction indexes;
- run real current/shadow retrieval and generation;
- execute repeated benchmark seeds/ablations;
- measure true retrieval/citation/semantic/conformal metrics;
- measure real latency, throughput, CPU/GPU/RAM/storage/provider cost;
- run online interleaving on approved traffic and evaluate engagement outcomes;
- establish real promotion evidence rather than fixture/synthetic source contracts.

### Production security/governance integrations

- connect and verify the real identity provider/MFA assertions;
- configure an enterprise DLP/NER provider where policy requires it;
- configure real KMS/HSM/secret-manager providers and rotation permissions;
- configure production object/database credentials, TLS, network egress and storage policies;
- perform real malware/parser sandbox exercises;
- create/verify real artifact signatures/SBOM/vulnerability results;
- verify legal-hold/retention policies against deployment storage.

### Reliability, DR and deployment

- provision the actual distributed services/databases/object stores/vector indexes/graph stores;
- execute crash/concurrency/fencing/fault-injection campaigns;
- execute evidence-graph and full-platform snapshot/restore drills;
- execute real multi-region failover/failback under residency policy;
- validate RPO/RTO and split-brain prevention with real data-plane adapters;
- collect real SLO/error-budget telemetry;
- conduct load/soak/canary/rollback drills;
- produce one final unchanged-head release proof.

## Source-only bottom line

At the end of this 2026-08-18 adversarial implementation sweep, the remaining work is predominantly **artifact acquisition, training, execution, integration verification and deployment evidence**, not another known missing source architecture family. The next engineering phase should therefore move to exact-head execution and real artifacts when that scope is allowed, while fixing any concrete defects those runs reveal.
