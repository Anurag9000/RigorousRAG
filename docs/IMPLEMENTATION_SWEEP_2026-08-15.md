# RigorousRAG implementation-only sweep — 2026-08-15

This ledger records the large direct-to-`main` implementation sweep performed after the
2026-08-15 mission re-audit. It deliberately excludes three categories requested by the
owner: **test/CI execution**, **downloading/curating real datasets**, and **actually
training/fine-tuning real models**. Accordingly, this document records committed software
contracts and integrations, not empirical model-quality or release-readiness claims.

The older August 1/2 TODO and capability documents remain useful historical records, but
many of their unchecked items were subsequently implemented. This file is the most recent
implementation-focused reconciliation for the categories covered below.

## 1. Retrieval, ranking, planning and model-lifecycle software

Implemented in the current sweep or already present on `main`:

- dense, sparse/BM25, hybrid, adaptive, corrective, heterogeneous and multi-hop retrieval;
- SPLADE/neural-sparse and late-interaction primitives and concrete injected/HF backends;
- versioned capability registry with dependencies, trust, resource envelopes and fallback;
- model-neutral training dataset/config/artifact contracts for rerankers, dense retrievers,
  sparse retrievers, late-interaction retrievers, routers, planners and entailment models;
- grouped leakage-resistant data splitting and hard-negative attachment contracts;
- objective definitions for pointwise, pairwise, listwise, contrastive, in-batch,
  distillation and Matryoshka-style training;
- candidate-plan ranking, bounded beams, failure-aware repair and dynamic budget
  redistribution;
- learned adaptive-policy provider interface with deterministic fail-closed fallback;
- dynamic runtime budget accounting for wall time, calls, tokens and provider cost;
- query transformation contracts including rewriting, expansion, HyDE-like variants,
  synonyms and citation chasing;
- programmable fusion, reranker cascades, metadata filter ASTs and diversity caps;
- cross-profile/multi-route orchestration with identity reconciliation and route lineage;
- hierarchical/contextual parent/section/neighbor scoring without changing citation identity;
- multilingual/Indic Unicode/script routing with transliteration/translation lineage;
- candidate -> shadow -> canary -> production -> retired/rejected lifecycle with immutable
  artifact/evaluation identities and rollback targets;
- backend-neutral cross-dimension migration coordination with preparation, publish,
  validation, fencing and compensating rollback.

Implementation still requiring provider/deployment work, not missing architecture:

- concrete trainer implementations for every supported external ML framework;
- operator-supplied trained weights and calibrated policies;
- specialized production index adapters where an external vector/database product is used;
- migration-participant adapters for each deployed external store.

## 2. Document-native and multimodal scientific RAG

Implemented:

- normalized scientific document IR covering pages, blocks, reading order, tables/cells,
  figures/panels, formulas, captions and cross-modal links;
- deterministic page-text fallback builder with explicit heuristic confidence;
- deterministic reading-order, caption, figure/table/equation/reference linking;
- immutable page-image late-interaction artifacts and MaxSim retrieval;
- hierarchical page -> region selection;
- injected transformer/ColPali/ColQwen-style page backend that never downloads weights;
- coordinate-preserving multimodal evidence regions and existing HF multimodal support;
- authoritative document-IR -> evidence-graph construction with generation fingerprints;
- bounded graph path retrieval/reranking and support/contradiction clustering.

Still implementation-worthy:

- provider-specific layout/table/formula/chart extraction adapters beyond the deterministic
  fallback and existing generic multimodal interfaces;
- specialized page-vector compression/ANN stores for very large page-native corpora;
- richer chart-series reconstruction and mathematical-symbol canonicalization;
- browser visualization for document blocks, figure/table regions and graph paths.

## 3. Claim rigor, scientific synthesis and causality

Implemented:

- atomic claim segmentation and fail-closed claim/evidence entailment aggregation;
- server-citation-only support and contradiction authority;
- optional live `SearchAgent` entailment post-gate that removes unsupported claims;
- injected transformer NLI adapter with explicit reviewed label mapping and no downloads;
- PICO/PECO/PICOS research-question and study-evidence structures;
- effect normalization for common ratios, differences, standardized differences,
  correlations and proportions;
- fixed/random-effects synthesis, heterogeneity and leave-one-out sensitivity primitives;
- structured risk-of-bias fields and transparent GRADE-inspired certainty bookkeeping;
- causal variables, assumptions, DAGs and causal-claim readiness kept separate from generic
  association/evidence graphs;
- transparent source trust/applicability policy that is explicitly **not** a truth score;
- source retraction/withdrawal/supersession propagation into graph/report warning state.

Still implementation-worthy:

- domain-specific extraction adapters mapping source text into every PICO/effect/risk field;
- more statistical estimands and robust/meta-regression/network-meta-analysis methods where
  scientifically appropriate;
- reviewed causal-identification helpers (adjustment sets, transportability and sensitivity)
  without converting structural assumptions into automatic truth;
- interactive human review UI for certainty, risk-of-bias and causal assumptions.

## 4. Hydrology and geospatial scientific vertical

Implemented:

- CRS-explicit geospatial evidence contracts;
- raster-window and hydrological time-series provenance;
- HEC-HMS and HEC-RAS scenario abstractions;
- CHIRPS adapter contract and local no-download CHIRPS manifest adapter;
- safe local HEC project metadata inspection;
- strict HEC-HMS/HEC-RAS exported time-series/profile adapters without executing HEC;
- hydrograph peak/scenario comparison and discharge-volume integration;
- unit/dimension-safe numerical reasoning with uncertainty-preserving arithmetic;
- generic hydrology domain adapter for query routing, units, graph enrichment and reports;
- bounded CRS-explicit spatiotemporal evidence index.

Still implementation-worthy:

- optional reviewed GeoTIFF/NetCDF/PostGIS adapters using operator-installed libraries;
- geometry/reach/network parsing for richer HEC-RAS spatial provenance;
- basin/reach topology and upstream/downstream spatial reasoning;
- raster/page/map region visualization in the browser;
- additional rainfall/runoff/hydraulic formats and domain adapters requested by operators.

## 5. Production, security, durability and operations

Implemented or consolidated:

- versioned object-store contract plus injected S3-compatible adapter;
- durable lease-queue contract plus injected Redis-compatible adapter;
- PostgreSQL DB-API control plane for artifacts, audit, lifecycle and research workspace;
- durable SQLite research workspace fallback;
- distributed-style admission/token-bucket controls;
- secret-provider abstraction plus environment and injected external-secret adapters;
- egress allowlists with public-IP validation;
- parser-sandbox and malware-scanner contracts plus concrete fixed-command subprocess
  adapters using `shell=False`, private temp directories, minimal environments, deadlines
  and byte ceilings;
- typed, versioned runtime configuration with secret references and allowlisted overlays;
- unified owner-scoped artifact lineage and privacy-safe audit-event primitives;
- cross-store reconciliation, retained-source reindex, conservative legacy adoption and
  exact-confirmation repair plans;
- dependency-aware exact-confirmation retention execution;
- research capsules containing content-addressed replay references while excluding secrets
  and raw private evidence;
- provenance lineage/downstream-impact query engine.

Important non-claims:

- a subprocess boundary is not equivalent to a kernel/container sandbox;
- injected cloud/broker adapters do not provision infrastructure or credentials;
- compensating migration is not distributed consensus;
- content-addressed lineage proves identity/derivation, not scientific truth.

Still implementation-worthy:

- Kubernetes/container/OS-specific hard sandbox adapters (seccomp/AppArmor/job objects);
- production cloud KMS/key rotation and provider-native secure deletion adapters;
- multi-region object/metadata replication policies;
- PostGIS/vector-store specific transaction participants and queue dead-letter dashboards;
- operator UI for repair, retention, cutover and lifecycle state.

## 6. Research product and agent reachability

Implemented:

- one explicit governed agent-tool registry for closed schemas, owner injection,
  permissions, runtime budgets and citation policy;
- live registry bridge installed through the existing import-order-safe research-agent
  hook alongside graph, multi-hop and optional entailment integration;
- persistent owner-scoped research projects, corpus bindings, sessions and turns;
- production `/research/*` APIs for project/session lifecycle and capability discovery;
- browser Workspace tab for creating/listing projects and sessions with safe DOM APIs;
- structured research reports and evidence matrices with JSON/CSV/Markdown-friendly data;
- bounded backward/forward citation-chasing contracts;
- generic scientific domain registry and hydrology domain adapter.

Still implementation-worthy:

- migrate the historical graph/multi-hop wrappers fully into registry-native tool specs and
  eventually retire their compatibility monkey-patches/import-hook state;
- connect finalized research-agent answers to workspace turns automatically rather than
  requiring an explicit turn-write call;
- expose report/evidence-matrix creation from *server-authoritative finalized result IDs*
  (not client-submitted citation objects);
- graph explorer, page/region citation viewer, contradiction/certainty panels and experiment
  comparison UI;
- operator-facing capability/model/policy lifecycle screens.

## 7. Next-order additions identified and implemented in this sweep

Beyond the original backlog, the sweep added:

- reproducible research capsules;
- provenance query and downstream-impact analysis;
- explicit causal-vs-associational evidence structures;
- transparent source trust/applicability policy separated from truth;
- spatiotemporal indexing for scientific/geospatial evidence;
- source-status/retraction propagation;
- domain-extension registry rather than central-agent domain conditionals.

Further next-order opportunities are listed below so they are not confused with missing
core RAG functionality:

1. provenance-aware answer diffing between corpus/model/policy generations;
2. reproducible "research capsule" import/replay orchestration across machines;
3. cryptographic signing/attestation of model, index, report and capsule manifests;
4. privacy-preserving collaboration/ACLs for shared projects and reviewed claims;
5. declarative experiment specifications that compile to retrieval/training/evaluation runs;
6. semantic schema migration/version negotiation across document IR, graphs and reports;
7. causal sensitivity/negative-control modules with explicit expert review;
8. geospatial topology and network-flow query primitives;
9. formal numerical consistency checks between answer prose, evidence tables and cited
   source quantities;
10. evidence-aware answer versioning that marks which claims changed when a source is
    corrected, superseded or retracted.

## 8. Requested exclusions and current claim boundary

No new test/CI execution is claimed by this ledger. No datasets were downloaded or
curated during this sweep. No real model was trained or fine-tuned. Injected model/provider
adapters require operator-supplied, already-approved resources.

Therefore the correct status is: **the implementation architecture is now substantially
broader and more integrated than the historical TODOs, but external-provider deployment,
real trained artifacts/data, remaining UI depth and complete compatibility-wrapper
retirement are still open.**
