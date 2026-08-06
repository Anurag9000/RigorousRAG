# RigorousRAG capability expansion roadmap

Last synchronized: 2026-08-06

The roadmap converts the accepted exhaustive capability program into ordered, independently verifiable waves. Every wave must include implementation, adversarial tests, benchmark coverage, configuration, operator documentation and ledger updates. Detailed status, historical PR reconciliation and the ordered remaining backlog are recorded in `docs/RIGOROUSRAG_EXHAUSTIVE_AUDIT_2026-08-06.md`.

## Wave 1 — retrieval and evaluation foundation

Status: implemented foundation on `main`; exact-head release verification remains required.

Deliverables: bounded hybrid ranking, lexical BM25, fusion, MMR, rerankers, normalized evaluation data, retrieval/citation metrics, deterministic manifests, resumable experiments and an offline baseline runner.

## Wave 2 — persistent hybrid retrieval and index governance

Status: substantial foundation implemented. Authoritative vector/sparse/generation lifecycle, migration shadow construction, benchmark promotion and non-mutating cutover preflight exist. Governed cutover/rollback execution, lifecycle retention and distributed reconciliation remain open.

1. Embedding profile registry: MiniLM compatibility profile, E5, BGE, GTE, Instructor, SPECTER2 and BGE-M3 capability metadata; query/document instructions; dimensions; normalization; language/domain; licensing notes; fingerprints and schema versions.
2. Persistent sparse index: owner/document/field keys, fielded BM25, positional provenance, generation snapshots, transactional replace/delete and corruption refusal.
3. Cross-store coordinator: vector and sparse snapshots, compensation on either-store failure, durable generation manifests and repair/reconciliation tooling.
4. Corpus-level hybrid retrieval: dense and sparse candidate generation, fusion, reranking, MMR, filters and complete score/provenance traces.
5. Migration/reindex tooling: dry-run plans, resumable manifests, profile compatibility checks, shadow indexes, validation, benchmark promotion, cutover, rollback and retention.

## Wave 3 — adaptive and corrective RAG

Status: deterministic routing, evidence sufficiency, corrective retrieval, privacy-safe traces, calibration primitives and offline route evaluation are implemented. Representative connected experiments, runtime calibrator promotion and trace retention remain open.

- Query classification, decomposition and routing.
- HyDE, multi-query, step-back and sub-question planning policies.
- Retrieval confidence and evidence-sufficiency estimators.
- Corrective retrieval loops with bounded budgets.
- Self-RAG-style retrieve/critique/revise contracts without hidden provenance mutation.
- Contextual compression, parent-child retrieval, sentence-window retrieval and metadata-aware filters.
- Abstention and explicit evidence-gap reporting.

## Wave 4 — multi-hop evidence graphs

Status: typed generation-bound evidence graphs, durable reconciliation jobs, authority-aware reads, cross-document operations, graph-job audit, conservative retention planning and exact-confirmation graph-payload compaction are implemented foundations. Production extraction/linking, temporal clustering and benchmarked graph-policy promotion remain open.

- Typed evidence, entity, claim, method, dataset, result and citation nodes.
- Citation/reference graph construction and temporal provenance.
- Graph-assisted entity linking and relation extraction.
- Multi-hop path search with bounded branching and loop detection.
- GraphRAG community summaries as derived, versioned artifacts.
- Contradiction, agreement and evidence-chain visualization contracts.
- Durable reconciliation, privacy-safe audit, retention planning and resumable compaction.

## Wave 5 — multimodal and structure-aware ingestion

Status: bounded native/OCR parsing and visual-evidence safety foundations exist; full structure-aware and cross-modal capability remains open.

- Layout-preserving PDF reading order.
- Table structure and cell-coordinate extraction.
- Formula/LaTeX detection and equation references.
- Figure, caption, panel and in-text mention linking.
- OCR coordinates and confidence propagation.
- Visual embedding and late-interaction retrieval profiles.
- Modality-aware chunking and cross-modal citations.
- Parser sandbox and malware scanning remain external deployment controls.

## Wave 6 — scientific evidence intelligence

Status: claim extraction, reviewer correction/custody, evidence-graph integration, extractor registry/evaluation/promotion and rollback foundations are implemented. Study-design depth, calibrated entailment, numerical extraction and domain benchmark promotion remain open.

- PICO/PECO and study-design extraction.
- Methods, interventions, outcomes, datasets and limitations schemas.
- Claim-evidence-entailment records with explicit uncertainty.
- Risk-of-bias and evidence-quality checklists.
- Contradiction clusters and replication links.
- Numerical result extraction, units, confidence intervals and effect sizes.
- Protocol-to-result traceability and citation-context analysis.
- No automatic scientific truth claims.

## Wave 7 — model, architecture and provider expansion

Status: embedding-profile and encoder-adapter governance exists; production adapters, neural sparse retrieval, late interaction and governed reranker cascades remain open.

- Dense bi-encoder profiles and adapter-aware scientific encoders.
- Sparse neural retrieval profiles such as SPLADE-compatible interfaces.
- ColBERT/late-interaction interfaces.
- Cross-encoder, listwise and LLM reranker interfaces.
- Local/OpenAI-compatible provider registry, capabilities and health checks.
- Quantization, batching, model cache, circuit breakers and resource budgets.
- Ensemble and cascade policies selected by benchmark evidence rather than default complexity.

## Wave 8 — datasets and benchmark suites

Status: normalized dataset/metric/experiment contracts and several repository-owned fixtures exist. Licensed external adapters, cards, checksums, leakage checks and structure-aware suites remain open.

- BEIR adapters plus domain subsets.
- SciFact, NFCorpus, TREC-COVID, FiQA and ArguAna.
- MS MARCO and Natural Questions adapters where licensing/storage permits.
- HotpotQA, 2WikiMultiHopQA and MuSiQue for multi-hop evaluation.
- PubMedQA, BioASQ and evidence-inference datasets for scientific reasoning.
- Table/figure/formula retrieval benchmarks.
- Synthetic corruption, prompt-injection, tenant-isolation and provenance adversarial suites.
- Dataset cards, licenses, checksums, split manifests and leakage checks.

## Wave 9 — experiment automation and observability

Status: deterministic matrices, immutable results, paired promotion evidence and privacy-safe traces exist. Connected resource measurement, benchmark history, dashboards and production telemetry remain open.

- Full retrieval/reranking/chunking/model ablation matrices.
- Latency, throughput, memory, cost and energy measurements.
- Statistical confidence intervals, paired tests and repeat seeds.
- Regression thresholds and benchmark history.
- Failure taxonomy and per-query diagnostics.
- OpenTelemetry/Prometheus integration, SLO dashboards and privacy-conscious traces.
- Reproducible reports and machine-readable model/index cards.

## Wave 10 — distributed production architecture

Status: not implemented as a distributed system. Current process-local/SQLite foundations must not be described as distributed exactly-once infrastructure.

- Shared SQL registry/jobs and distributed queue/leases.
- Transactional outbox/saga across stores.
- Versioned encrypted object storage.
- Distributed rate limits and idempotency keys.
- Dedicated model serving, GPU scheduling, batching and circuit breakers.
- Backup, restore, migration and disaster-recovery drills.
- Egress firewall, TLS ingress, secret manager, parser isolation and malware scanning as deployment controls.

## Completion rule

A wave is not complete because files exist. Completion requires focused and adversarial tests, benchmark evidence, documentation, migration and rollback paths, operational limits, security review and an updated status ledger. Release readiness additionally requires the exact-head repository workflow to pass on one unchanged commit.
