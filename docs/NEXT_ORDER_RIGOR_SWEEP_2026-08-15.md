# RigorousRAG next-order rigor sweep — 2026-08-15

This document follows `IMPLEMENTATION_SWEEP_2026-08-15.md`. It records the second large
implementation-only pass performed under the same exclusions: no test/CI execution, no
real dataset downloads/curation, and no actual model training/fine-tuning.

## Implemented beyond the original backlog

The second pass added concrete software for:

- injected S3-compatible object storage, Redis-compatible lease queues and external
  secret providers without SDK installation or credential discovery;
- retraction/withdrawal/supersession propagation into graph/report state;
- content-addressed reproducible research capsules and verified replay orchestration;
- causal-vs-associational variables, assumptions, DAGs and claim readiness;
- CRS-explicit spatiotemporal retrieval and hydrologic upstream/downstream topology;
- transparent source trust/applicability policies separated from truth claims;
- provenance lineage and downstream-impact queries;
- dependency-aware exact-confirmation retention execution;
- evidence-aware answer versioning and claim/citation/input diffs;
- cryptographic manifest-attestation contracts plus a local HMAC-SHA256 reference signer;
- owner-scoped research-project ACL roles and explicit permissions;
- declarative experiment DAG specifications compiled to deterministic execution tasks;
- explicit schema-evolution/migration registry with migration-path fingerprints;
- structured citation-linked numerical consistency checks;
- built-in hydrology domain registration.

## Remaining implementation-only gaps after both sweeps

The following are the most material remaining software tasks. Items that require actual
training, downloading real data, or running the full test/release matrix are intentionally
not listed as implementation gaps here.

### A. Live integration and compatibility retirement

1. Instantiate the default capability/domain registries in the production composition
   root and expose them through `/research/capabilities` by default.
2. Migrate the older evidence-graph and multi-hop compatibility wrappers completely into
   native `AgentToolRegistry` specifications, then retire one-off monkey-patch/import-hook
   dispatch state once every live tool is registry-native.
3. Automatically append successful finalized research-agent results to a selected
   research workspace session using server-computed query/result/citation fingerprints.
4. Provide server-authoritative finalized-result storage so reports/evidence matrices can
   be generated from result IDs rather than accepting client-authored citations.
5. Apply typed `RuntimeConfig` as the composition root for production backend/provider
   construction instead of keeping legacy environment-specific assembly alongside it.
6. Install optional entailment, learned-routing, multimodal and domain providers through
   capability selection at application startup rather than per-agent manual attributes.

### B. Document-native scientific extraction depth

1. Provider-specific layout adapters that emit `ScientificDocumentIR` directly from
   layout models while preserving deterministic fallback semantics.
2. Rich table structure adapters for multi-row/multi-column headers, merged cells and
   cross-page tables.
3. Mathematical OCR/symbol canonicalization adapters and equation-reference resolution.
4. Chart/plot series extraction with axis-scale/unit/legend provenance.
5. Figure panel segmentation and image-region/caption/reference alignment beyond current
   deterministic caption linking and generic multimodal regions.
6. Large-corpus page-vector compression/ANN storage for page-native late interaction.

### C. Scientific reasoning depth

1. Domain extraction adapters that populate PICO/effect/risk/certainty/causal fields from
   source evidence under human-review lineage.
2. Additional reviewed synthesis methods such as robust variance estimation,
   meta-regression and network meta-analysis where assumptions are explicitly satisfied.
3. Causal adjustment-set, negative-control and sensitivity-analysis helpers that remain
   clearly separated from causal proof.
4. Formal consistency checks across prose claims, evidence-matrix values, tables,
   equations and cited source quantities.
5. Structured uncertainty propagation from extraction through synthesis through final
   answer wording.

### D. Hydrology/geospatial depth

1. Optional operator-installed GeoTIFF/NetCDF/PostGIS adapters.
2. HEC-RAS geometry/reach/cross-section parsing into authoritative geospatial objects.
3. Basin/reach snapping and topology reconciliation against raster/vector river networks.
4. Spatiotemporal query compilation to PostGIS/R-tree indexes for large deployments.
5. Map/raster citation visualization and profile/hydrograph comparison interfaces.
6. Additional model/result adapters requested by operators while preserving no-execution
   and source-identity boundaries.

### E. Production-provider depth

1. Native cloud KMS/public-key attestation signers and rotation policies.
2. Provider-native secure deletion/retention/version-lock adapters for object storage.
3. Kubernetes/OS hard parser sandboxes using seccomp/AppArmor/gVisor/job-level resource
   ceilings where deployments require stronger isolation than subprocess boundaries.
4. Dead-letter queue, retry-inspection and distributed admission-control operator APIs.
5. Concrete migration participants for each deployed vector/sparse/graph/object backend.
6. Multi-region replication/failover policy and regional consistency metadata.
7. Persist project ACLs and connect them to shared-project API authorization rather than
   owner-only project access.

### F. Product and operator UX

1. Evidence graph/path explorer.
2. PDF page/block/figure/table/formula citation viewer with coordinate highlights.
3. Claim support/contradiction and source-status/retraction panels.
4. PICO/effect/risk-of-bias/certainty/causal review screens.
5. Research report/evidence matrix/capsule/attestation export and replay screens.
6. Model/policy/capability candidate-shadow-canary-production lifecycle UI.
7. Migration/repair/reindex/adoption/retention/cutover operator UI.
8. Hydrology map, hydrograph, profile and scenario-comparison views.
9. Experiment-spec authoring and run-comparison UI.
10. Collaboration/ACL screens for shared research projects.

## Further ideas after the second sweep

These are genuinely additional directions rather than unimplemented parts of the original
RAG foundation:

- privacy-preserving multi-user annotations with field-level disclosure policies;
- signed reviewer decisions and provenance-bound peer-review workflows;
- federated retrieval across institutions without centralizing private source text;
- differential-privacy or secure-aggregation interfaces for aggregate feedback signals;
- evidence dependency invalidation jobs that automatically queue re-evaluation when an
  upstream source/model/policy changes;
- formal policy-as-code for allowable evidence types by domain/question class;
- counterfactual retrieval audits (which sources/rankers changed the supported answer);
- uncertainty-aware answer templates that bind language strength to reviewed evidence
  certainty rather than unconstrained generation;
- reproducible offline research bundles containing manifests, reports and public artifacts
  while keeping private source bytes as separately authorized references;
- domain-specific topology/reasoning adapters beyond hydrology using the same generic
  domain, provenance and capability registries.

## Claim boundary

Both sweeps are implementation commits only. They do not imply empirical superiority,
model calibration, complete security certification or release readiness. Real external
provider behavior, trained artifacts, real datasets and full exact-head verification are
separate work by explicit request.
