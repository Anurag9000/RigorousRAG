# Capability Wave 2C — generation-validated corpus hybrid retrieval

Last updated: 2026-08-01

## Purpose

Wave 1 lexical and hybrid modes reranked only the candidate pool already returned by dense retrieval. That improves ordering but cannot recover a document that dense retrieval never selected. Wave 2C adds independent corpus-level sparse retrieval and fuses its document candidates with dense retrieval before selecting evidence.

## Implemented modes

`search_uploaded_docs` now supports:

- `dense` — historical dense ordering;
- `lexical` — BM25 over the bounded dense candidate pool;
- `hybrid` — dense/candidate-pool lexical fusion;
- `corpus-sparse` — persistent fielded sparse retrieval across the owner corpus;
- `corpus-hybrid` — independent dense and sparse candidate generation followed by document-level fusion.

The existing dense default remains unchanged for compatibility.

## Retrieval sequence

1. validate query, owner, optional document ID, budgets and mode;
2. optionally generate bounded HyDE and multi-query variants;
3. retrieve dense chunks and sparse documents independently;
4. discard cross-owner, malformed and document-filter-mismatched results;
5. load the durable current generation for every candidate document;
6. reject deleted, missing or stale generations;
7. verify sparse generation and embedding-profile fingerprint;
8. verify dense content hash and embedding-profile fingerprint;
9. fuse document scores;
10. materialize dense chunks and sparse fields with page/section/position provenance;
11. apply bounded MMR evidence selection;
12. optionally apply one bounded second-stage reranker;
13. emit citations with protected generation and score traces.

## Provenance returned

Corpus citations may include:

- raw dense relevance;
- sparse score;
- fused evidence score;
- durable generation sequence;
- embedding-profile fingerprint;
- dense-chunk or sparse-field evidence kind;
- field type;
- page and section;
- term frequencies and token positions;
- bounded filename/title metadata.

Protected ranking and generation fields are written after all optional evidence extras, so evidence metadata cannot overwrite them.

## Failure behavior

- One failed expanded query does not discard successful variants.
- If every corpus query fails, retrieval is explicitly unavailable.
- Missing/deleted generation pointers are not returned.
- Stale sparse generations and mismatched dense hashes/profiles are not returned.
- Cross-owner dense chunks are discarded before fusion.
- Malformed citation candidates are skipped individually.

## Focused contracts

Tests cover:

- retrieval of a sparse-only document absent from dense candidates;
- stale sparse-generation rejection;
- dense owner/profile/content-hash validation;
- document-filter propagation to both retrievers;
- deleted and missing manifest exclusion;
- corpus-mode routing and citation provenance;
- protected metadata precedence;
- partial multi-query failure;
- all-query failure;
- validation before sparse/generation initialization.

## Remaining Wave 2C work

- benchmark-calibrated fusion weights rather than fixed defaults;
- reciprocal-rank fusion at the independent corpus layer;
- explicit per-document/source caps;
- field, date, MIME, section and provenance filters;
- multi-stage reranker cascades with latency and cost budgets;
- profile-aware reindex and migration workflows;
- shadow-index validation and atomic cutover;
- benchmark reports comparing dense, sparse, candidate hybrid and corpus hybrid modes;
- exact-head full-suite and container verification.

## Non-claims

Generation alignment establishes storage provenance, not semantic truth. A high sparse, dense, fused or reranker score does not prove factual correctness or claim entailment.
