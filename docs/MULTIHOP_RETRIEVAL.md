# Bounded multi-hop retrieval

RigorousRAG now includes a deterministic foundation for decomposing complex uploaded-document questions into an acyclic dependency graph and executing that graph without losing citation lineage.

## Components

- `tools/query_decomposition.py`
  - validates the original query and every subquestion;
  - supports explicit operator/model proposals or deterministic heuristic decomposition;
  - extracts bounded entity and temporal constraints;
  - rejects duplicate IDs, missing dependencies, cycles, controls and oversized values;
  - emits stable topological batches, terminal nodes and a SHA-256 plan fingerprint.
- `tools/multihop_retrieval.py`
  - runs independent nodes in the same batch concurrently;
  - runs dependent batches only after their prerequisites resolve;
  - separately supplies dependency evidence to the retrieval callback;
  - bounds workers, timeouts, per-hop results, dependency evidence and total evidence;
  - preserves hop, source, document and page lineage;
  - contains individual hop failures and timeouts;
  - skips dependent hops when required evidence is absent;
  - groups cross-hop evidence without replacing the original source identities.
- `tools/multihop_rag_tool.py`
  - exposes the public `search_uploaded_docs_multihop` tool contract;
  - uses bounded adaptive uploaded-document retrieval for each hop;
  - propagates explicit entity/time constraints;
  - derives only bounded lexical search hints from dependency evidence;
  - never promotes dependency prose into a new citation;
  - serializes citations and lineage in separate fields.

## Execution model

```mermaid
graph TD
    Q[Complex query] --> D[Validated decomposition plan]
    D --> B1[Independent batch]
    B1 --> H1[Hop 1 adaptive retrieval]
    B1 --> H2[Hop 2 adaptive retrieval]
    H1 --> L1[Source-preserving evidence]
    H2 --> L2[Source-preserving evidence]
    L1 --> B2[Dependent batch]
    L2 --> B2
    B2 --> H3[Dependent adaptive retrieval]
    H3 --> L3[Source-preserving evidence]
    L1 --> J[Provenance-safe joins]
    L2 --> J
    L3 --> J
```

A topological batch is a set of nodes whose dependencies are already resolved. Nodes inside one batch may run in parallel. Batches run serially. The executor does not let a dependent node run when a required prerequisite has no evidence unless the caller explicitly disables that safeguard.

## Citation-lineage rule

A dependency can influence a later search, but it cannot become a synthetic citation. Each returned evidence item retains:

- the hop that retrieved it;
- its original source identifier;
- its original document identifier when present;
- its original page number when present;
- its retrieval score;
- the untouched underlying citation/evidence object.

Cross-hop joins group compatible evidence by document or source. They do not collapse several sources into one invented source, do not change citation authority and do not imply entailment.

## Bounded constraint propagation

Dependent searches receive:

1. the validated subquestion;
2. extracted entity constraints;
3. extracted temporal constraints;
4. a small deterministic set of lexical terms derived from prerequisite evidence.

Raw prerequisite passages are not concatenated into the query. The propagated term list is bounded and filtered. This reduces prompt-injection and query-amplification risk while still letting later hops use facts discovered by earlier hops as retrieval hints.

## Failure behavior

- Invalid decomposition proposals fail before retrieval.
- Cyclic or dangling dependency graphs fail closed.
- A malformed retrieval collection becomes a hop error.
- A timed-out hop publishes no late evidence.
- One failed parallel hop does not erase successful sibling evidence.
- Dependent hops can be skipped when required evidence is missing.
- The final result abstains when terminal hops produce no evidence.

Python threads cannot forcibly terminate provider code that ignores deadlines. The executor records the timeout and refuses late evidence, but host-level isolation is still required for forcible termination.

## Verification committed

Focused contracts cover:

- comparison decomposition into two parallel lookups and one dependent comparison;
- explicit DAG batching;
- stable plan fingerprints;
- duplicate, dangling and cyclic dependency rejection;
- hostile iterable and boolean-limit rejection;
- parallel evidence preservation;
- dependent-hop evidence delivery;
- source-preserving joins;
- missing-dependency skips;
- contained backend errors;
- timeout behavior without fabricated evidence;
- public adaptive multi-hop execution;
- bounded dependency-term propagation;
- citation/lineage payload separation.

The focused local suite passed 12 tests. This is not a substitute for the complete exact-head repository matrix.

## Remaining multi-hop work

- model-assisted decomposition under a strict schema and deterministic fallback;
- learned decomposition ranking;
- dynamic entity resolution and temporal normalization;
- graph-aware retrieval budgets across heterogeneous corpora;
- web/scholarly/uploaded-document cross-corpus hops;
- benchmark datasets such as HotpotQA, 2WikiMultiHopQA, MuSiQue and scientific multi-document tasks;
- answer-level support/entailment evaluation per hop and over the final synthesis;
- latency, cost, memory, coverage and calibration regression gates;
- agent registration and browser/API presentation after full integration tests.

## Non-claims

- A valid dependency graph does not prove that the decomposition is semantically optimal.
- A cross-hop join does not prove that two passages support the same claim.
- Retrieval success does not prove answer correctness.
- Citation lineage does not prove semantic entailment.
- Final scientific conclusions still require source inspection, expert review and replication.
