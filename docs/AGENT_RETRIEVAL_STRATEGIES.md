# Authoritative agent retrieval strategies

Last updated: 2026-08-01

RigorousRAG exposes one authenticated research-agent tool named `search_uploaded_docs`. The tool now has four explicit strategies while preserving one server-owned citation registry and one API/browser response path.

## Why one tool surface

Adding separate adaptive, multi-hop and heterogeneous tools directly to the legacy dispatcher would create several risks:

- duplicated citation conversion and relabeling;
- different owner-scoping behavior across tools;
- separate result-size and evidence-source limits;
- inconsistent model-authored argument validation;
- accidental publication of trace, provider or storage metadata;
- a large security-sensitive rewrite of the mature agent loop.

Instead, the existing `tools.rag_tool` schema and callable are extended lazily after the classic implementation loads. The agent continues to import the same `RAG_SEARCH_TOOL_DEF` and `search_uploaded_docs` symbols. All successful strategies return only validated `Citation` objects.

## Lazy compatibility boundary

`tools/rag_strategy_import_hook.py` installs a `MetaPathFinder` that targets only `tools.rag_tool`.

1. Python's standard `PathFinder` locates and executes the original module.
2. `tools/rag_strategy_integration.install_rag_strategies` receives the original schema and callable.
3. The original implementation is retained as `_strategy_original_search_uploaded_docs`.
4. The public schema and callable are replaced with backward-compatible strategy-aware versions.
5. Reloading `tools.rag_tool` reruns the original source and reapplies the boundary.

The hook does not eagerly import the RAG stack when an unrelated `tools` submodule is imported.

## Strategies

### `single`

Calls the captured classic implementation unchanged and supports:

- dense retrieval;
- candidate-pool lexical or hybrid retrieval;
- generation-validated corpus sparse or hybrid retrieval;
- HyDE and bounded multi-query expansion;
- optional heuristic or cross-encoder reranking;
- candidate-pool and diversity controls.

This is the backward-compatible default.

### `adaptive`

Calls bounded adaptive/corrective uploaded-document retrieval. It:

- routes among existing uploaded-document retrieval modes;
- executes only within attempt and estimated-cost ceilings;
- evaluates evidence sufficiency;
- preserves privacy-safe optional traces;
- publishes no citations when the terminal result abstains;
- accepts only actual `Citation` objects from adaptive evidence;
- attaches bounded aggregate attempt, exhaustion and estimated-cost metadata.

Attempt traces and private provider details are not attached to citations.

### `multihop`

Calls uploaded-document adaptive multi-hop retrieval. It:

- uses deterministic or strict-schema model-assisted decomposition;
- validates an acyclic dependency graph;
- executes independent hops in parallel and dependent batches serially;
- enforces per-hop, global deadline, evidence and estimated-cost ceilings;
- publishes no evidence when terminal paths abstain;
- accepts only underlying `Citation` objects;
- attaches the retrieving hop, source, evidence identity, plan fingerprint, plan quality and bounded budget summary.

Dependency evidence may guide later retrieval but never becomes a synthetic citation.

### `heterogeneous`

Calls heterogeneous multi-hop research retrieval. It may route each subquestion across:

- uploaded dense retrieval;
- uploaded corpus sparse retrieval;
- uploaded corpus hybrid retrieval;
- public web retrieval;
- scholarly retrieval.

The strategy enforces global estimated workload, latency, monetary, evidence and deadline ceilings. Public routes never receive raw private dependency passages. Returned citations retain hop and route metadata plus bounded aggregate resource estimates.

Because current public provider adapters allow at most ten results per hop, `n_results` is capped at ten for this strategy.

## Argument separation

Classic controls such as `retrieval_mode`, `reranker`, `use_hyde`, `use_multi_query` and `candidate_pool` may be changed only for `single`. Non-single strategies reject ambiguous combinations rather than silently ignoring them.

The strategy wrapper validates:

- strategy names;
- result, attempt, subquestion and worker ceilings;
- finite positive deadlines;
- per-hop and global estimated-cost ceilings;
- decomposition model identifiers;
- scopes and domains;
- domain allowlists;
- chronological year ranges;
- heterogeneous workload, latency and monetary ceilings.

Model-authored arguments still pass through the agent's closed JSON schema before dispatch.

## Citation authority and API/browser propagation

The live service imports `SearchAgent` from `search_agent`, whose validated boundary delegates to `search_agent_legacy`. The legacy dispatcher still has one `search_uploaded_docs` branch and one citation return type.

Consequently every strategy passes through the same sequence:

1. authenticated request owner selected by the server;
2. closed tool schema validation;
3. bounded tool executor and deadline;
4. strategy-specific retrieval;
5. `Citation` validation and metadata sanitization;
6. server evidence deduplication and `[n]` relabeling;
7. selection of only markers actually used by the answer;
8. `AgentAnswer` validation;
9. FastAPI JSON serialization;
10. browser rendering through the existing safe DOM path.

The model cannot provide `owner_id`, trace stores, route adapters, provider clients or citation objects through tool arguments.

## Focused contracts

Committed tests cover:

- lazy first-import and reload installation;
- exact strategy schema and ceilings;
- adaptive Citation-only publication;
- adaptive abstention;
- multi-hop lineage and budget metadata;
- multi-hop abstention;
- heterogeneous route/resource lineage;
- ambiguous classic-control rejection;
- heterogeneous result and year-range limits;
- live agent schema visibility;
- live dispatcher owner scoping and argument forwarding;
- unknown-strategy refusal before retrieval.

The strategy helper, import hook and focused tests compile locally. A stubbed integration harness passed all four strategies plus import/reload behavior. Full repository and exact-head CI verification remain required.

## Non-claims

- Strategy metadata does not prove answer support.
- A route selection does not prove that the selected backend was optimal.
- Estimated workload, latency and monetary values are planning proxies unless backed by measured experiments.
- Adaptive or multi-hop non-abstention does not prove semantic entailment.
- A validated `Citation` proves that evidence passed the server boundary, not that a generated claim follows from it.
