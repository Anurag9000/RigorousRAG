# Canonical GraphRAG citation and agent integration

Last updated: 2026-08-02

## Purpose

The authoritative GraphRAG selector already returned privacy-finalized evidence with exact graph-set, generation, node and traversal lineage, but deliberately could not create citations or enter the research agent. This slice connects it to the repository’s existing server-owned `tools.models.Citation` boundary without introducing a parallel citation schema or answer generator.

## Canonical citation conversion

`tools/evidence_graph_citations.py` converts one authority-checked `GraphEvidenceSelection` into the same `Citation` model used by uploaded-document, web, internal-index and handbook tools.

Each citation contains:

- a server-relabelable `[n]` label;
- the evidence-node label as title;
- a `local://<document-id>` URL;
- `uploaded_document` source type so existing API/browser rendering remains compatible;
- bounded privacy-finalized snippet and quote text;
- generation-scoped node ID as `chunk_id`;
- graph evidence digest as `source_id`;
- document ID and page number;
- graph-set ID and digest;
- authoritative member-generation and graph digest;
- node type, ID and provenance digest;
- evidence and text digests;
- lexical/within-document/cross-document origin;
- bounded graph score;
- traversal-step digests;
- section title;
- matched-term count and digest.

The adapter does **not** emit:

- authenticated owner IDs;
- raw query text;
- raw matched query terms;
- retained-source paths;
- provider responses;
- unreviewed relations;
- a second citation model.

It verifies that every item remains within the selection owner scope and refuses a selection that already claims citation conversion or answer generation. An abstained empty selection produces an empty citation list; an inconsistent abstention fails closed.

## Authoritative retrieval tool

`tools/evidence_graph_rag_tool.py` exposes:

```python
search_evidence_graph(...)
```

and the closed tool definition:

```text
search_evidence_graph
```

Required arguments:

- `query`;
- `graph_set_key`.

Optional bounded controls include:

- node types;
- within-document edge types;
- cross-document edge types;
- allowed evidence origins;
- per-document lexical hits;
- lexical seed ceiling;
- within-document expansion ceiling;
- cross-document depth and per-seed expansion ceilings;
- maximum citations, capped at 50.

The caller cannot supply `owner_id` through the tool schema. The server injects it. The tool obtains the current graph-set, generation and graph stores through existing runtime factories, invokes `select_current_graph_set_evidence`, and converts only through `graph_evidence_to_citations`.

No answer is generated. Empty evidence returns an empty citation list.

## Lazy research-agent registration

`tools/evidence_graph_agent_integration.py` extends the existing `search_agent_legacy` surface instead of replacing it.

It:

1. appends exactly one GraphRAG definition to `TOOLS_SCHEMA`;
2. updates the existing `_TOOL_PARAMETER_SCHEMAS` registry;
3. wraps only `SearchAgent._dispatch`;
4. injects `self.owner_id` at dispatch;
5. calls `search_evidence_graph`;
6. returns canonical `Citation` objects to the existing `ToolExecution` path;
7. preserves the original dispatcher for every other tool;
8. adds one capability line to the existing system prompt;
9. remains idempotent.

`tools/evidence_graph_agent_import_hook.py` targets only top-level `search_agent_legacy` and delegates module loading to Python’s standard `PathFinder` loader. It remains lazy: unrelated `tools` imports do not load the GraphRAG stack. It also refuses to patch a partially initialized agent module.

The hook is activated from `tools.__init__`, alongside the existing security, lifecycle and RAG-strategy compatibility hooks.

## Existing server-owned publication path

After dispatch, the existing agent remains responsible for:

- tool argument schema validation;
- process-wide admission and timeout controls;
- citation-type filtering;
- evidence deduplication;
- server relabeling;
- final-answer marker filtering;
- `AgentAnswer` construction;
- API serialization;
- frontend citation rendering.

GraphRAG does not create a new evidence registry. Its citations enter the same registry used by every other research tool.

## Focused verification

The local citation/tool/agent-hook slice passed **13/13 tests** covering:

- canonical `Citation` construction;
- graph/path/generation metadata preservation;
- owner-scope refusal;
- omission of owner IDs, raw terms, raw queries and source paths;
- abstention and inconsistent-abstention handling;
- origin filtering, limits, label offsets and generation-scoped deduplication;
- closed tool schema and bounded argument forwarding;
- runtime dependency injection;
- invalid filter/budget refusal before selection;
- idempotent agent schema/dispatch installation;
- authenticated owner injection;
- fallback dispatcher preservation;
- incompatible module refusal;
- unique lazy-hook registration.

A production-agent contract is also committed in `tests/unit/test_evidence_graph_agent_live.py`. It exercises the real `search_agent.SearchAgent._execute_tool` and citation-registration boundary. That test was not executed in the isolated local harness and must be included in the next fresh exact-current repository run.

Together with the publication-journal and operations suites, the available local post-head harness passed **45/45 tests**, and compilation passed for the new modules.

## Remaining verification and integration

- Run the complete repository test suite on one unchanged final `main` SHA.
- Run the production live-agent test in that exact repository environment.
- Add API endpoint response tests containing graph citations.
- Add browser safe-DOM rendering tests for graph metadata and locators.
- Add connected-provider tool-selection tests.
- Add Windows and Docker/Compose verification.
- Add full concurrency/fault injection with unrelated processes.

## Permanent non-claims

- A canonical citation establishes provenance, not scientific truth.
- A graph path is not proof of causality or entailment.
- A reviewed relation remains an assertion subject to reviewer error.
- Agent registration does not imply that a model will select the tool appropriately.
- Focused local verification is not the complete release matrix.
- Release readiness is not claimed.
