# Multi-hop research-agent integration

Status: implemented on `main` as a fail-closed extension of the existing research-agent tool boundary.

## Scope

The bounded multi-hop uploaded-document pipeline already provided deterministic or model-assisted decomposition, dependency-DAG execution, per-hop/global deadlines, estimated-cost allocation, adaptive retrieval, immutable hop/source/document/page lineage, evidence joins, trace persistence, and terminal-evidence abstention. The missing production link was registration on the live `SearchAgent` tool surface used by the FastAPI research path.

This integration registers `search_uploaded_docs_multihop` without replacing the core reasoning loop. It reuses the same schema validation, owner scope, tool concurrency/admission limits, timeout handling, server-side citation registry, citation relabeling, provider-content bounds, and evidence-oriented system prompt as the other research tools.

## Provenance and fail-closed behavior

The multi-hop implementation carries each original server-constructed uploaded-document `Citation` in the hop evidence record. The agent integration extracts only those authoritative citation objects and hands them to the existing central citation registry.

The JSON result visible to the model deliberately strips hop-local citation payloads and labels. It retains only decomposition/budget/trace/join data plus evidence lineage. This avoids exposing labels that could disagree with the request-global citation labels assigned by the research agent.

If terminal evidence is absent and the multi-hop pipeline abstains, the integration returns zero authoritative citations. Partial prerequisite-hop evidence therefore cannot be promoted into a citation-supported final claim through this tool call.

## Import and reload safety

The existing `search_agent_legacy` import hook now installs both reviewed evidence-graph tools and the multi-hop tool through a single deferred watcher. This avoids competing temporary module classes while preserving independent idempotence for both integrations. Reloads clear both integrations' installation markers before re-execution and reinstall exactly one schema/dispatch wrapper for each capability.

## Verification contract

Unit coverage verifies:

- closed-schema registration;
- authenticated owner propagation;
- reuse of the configured agent client and retrieval-expansion model;
- stripping of non-authoritative local citation payloads;
- deduplication and bounded return of authoritative citations;
- zero-citation behavior on multi-hop abstention;
- idempotent installation and preservation of pre-existing dispatch;
- direct `search_agent_legacy`, public `search_agent`, and module-reload import orders.

The repository-level exact-head and release-lock workflows remain the authority for complete-commit verification.
