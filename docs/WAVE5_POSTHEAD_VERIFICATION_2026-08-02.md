# Wave 5 post-head focused verification — 2026-08-02

## Scope

This addendum records focused verification performed after the earlier unchanged-archive Wave 5 and full-repository run. It covers the newest durable publication-recovery, publication-operations, canonical citation, agent registration, API/frontend contract and authoritative graph-set discovery work.

It does not retroactively extend the earlier exact-current 114-test evidence-graph count.

## Newly committed after the prior exact-current archive

- Durable graph-set publication phase journal and crash recovery.
- Publication audit and planning-only retention classification.
- Canonical GraphRAG-to-`Citation` conversion.
- Standalone authoritative `search_evidence_graph` tool.
- Lazy idempotent registration into the existing research agent.
- Production-agent, API serialization and safe frontend rendering contracts.
- Owner-scoped `list_evidence_graph_sets` discovery tool.
- Stale-set filtering and aggregate-only unavailable reporting.
- Read-only discovery CLI and script wrapper.
- Pointer key, ID, digest and schema integrity checks.

## Focused local result

The complete locally available post-head harness passed:

```text
52 passed
```

Python compilation also passed for the local publication, citation, GraphRAG tool, agent-hook, discovery and focused-test modules.

The 52 tests include:

- publication journal identity, leases and transitions;
- publication crash recovery and compensation;
- publication audit and retention planning;
- canonical citation conversion and privacy metadata;
- authoritative GraphRAG wrapper validation;
- fake-module agent registration and fallback preservation;
- graph-set discovery via future public and current private store boundaries;
- current pointer key/ID/digest/schema checks;
- stale/unavailable authority filtering;
- read-only discovery CLI behavior.

## Committed but not executed in the isolated harness

These repository-native tests are committed but require a fresh exact-current checkout:

- `tests/unit/test_evidence_graph_agent_live.py`;
- `tests/integration/test_evidence_graph_api.py`;
- `tests/unit/test_evidence_graph_frontend_contract.py`;
- the complete pre-existing repository test suite.

## Last unchanged-archive full verification

The last full verification on one unchanged earlier archive remains:

- 114/114 evidence-graph focused tests;
- complete repository pytest pass;
- `pip check` pass;
- whole-tree Python compilation pass.

That archive predates the newest publication/citation/agent/discovery commits.

## Repository topology at audit time

- Only `main` was returned by the branch audit.
- No open pull requests were returned.
- No combined status checks were visible for the audited code head.

## Verification still required

- Fresh unchanged-head complete pytest and coverage run.
- Production live-agent discovery/search execution.
- FastAPI graph-citation integration execution.
- Browser safe-DOM test execution.
- Ruff and configured lint policy.
- Windows.
- Docker/Compose build and readiness.
- Connected-provider tool-selection behavior.
- Multi-process publication and pointer-race fault injection.

## Permanent non-claims

- Focused local contracts are not the complete release matrix.
- A current reviewed graph set is provenance-aligned, not scientifically true.
- Citation and discovery integration do not prove correct model tool selection.
- Release readiness is not claimed.
