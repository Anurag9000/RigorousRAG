# Capability Wave 1 verification record

## Committed implementation

- `tools/hybrid_retrieval.py`
- `tools/reranking.py`
- `tools/rag_tool.py`
- `evaluation/__init__.py`
- `experiments/__init__.py`
- `scripts/run_retrieval_benchmarks.py`
- focused tests under `tests/unit/`

## Local constrained verification

Executed from the generated source root with `PYTHONPATH=.`:

```text
12 passed
Python compileall: passed
AST parse of every changed Python file: passed
Git blob SHA comparison against local tested bytes: passed
```

The first one-shot attempt found and led to correction of a hostile metadata mapping boundary. Subsequent push-triggered one-shot workflows did not register reliably for connector-authored commits and are removed rather than retained as misleading automation.

## Not executed locally

Ruff was unavailable in the constrained environment, and external package resolution was blocked. The full repository matrix, Windows execution, Compose validation and Docker build therefore remain exact-head CI obligations. This record intentionally does not claim release readiness.
