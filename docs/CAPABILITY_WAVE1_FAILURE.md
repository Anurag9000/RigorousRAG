# Capability wave one diagnostic

```text
All checks passed!
.......................F..................                               [100%]
=================================== FAILURES ===================================
____________ test_hostile_metadata_mapping_is_skipped_without_leak _____________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7fc273b894f0>

    def test_hostile_metadata_mapping_is_skipped_without_leak(monkeypatch):
        class BrokenMetadata(dict):
            def get(self, *_args, **_kwargs):
                raise RuntimeError("private metadata detail")
    
        rag = MagicMock()
        rag.query.return_value = [
            SimpleNamespace(
                id="chunk-1",
                text="evidence",
                score=1.0,
                metadata=BrokenMetadata(),
            )
        ]
        monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)
    
>       assert rag_tool.search_uploaded_docs("question", owner_id="alice") == []
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/unit/test_rag_tool_backend_boundaries.py:82: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tools/rag_tool.py:273: in search_uploaded_docs
    metadata_owner = metadata.get("owner_id")
                     ^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = {}, _args = ('owner_id',), _kwargs = {}

    def get(self, *_args, **_kwargs):
>       raise RuntimeError("private metadata detail")
E       RuntimeError: private metadata detail

tests/unit/test_rag_tool_backend_boundaries.py:69: RuntimeError
=========================== short test summary info ============================
FAILED tests/unit/test_rag_tool_backend_boundaries.py::test_hostile_metadata_mapping_is_skipped_without_leak - RuntimeError: private metadata detail
1 failed, 41 passed in 1.50s

```

```diff
diff --git a/.github/workflows/release-locks.yml b/.github/workflows/release-locks.yml
index 4d0f9a5..a159f3a 100644
--- a/.github/workflows/release-locks.yml
+++ b/.github/workflows/release-locks.yml
@@ -5,7 +5,7 @@ on:
   pull_request:
   merge_group:
   push:
-    branches: [main, "agent/**"]
+    branches: [main]
     tags: ["v*"]
 
 permissions:
diff --git a/README.md b/README.md
index e9e4d16..9066f19 100644
--- a/README.md
+++ b/README.md
@@ -45,7 +45,7 @@ graph LR
     Evidence --> Answer[Bounded AgentAnswer]
 ```
 
-See [Goals and Architecture](docs/GOALS_AND_ARCHITECTURE.md), [Security Model](docs/SECURITY.md), [Remediation Status](docs/REMEDIATION_STATUS.md), and the continuation-audit records in `docs/`.
+See [Goals and Architecture](docs/GOALS_AND_ARCHITECTURE.md), [Security Model](docs/SECURITY.md), [Remediation Status](docs/REMEDIATION_STATUS.md), [Capability Expansion Roadmap](docs/CAPABILITY_EXPANSION_ROADMAP.md), [Exhaustive TODO](docs/TODO.md), and the continuation-audit records in `docs/`.
 
 ## Security and reliability properties
 
@@ -283,7 +283,7 @@ docker build --tag rigorousrag:local .
 
 CI is configured to run dependency checks, compile, fatal Ruff checks, pytest/coverage across Python 3.10, 3.11, and 3.12, validate Compose, and build the container. Coverage is a regression signal, not a correctness certificate.
 
-**Current PR verification warning:** the remediation environment cannot clone or download the branch because `github.com` DNS resolution fails. No exact-current-head GitHub Actions result has been observed through the available connector. The remediation PR must remain draft until every configured check executes against the final head and all failures are corrected.
+**Exact-head verification warning:** only `main` is active and changes are committed directly to it. A release claim requires the authoritative 16-job workflow to succeed for the exact current `main` SHA; a stale result from an older commit is not certification.
 
 ## Known limitations
 
diff --git a/docs/REMEDIATION_STATUS.md b/docs/REMEDIATION_STATUS.md
index b777924..46c92be 100644
--- a/docs/REMEDIATION_STATUS.md
+++ b/docs/REMEDIATION_STATUS.md
@@ -1,16 +1,16 @@
 # Exhaustive Remediation Status
 
 This document is the current status index for the repository-wide remediation begun on
-2026-07-27 and continued through **nine** regression/audit passes. It does not claim that
+2026-07-27 and continued through **fifteen** remediation passes and capability wave one. It does not claim that
 the software is defect-free or merge-ready. Detailed findings and changes are recorded in
 `CONTINUATION_AUDIT.md` and `CONTINUATION_AUDIT_PASS2.md` through
 `CONTINUATION_AUDIT_PASS9.md`.
 
 ## Current repository state
 
-- Branch: `agent/exhaustive-remediation`
-- Draft pull request: #1
-- Pull-request state: open, mergeable, draft
+- Default and only branch: `main`
+- Open pull requests: none
+- Development policy: verified commits directly to `main`
 - Authoritative workflow: `.github/workflows/release-locks.yml`
 - Configured gate: 16 jobs
 - Final-head executable result: **not yet established**
@@ -99,7 +99,7 @@ release certification. See `EXECUTABLE_VERIFICATION.md`.
 
 ## Required final gate
 
-PR #1 must remain draft until one final exact head completes:
+One unchanged final `main` head must complete:
 
 - Linux Python 3.10, 3.11, and 3.12 full dependency, whitespace, compilation, fatal Ruff,
   pytest, and measured coverage checks;
@@ -109,7 +109,7 @@ PR #1 must remain draft until one final exact head completes:
   `--require-hashes --no-deps --dry-run`, and artifact publication.
 
 Every failure must be corrected and the entire 16-job workflow rerun. The final diff,
-documentation, and generated artifacts must then be re-audited before the PR leaves draft.
+documentation, and generated artifacts must then be re-audited before a release claim is made.
 
 No current-head success is claimed. GitHub has not exposed a pull-request workflow run for
 connector-authored heads, and the available execution container cannot clone the branch
@@ -133,3 +133,8 @@ These remain disclosed rather than falsely marked complete:
 - readiness does not prove model availability or representative semantic retrieval;
 - structural provenance does not prove semantic entailment;
 - scientific outputs require source inspection, expert review, and replication.
+
+
+## Capability expansion
+
+Capability wave one adds hybrid fusion, BM25 candidate scoring, MMR diversity, optional reranking, normalized BEIR datasets, comprehensive retrieval/citation metrics, deterministic experiment manifests, resumable result storage, and an offline benchmark runner. The accepted exhaustive future program is tracked in `CAPABILITY_EXPANSION_ROADMAP.md` and `TODO.md`.
diff --git a/scripts/capability_wave1_tests_docs.py b/scripts/capability_wave1_tests_docs.py
index 14d3a1d..414de00 100644
--- a/scripts/capability_wave1_tests_docs.py
+++ b/scripts/capability_wave1_tests_docs.py
@@ -698,4 +698,3 @@ status = status.replace("PR #1 must remain draft until one final exact head comp
 status = status.replace("before the PR leaves draft.", "before a release claim is made.")
 status += "\n\n## Capability expansion\n\nCapability wave one adds hybrid fusion, BM25 candidate scoring, MMR diversity, optional reranking, normalized BEIR datasets, comprehensive retrieval/citation metrics, deterministic experiment manifests, resumable result storage, and an offline benchmark runner. The accepted exhaustive future program is tracked in `CAPABILITY_EXPANSION_ROADMAP.md` and `TODO.md`.\n"
 status_path.write_text(status, encoding="utf-8")
-''', encoding="utf-8")
diff --git a/tests/unit/test_deployment_parity.py b/tests/unit/test_deployment_parity.py
index 7d3abd8..efa01fa 100644
--- a/tests/unit/test_deployment_parity.py
+++ b/tests/unit/test_deployment_parity.py
@@ -63,6 +63,8 @@ def test_exact_head_workflow_is_unconditional_and_complete():
     assert "name: Exact-head verification and release locks" in workflow
     assert "  pull_request:\n" in workflow
     assert "  merge_group:\n" in workflow
+    assert "    branches: [main]\n" in workflow
+    assert "agent/**" not in workflow
     assert "paths:" not in workflow.split("permissions:", 1)[0]
     assert "python-version: [\"3.10\", \"3.11\", \"3.12\"]" in workflow
     assert "runs-on: windows-latest" in workflow
diff --git a/tools/rag_tool.py b/tools/rag_tool.py
index f4ad44c..bad417e 100644
--- a/tools/rag_tool.py
+++ b/tools/rag_tool.py
@@ -8,8 +8,10 @@ import operator
 from collections.abc import Mapping
 from typing import Any, List, Optional
 
+from tools.hybrid_retrieval import RetrievalCandidate, rank_candidates
 from tools.models import Citation
 from tools.rag import get_rag_layer
+from tools.reranking import build_reranker
 from tools.security import normalize_owner_id
 
 RAG_SEARCH_TOOL_DEF = {
@@ -45,6 +47,32 @@ RAG_SEARCH_TOOL_DEF = {
                     "description": "Generate a small number of alternative retrieval queries.",
                     "default": False,
                 },
+                "retrieval_mode": {
+                    "type": "string",
+                    "enum": ["dense", "lexical", "hybrid"],
+                    "description": "Dense, candidate-pool lexical, or fused hybrid ranking.",
+                    "default": "hybrid",
+                },
+                "reranker": {
+                    "type": "string",
+                    "enum": ["none", "heuristic"],
+                    "description": "Optional bounded second-stage reranker.",
+                    "default": "heuristic",
+                },
+                "candidate_pool": {
+                    "type": "integer",
+                    "minimum": 1,
+                    "maximum": 50,
+                    "description": "Dense candidate pool before fusion and diversity selection.",
+                    "default": 20,
+                },
+                "diversity_lambda": {
+                    "type": "number",
+                    "minimum": 0.0,
+                    "maximum": 1.0,
+                    "description": "MMR relevance/diversity trade-off; 1.0 disables redundancy penalty.",
+                    "default": 0.82,
+                },
             },
             "required": ["query"],
             "additionalProperties": False,
@@ -107,6 +135,24 @@ def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
     return numeric
 
 
+def _choice(value: Any, label: str, allowed: set[str]) -> str:
+    if not isinstance(value, str) or value not in allowed:
+        raise ValueError(f"{label} must be one of: {', '.join(sorted(allowed))}.")
+    return value
+
+
+def _unit_float(value: Any, label: str) -> float:
+    if isinstance(value, bool):
+        raise ValueError(f"{label} must be numeric.")
+    try:
+        numeric = float(value)
+    except (TypeError, ValueError, OverflowError) as exc:
+        raise ValueError(f"{label} must be numeric.") from exc
+    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
+        raise ValueError(f"{label} must be between 0 and 1.")
+    return numeric
+
+
 def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
     try:
         return getattr(value, name, default)
@@ -153,6 +199,10 @@ def search_uploaded_docs(
     agent_client: Optional[Any] = None,
     expansion_model: str = "gpt-4o-mini",
     n_results: int = 5,
+    retrieval_mode: str = "dense",
+    reranker: str = "none",
+    candidate_pool: Optional[int] = None,
+    diversity_lambda: float = 1.0,
 ) -> List[Citation]:
     """Retrieve evidence with mandatory owner and document provenance checks."""
 
@@ -176,6 +226,15 @@ def search_uploaded_docs(
         minimum=1,
         maximum=_MAX_CITATIONS,
     )
+    mode = _choice(retrieval_mode, "retrieval_mode", {"dense", "lexical", "hybrid"})
+    reranker_name = _choice(reranker, "reranker", {"none", "heuristic"})
+    pool = (
+        requested
+        if candidate_pool is None
+        else _integer(candidate_pool, "candidate_pool", minimum=1, maximum=_MAX_CITATIONS)
+    )
+    pool = max(requested, pool)
+    diversity = _unit_float(diversity_lambda, "diversity_lambda")
 
     rag = get_rag_layer()
     if use_hyde:
@@ -193,7 +252,7 @@ def search_uploaded_docs(
             raise RuntimeError("The retrieval expansion backend returned invalid text.")
     chunks = rag.query(
         retrieval_query,
-        n_results=requested,
+        n_results=pool,
         owner_id=owner,
         doc_id=document_id,
         use_multi_query=use_multi_query,
@@ -201,8 +260,52 @@ def search_uploaded_docs(
         expansion_model=model,
     )
 
+    candidate_chunks = _bounded_chunks(chunks, pool)
+    chunk_map: dict[str, Any] = {}
+    ranking_inputs: List[RetrievalCandidate] = []
+    for chunk in candidate_chunks:
+        raw_chunk_id = _safe_attr(chunk, "id", "")
+        raw_text = _safe_attr(chunk, "text", "")
+        metadata = _metadata(_safe_attr(chunk, "metadata", {}))
+        if not isinstance(raw_chunk_id, str) or not isinstance(raw_text, str):
+            continue
+        chunk_id = raw_chunk_id.strip()
+        metadata_owner = metadata.get("owner_id")
+        source_id = metadata.get("doc_id")
+        if metadata_owner != owner or not isinstance(source_id, str):
+            continue
+        if document_id is not None and source_id != document_id:
+            continue
+        try:
+            candidate = RetrievalCandidate(
+                candidate_id=chunk_id,
+                text=raw_text[:100_000],
+                source_id=source_id,
+                dense_score=_finite_score(_safe_attr(chunk, "score", 0.0)),
+                metadata=metadata,
+            )
+        except ValueError:
+            continue
+        chunk_map[chunk_id] = chunk
+        ranking_inputs.append(candidate)
+    ranked = rank_candidates(
+        retrieval_query,
+        ranking_inputs,
+        mode=mode,
+        limit=requested,
+        reranker=build_reranker(reranker_name),
+        diversity_lambda=diversity,
+        per_source_limit=(requested if document_id is not None else max(1, min(requested, 3))),
+    )
+    ordered_chunks = [
+        chunk_map[item.candidate.candidate_id]
+        for item in ranked
+        if item.candidate.candidate_id in chunk_map
+    ]
+    ranking_scores = {item.candidate.candidate_id: item for item in ranked}
+
     citations: List[Citation] = []
-    for chunk in _bounded_chunks(chunks, requested):
+    for chunk in ordered_chunks:
         metadata = _metadata(_safe_attr(chunk, "metadata", {}))
         try:
             metadata_owner = metadata.get("owner_id")
@@ -275,6 +378,19 @@ def search_uploaded_docs(
                         _finite_score(_safe_attr(chunk, "score", 0.0)),
                         6,
                     ),
+                    "ranking_score": round(
+                        ranking_scores.get(chunk_id).score
+                        if chunk_id in ranking_scores
+                        else _finite_score(_safe_attr(chunk, "score", 0.0)),
+                        6,
+                    ),
+                    "retrieval_mode": mode,
+                    "reranker": reranker_name,
+                    "rank_components": (
+                        dict(ranking_scores[chunk_id].components)
+                        if chunk_id in ranking_scores
+                        else {}
+                    ),
                 },
             )
         except Exception:

```
