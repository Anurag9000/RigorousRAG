# Capability wave one diagnostic

```text
  File "/home/runner/work/RigorousRAG/RigorousRAG/scripts/capability_wave1_tests_docs.py", line 701
    ''', encoding="utf-8")
    ^
SyntaxError: unterminated triple-quoted string literal (detected at line 701)

```

```diff
diff --git a/tools/rag_tool.py b/tools/rag_tool.py
index f4ad44c..2b853c5 100644
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
+    retrieval_mode: str = "hybrid",
+    reranker: str = "heuristic",
+    candidate_pool: int = 20,
+    diversity_lambda: float = 0.82,
 ) -> List[Citation]:
     """Retrieve evidence with mandatory owner and document provenance checks."""
 
@@ -176,6 +226,11 @@ def search_uploaded_docs(
         minimum=1,
         maximum=_MAX_CITATIONS,
     )
+    mode = _choice(retrieval_mode, "retrieval_mode", {"dense", "lexical", "hybrid"})
+    reranker_name = _choice(reranker, "reranker", {"none", "heuristic"})
+    pool = _integer(candidate_pool, "candidate_pool", minimum=1, maximum=_MAX_CITATIONS)
+    pool = max(requested, pool)
+    diversity = _unit_float(diversity_lambda, "diversity_lambda")
 
     rag = get_rag_layer()
     if use_hyde:
@@ -193,7 +248,7 @@ def search_uploaded_docs(
             raise RuntimeError("The retrieval expansion backend returned invalid text.")
     chunks = rag.query(
         retrieval_query,
-        n_results=requested,
+        n_results=pool,
         owner_id=owner,
         doc_id=document_id,
         use_multi_query=use_multi_query,
@@ -201,8 +256,49 @@ def search_uploaded_docs(
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
+        source_id = metadata.get("doc_id")
+        if not isinstance(source_id, str):
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
+        per_source_limit=max(1, min(requested, 3)),
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
@@ -272,9 +368,18 @@ def search_uploaded_docs(
                         else None
                     ),
                     "relevance": round(
-                        _finite_score(_safe_attr(chunk, "score", 0.0)),
+                        ranking_scores.get(chunk_id).score
+                        if chunk_id in ranking_scores
+                        else _finite_score(_safe_attr(chunk, "score", 0.0)),
                         6,
                     ),
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
