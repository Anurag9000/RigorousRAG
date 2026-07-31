# Pass fourteen diagnostic

```text
Expected visual-entailment patch seam is missing.

```

```diff
diff --git a/tests/unit/test_rag_public_boundaries.py b/tests/unit/test_rag_public_boundaries.py
index 9344ae2..b7ec20d 100644
--- a/tests/unit/test_rag_public_boundaries.py
+++ b/tests/unit/test_rag_public_boundaries.py
@@ -246,3 +246,40 @@ def test_singleton_factory_refuses_symlinked_chroma_root(tmp_path):
 
     with pytest.raises(ValueError, match="CHROMA_PATH"):
         get_rag_layer(str(link))
+
+def test_rag_integer_limits_require_the_index_protocol():
+    from decimal import Decimal
+    from fractions import Fraction
+
+    class ExactIndex:
+        def __index__(self):
+            return 3
+
+    layer = _layer()
+    assert layer.query("question", owner_id="alice", n_results=ExactIndex()) == []
+    for value in (1.0, Decimal("1"), Fraction(1, 1), Fraction(3, 2)):
+        with pytest.raises(ValueError, match="n_results"):
+            layer.query("question", owner_id="alice", n_results=value)
+
+
+def test_reparse_chroma_component_is_rejected(monkeypatch, tmp_path):
+    from types import SimpleNamespace
+
+    root = tmp_path / "vectors"
+    root.mkdir()
+    original_lstat = type(root).lstat
+
+    def reparse_lstat(self):
+        metadata = original_lstat(self)
+        if self == root:
+            return SimpleNamespace(
+                st_mode=metadata.st_mode,
+                st_file_attributes=rag_module._WINDOWS_REPARSE_POINT,
+                st_dev=metadata.st_dev,
+                st_ino=metadata.st_ino,
+            )
+        return metadata
+
+    monkeypatch.setattr(type(root), "lstat", reparse_lstat)
+    with pytest.raises(ValueError, match="reparse points"):
+        RAGLayer(persist_directory=str(root))
diff --git a/tools/integrity.py b/tools/integrity.py
index 6ad133a..969a25c 100644
--- a/tools/integrity.py
+++ b/tools/integrity.py
@@ -12,14 +12,46 @@ from tools import integrity_boundary as _implementation
 from tools.security import normalize_owner_id
 
 _MAX_SCIENTIFIC_JSON_CHARS = 100_000
-_original_extract_figure_region = _implementation._extract_figure_region
-_original_check_visual_entailment = _implementation.check_visual_entailment
-_original_compare_papers = _implementation.compare_papers
-_original_generate_comparison_matrix = _implementation.generate_comparison_matrix
-_original_extract_protocol = _implementation.extract_protocol
-_original_run_scientific_debate = _implementation.run_scientific_debate
-_original_detect_conflicts = _implementation.detect_conflicts
-_original_extract_limitations = _implementation.extract_limitations
+
+
+def _persisted_original(name: str, value: Any) -> Any:
+    if not hasattr(_implementation, name):
+        setattr(_implementation, name, value)
+    return getattr(_implementation, name)
+
+
+_original_extract_figure_region = _persisted_original(
+    "_integrity_final_original_extract_figure_region",
+    _implementation._extract_figure_region,
+)
+_original_check_visual_entailment = _persisted_original(
+    "_integrity_final_original_check_visual_entailment",
+    _implementation.check_visual_entailment,
+)
+_original_compare_papers = _persisted_original(
+    "_integrity_final_original_compare_papers",
+    _implementation.compare_papers,
+)
+_original_generate_comparison_matrix = _persisted_original(
+    "_integrity_final_original_generate_comparison_matrix",
+    _implementation.generate_comparison_matrix,
+)
+_original_extract_protocol = _persisted_original(
+    "_integrity_final_original_extract_protocol",
+    _implementation.extract_protocol,
+)
+_original_run_scientific_debate = _persisted_original(
+    "_integrity_final_original_run_scientific_debate",
+    _implementation.run_scientific_debate,
+)
+_original_detect_conflicts = _persisted_original(
+    "_integrity_final_original_detect_conflicts",
+    _implementation.detect_conflicts,
+)
+_original_extract_limitations = _persisted_original(
+    "_integrity_final_original_extract_limitations",
+    _implementation.extract_limitations,
+)
 _SAFE_VISUAL_ERROR_PREFIXES = (
     "The retained PDF source bytes are missing or oversized.",
     "figure_id must contain",
diff --git a/tools/integrity_boundary.py b/tools/integrity_boundary.py
index 978726e..070356e 100644
--- a/tools/integrity_boundary.py
+++ b/tools/integrity_boundary.py
@@ -36,8 +36,24 @@ for _name, _default, _minimum, _maximum in (
 from tools import integrity_legacy as _implementation
 from tools.security import DEFAULT_MAX_UPLOAD_BYTES
 
-_original_compare_papers = _implementation.compare_papers
-_original_generate_comparison_matrix = _implementation.generate_comparison_matrix
+if not hasattr(_implementation, "_integrity_boundary_original_compare_papers"):
+    _implementation._integrity_boundary_original_compare_papers = (
+        _implementation.compare_papers
+    )
+if not hasattr(
+    _implementation,
+    "_integrity_boundary_original_generate_comparison_matrix",
+):
+    _implementation._integrity_boundary_original_generate_comparison_matrix = (
+        _implementation.generate_comparison_matrix
+    )
+
+_original_compare_papers = (
+    _implementation._integrity_boundary_original_compare_papers
+)
+_original_generate_comparison_matrix = (
+    _implementation._integrity_boundary_original_generate_comparison_matrix
+)
 
 
 def _contains_ascii_control(value: str) -> bool:
@@ -213,7 +229,7 @@ def check_visual_entailment(
             ).model_dump()
         )
     try:
-        image_b64, page_number, caption_text = _implementation._extract_figure_region(
+        image_b64, page_number, caption_text = _extract_figure_region(
             source_bytes,
             figure,
         )
diff --git a/tools/rag.py b/tools/rag.py
index f59e498..33acd84 100644
--- a/tools/rag.py
+++ b/tools/rag.py
@@ -5,7 +5,9 @@ from __future__ import annotations
 import itertools
 import json
 import math
+import operator
 import os
+import stat
 import sys
 from collections.abc import Mapping
 from pathlib import Path
@@ -30,9 +32,34 @@ for _name, _default, _minimum, _maximum in (
         write_back=True,
     )
 
+_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
+
+
+def _is_redirecting_path(metadata: Any) -> bool:
+    return stat.S_ISLNK(metadata.st_mode) or bool(
+        int(getattr(metadata, "st_file_attributes", 0))
+        & _WINDOWS_REPARSE_POINT
+    )
+
+
+def _reject_redirecting_components(path: Path) -> None:
+    for candidate in (path, *path.parents):
+        try:
+            metadata = candidate.lstat()
+        except FileNotFoundError:
+            continue
+        except OSError as exc:
+            raise ValueError("CHROMA_PATH could not be validated.") from exc
+        if _is_redirecting_path(metadata):
+            raise ValueError(
+                "CHROMA_PATH may not contain symbolic links or reparse points."
+            )
+
+
 _raw_chroma_path = Path(os.getenv("CHROMA_PATH", "rag_storage"))
-if _raw_chroma_path.is_symlink():
-    raise ValueError("CHROMA_PATH may not be a symbolic link.")
+if not _raw_chroma_path.is_absolute():
+    _raw_chroma_path = Path.cwd() / _raw_chroma_path
+_reject_redirecting_components(Path(os.path.abspath(_raw_chroma_path)))
 
 from tools import rag_legacy as _implementation
 from tools.security import normalize_owner_id
@@ -107,11 +134,9 @@ def _bounded_integer(value: Any, label: str, *, minimum: int, maximum: int) -> i
     if isinstance(value, bool):
         raise ValueError(f"{label} must be an integer.")
     try:
-        numeric = int(value)
+        numeric = int(operator.index(value))
     except (TypeError, ValueError, OverflowError) as exc:
         raise ValueError(f"{label} must be an integer.") from exc
-    if isinstance(value, float) and not value.is_integer():
-        raise ValueError(f"{label} must be an integer.")
     if not minimum <= numeric <= maximum:
         raise ValueError(f"{label} must be between {minimum} and {maximum}.")
     return numeric
@@ -263,14 +288,7 @@ def _absolute_storage_path(value: Any) -> str:
     if not raw.is_absolute():
         raw = Path.cwd() / raw
     absolute = Path(os.path.abspath(raw))
-    for candidate in (absolute, *absolute.parents):
-        try:
-            if candidate.is_symlink():
-                raise ValueError(
-                    "CHROMA_PATH may not contain symbolic-link components."
-                )
-        except OSError as exc:
-            raise ValueError("CHROMA_PATH could not be validated.") from exc
+    _reject_redirecting_components(absolute)
     return str(absolute)
 
 
@@ -327,7 +345,13 @@ def _bounded_generated_queries(value: object, maximum: int) -> List[str]:
         return []
 
 
-class RAGLayer(_implementation.RAGLayer):
+if not hasattr(_implementation, "_boundary_original_RAGLayer"):
+    _implementation._boundary_original_RAGLayer = _implementation.RAGLayer
+
+_BaseRAGLayer = _implementation._boundary_original_RAGLayer
+
+
+class RAGLayer(_BaseRAGLayer):
     """RAG implementation with caller-independent input and result validation."""
 
     def __init__(
@@ -672,8 +696,13 @@ class RAGLayer(_implementation.RAGLayer):
         )[:requested]
 
 
-_RAG_INSTANCES: Dict[str, RAGLayer] = {}
-_RAG_LOCK = _implementation.threading.Lock()
+if not hasattr(_implementation, "_boundary_rag_instances"):
+    _implementation._boundary_rag_instances = {}
+if not hasattr(_implementation, "_boundary_rag_lock"):
+    _implementation._boundary_rag_lock = _implementation.threading.Lock()
+
+_RAG_INSTANCES: Dict[str, RAGLayer] = _implementation._boundary_rag_instances
+_RAG_LOCK = _implementation._boundary_rag_lock
 
 
 def get_rag_layer(

```
