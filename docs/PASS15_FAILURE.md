# Pass fifteen diagnostic

```text
ERROR: file or directory not found: tests/unit/test_search_agent_extended.py


no tests ran in 0.00s

```

```diff
diff --git a/search_agent.py b/search_agent.py
index 67a07b6..4ed45c9 100644
--- a/search_agent.py
+++ b/search_agent.py
@@ -10,6 +10,7 @@ from __future__ import annotations
 import itertools
 import json
 import math
+import operator
 import os
 import re
 import sys
@@ -52,8 +53,20 @@ import search_agent_legacy as _implementation
 
 from tools.security import normalize_owner_id
 
-_original_validate_schema_value = _implementation._validate_schema_value
-_original_tool_execution = _implementation.ToolExecution
+if not hasattr(_implementation, "_boundary_original_validate_schema_value"):
+    _implementation._boundary_original_validate_schema_value = (
+        _implementation._validate_schema_value
+    )
+if not hasattr(_implementation, "_boundary_original_ToolExecution"):
+    _implementation._boundary_original_ToolExecution = _implementation.ToolExecution
+if not hasattr(_implementation, "_boundary_original_SearchAgent"):
+    _implementation._boundary_original_SearchAgent = _implementation.SearchAgent
+
+_original_validate_schema_value = (
+    _implementation._boundary_original_validate_schema_value
+)
+_original_tool_execution = _implementation._boundary_original_ToolExecution
+_original_search_agent = _implementation._boundary_original_SearchAgent
 _MAX_IDENTIFIER_CHARS = 200
 _MAX_PROVIDER_FIELD_CHARS = 4096
 _INVALID_ARGUMENTS = "__INVALID_TOOL_ARGUMENTS__"
@@ -101,7 +114,7 @@ def _optional_provider_value(value: Any, label: str) -> Optional[str]:
         raise ValueError(
             f"{label} may contain at most {_MAX_PROVIDER_FIELD_CHARS:,} characters."
         )
-    if any(character in rendered for character in ("\x00", "\r", "\n")):
+    if any(ord(character) < 32 or ord(character) == 127 for character in rendered):
         raise ValueError(f"{label} contains invalid control characters.")
     return rendered or None
 
@@ -113,7 +126,7 @@ def _required_model(value: Any) -> str:
     if (
         not selected
         or len(selected) > _MAX_IDENTIFIER_CHARS
-        or any(character in selected for character in ("\x00", "\r", "\n"))
+        or any(ord(character) < 32 or ord(character) == 127 for character in selected)
     ):
         raise ValueError("model must contain between 1 and 200 valid characters.")
     return selected
@@ -123,17 +136,17 @@ def _strict_int(value: Any, label: str, *, minimum: int, maximum: int) -> int:
     if isinstance(value, bool):
         raise ValueError(f"{label} must be an integer.")
     try:
-        parsed = int(value)
+        parsed = int(operator.index(value))
     except (TypeError, ValueError, OverflowError) as exc:
         raise ValueError(f"{label} must be an integer.") from exc
-    if isinstance(value, float) and not value.is_integer():
-        raise ValueError(f"{label} must be an integer.")
     if not minimum <= parsed <= maximum:
         raise ValueError(f"{label} must be between {minimum} and {maximum}.")
     return parsed
 
 
 def _finite_timeout(value: Any, label: str) -> float:
+    if isinstance(value, bool):
+        raise ValueError(f"{label} must be numeric.")
     try:
         parsed = float(value)
     except (TypeError, ValueError, OverflowError) as exc:
@@ -213,7 +226,7 @@ def _bounded_tool_calls(raw_calls: Any, maximum: int) -> Tuple[List[_SafeToolCal
     ], overflow
 
 
-class ToolExecution(_original_tool_execution):
+class _ToolExecutionBoundary(_original_tool_execution):
     """Bound provider-controlled identifiers, content, citations, and telemetry."""
 
     def __init__(
@@ -264,7 +277,7 @@ class ToolExecution(_original_tool_execution):
         )
 
 
-class SearchAgent(_implementation.SearchAgent):
+class _SearchAgentBoundary(_original_search_agent):
     """Research agent with sanitized provider calls and authoritative evidence."""
 
     def __init__(
@@ -641,6 +654,13 @@ class SearchAgent(_implementation.SearchAgent):
         return "gpt-4o-mini"
 
 
+if not hasattr(_implementation, "_boundary_public_ToolExecution"):
+    _implementation._boundary_public_ToolExecution = _ToolExecutionBoundary
+if not hasattr(_implementation, "_boundary_public_SearchAgent"):
+    _implementation._boundary_public_SearchAgent = _SearchAgentBoundary
+ToolExecution = _implementation._boundary_public_ToolExecution
+SearchAgent = _implementation._boundary_public_SearchAgent
+
 _implementation._validate_schema_value = _validate_schema_value
 _implementation.ToolExecution = ToolExecution
 _implementation.SearchAgent = SearchAgent
diff --git a/storage.py b/storage.py
index fd1fd30..3839b60 100644
--- a/storage.py
+++ b/storage.py
@@ -31,7 +31,9 @@ os.environ["CLASSIC_MAX_SNAPSHOT_FILE_BYTES"] = str(
 
 import storage_legacy as _implementation
 
-_original_storage_manager = _implementation.StorageManager
+if not hasattr(_implementation, "_boundary_original_StorageManager"):
+    _implementation._boundary_original_StorageManager = _implementation.StorageManager
+_original_storage_manager = _implementation._boundary_original_StorageManager
 _FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
 
 
@@ -80,7 +82,7 @@ def _reject_json_constant(value: str) -> None:
     raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")
 
 
-class StorageManager(_original_storage_manager):
+class _StorageManagerBoundary(_original_storage_manager):
     """Classic storage manager with identity-bound, bounded persistent I/O."""
 
     def __init__(self, base_dir: Path | str = "data") -> None:
@@ -488,6 +490,10 @@ class StorageManager(_original_storage_manager):
         self._write_bytes(path, self._encode_json(payload))
 
 
+if not hasattr(_implementation, "_boundary_public_StorageManager"):
+    _implementation._boundary_public_StorageManager = _StorageManagerBoundary
+StorageManager = _implementation._boundary_public_StorageManager
+
 _implementation.StorageManager = StorageManager
 _implementation.__doc__ = __doc__
 sys.modules[__name__] = _implementation
diff --git a/tests/unit/test_compatibility_reload_boundaries.py b/tests/unit/test_compatibility_reload_boundaries.py
index 7a56f04..d461316 100644
--- a/tests/unit/test_compatibility_reload_boundaries.py
+++ b/tests/unit/test_compatibility_reload_boundaries.py
@@ -65,3 +65,63 @@ for _ in range(3):
 """
     )
     assert result.returncode == 0, result.stderr
+
+def test_stateful_class_wrappers_preserve_public_identity_across_reimports():
+    result = _run(
+        r"""
+import importlib
+import sys
+import tools
+
+# Classic storage.
+legacy_storage = importlib.import_module("storage_legacy")
+public_storage = importlib.import_module("storage")
+storage_base = legacy_storage._boundary_original_StorageManager
+storage_public = legacy_storage._boundary_public_StorageManager
+for _ in range(3):
+    sys.modules.pop("storage", None)
+    public_storage = importlib.import_module("storage")
+    assert public_storage.StorageManager is storage_public
+    assert storage_public.__mro__[1] is storage_base
+
+# Document registry.
+legacy_document = importlib.import_module("tools.document_store_legacy")
+public_document = importlib.import_module("tools.document_store")
+document_base = legacy_document._boundary_original_DocumentStore
+document_public = legacy_document._boundary_public_DocumentStore
+for _ in range(3):
+    sys.modules.pop("tools.document_store", None)
+    tools.__dict__.pop("document_store", None)
+    public_document = importlib.import_module("tools.document_store")
+    assert public_document.DocumentStore is document_public
+    assert document_public.__mro__[1] is document_base
+
+# Search agent.
+legacy_agent = importlib.import_module("search_agent_legacy")
+public_agent = importlib.import_module("search_agent")
+agent_base = legacy_agent._boundary_original_SearchAgent
+agent_public = legacy_agent._boundary_public_SearchAgent
+execution_base = legacy_agent._boundary_original_ToolExecution
+execution_public = legacy_agent._boundary_public_ToolExecution
+validator = legacy_agent._boundary_original_validate_schema_value
+for _ in range(3):
+    sys.modules.pop("search_agent", None)
+    public_agent = importlib.import_module("search_agent")
+    assert public_agent.SearchAgent is agent_public
+    assert agent_public.__mro__[1] is agent_base
+    assert public_agent.ToolExecution is execution_public
+    assert execution_public.__mro__[1] is execution_base
+    assert legacy_agent._boundary_original_validate_schema_value is validator
+
+# RAG public class, in addition to base/cache checks from pass fourteen.
+legacy_rag = importlib.import_module("tools.rag_legacy")
+public_rag = importlib.import_module("tools.rag")
+rag_public = legacy_rag._boundary_public_RAGLayer
+for _ in range(3):
+    sys.modules.pop("tools.rag", None)
+    tools.__dict__.pop("rag", None)
+    public_rag = importlib.import_module("tools.rag")
+    assert public_rag.RAGLayer is rag_public
+"""
+    )
+    assert result.returncode == 0, result.stderr
diff --git a/tests/unit/test_document_store_root_paths.py b/tests/unit/test_document_store_root_paths.py
index cdfa9b7..e874b1c 100644
--- a/tests/unit/test_document_store_root_paths.py
+++ b/tests/unit/test_document_store_root_paths.py
@@ -82,3 +82,41 @@ def test_visual_flag_cleanup_clock_and_identifiers_are_strict(tmp_path):
         store.get(owner_id="alice", doc_id="")
     with pytest.raises(ValueError, match="finite"):
         store.cleanup_orphans(now=float("nan"), job_store=object())
+
+def test_registry_detects_nonlink_root_replacement_and_boolean_clock(tmp_path):
+    parent = tmp_path / "state"
+    database = parent / "documents.sqlite3"
+    uploads = tmp_path / "uploads"
+    store = DocumentStore(database, uploads)
+
+    moved = tmp_path / "state-original"
+    parent.rename(moved)
+    parent.mkdir()
+    assert store.ping() is False
+
+    with pytest.raises(ValueError, match="numeric"):
+        store.cleanup_orphans(now=True, job_store=object())
+
+
+def test_registry_rejects_windows_reparse_components(monkeypatch, tmp_path):
+    from types import SimpleNamespace
+
+    uploads = tmp_path / "uploads"
+    uploads.mkdir()
+    database = tmp_path / "documents.sqlite3"
+    original_lstat = type(uploads).lstat
+
+    def reparse_lstat(self):
+        metadata = original_lstat(self)
+        if self == uploads:
+            return SimpleNamespace(
+                st_mode=metadata.st_mode,
+                st_file_attributes=0x400,
+                st_dev=metadata.st_dev,
+                st_ino=metadata.st_ino,
+            )
+        return metadata
+
+    monkeypatch.setattr(type(uploads), "lstat", reparse_lstat)
+    with pytest.raises(ValueError, match="reparse points"):
+        DocumentStore(database, uploads)
diff --git a/tests/unit/test_search_agent_provider_boundaries.py b/tests/unit/test_search_agent_provider_boundaries.py
index 8721a67..19ad24e 100644
--- a/tests/unit/test_search_agent_provider_boundaries.py
+++ b/tests/unit/test_search_agent_provider_boundaries.py
@@ -169,3 +169,29 @@ def test_matched_handbook_lookup_keeps_one_real_citation(monkeypatch):
     assert "Policy evidence" in content
     assert len(citations) == 1
     assert citations[0].source_type == "handbook"
+
+def test_agent_integer_limits_require_index_and_timeouts_reject_booleans():
+    from decimal import Decimal
+    from fractions import Fraction
+
+    class ExactIndex:
+        def __index__(self):
+            return 3
+
+    agent = SearchAgent(owner_id="alice", max_turns=ExactIndex())
+    assert agent.max_turns == 3
+    for value in (1.0, Decimal("2"), Fraction(2, 1), Fraction(3, 2)):
+        with pytest.raises(ValueError, match="max_turns"):
+            SearchAgent(owner_id="alice", max_turns=value)
+    for name in ("request_timeout", "tool_timeout"):
+        with pytest.raises(ValueError, match=name):
+            SearchAgent(owner_id="alice", **{name: True})
+
+
+def test_agent_model_and_provider_fields_reject_all_ascii_controls():
+    for model in ("bad\tmodel", "bad\x1bmodel", "bad\x7fmodel"):
+        with pytest.raises(ValueError, match="model"):
+            SearchAgent(owner_id="alice", model=model)
+    for value in ("bad\tkey", "bad\x1bkey", "bad\x7fkey"):
+        with pytest.raises(ValueError, match="control"):
+            SearchAgent(owner_id="alice", api_key=value)
diff --git a/tools/document_store.py b/tools/document_store.py
index 6ac998a..9531998 100644
--- a/tools/document_store.py
+++ b/tools/document_store.py
@@ -19,7 +19,10 @@ from tools.privacy import mask_metadata_text
 from tools.security import normalize_owner_id
 from tools import document_store_legacy as _implementation
 
-_original_document_store = _implementation.DocumentStore
+if not hasattr(_implementation, "_boundary_original_DocumentStore"):
+    _implementation._boundary_original_DocumentStore = _implementation.DocumentStore
+_original_document_store = _implementation._boundary_original_DocumentStore
+_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
 
 
 def _normalize_registry_environment() -> None:
@@ -41,6 +44,31 @@ def _contains_ascii_control(value: str) -> bool:
     return any(ord(character) < 32 or ord(character) == 127 for character in value)
 
 
+def _is_redirecting(metadata: os.stat_result) -> bool:
+    return stat.S_ISLNK(metadata.st_mode) or bool(
+        int(getattr(metadata, "st_file_attributes", 0))
+        & _WINDOWS_REPARSE_POINT
+    )
+
+
+def _identity(metadata: os.stat_result) -> tuple[int, int]:
+    return int(metadata.st_dev), int(metadata.st_ino)
+
+
+def _path_identity(path: Path, label: str, *, directory: bool) -> tuple[int, int]:
+    try:
+        metadata = path.lstat()
+    except OSError as exc:
+        raise OSError(f"{label} could not be inspected safely.") from exc
+    if _is_redirecting(metadata):
+        raise ValueError(f"{label} may not be a symbolic link or reparse point.")
+    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
+    if not expected:
+        kind = "directory" if directory else "regular file"
+        raise OSError(f"{label} must remain a {kind}.")
+    return _identity(metadata)
+
+
 def _lexical_absolute(value: str | os.PathLike[str], label: str) -> Path:
     if not isinstance(value, (str, os.PathLike)):
         raise ValueError(f"{label} must be a filesystem path.")
@@ -56,8 +84,16 @@ def _lexical_absolute(value: str | os.PathLike[str], label: str) -> Path:
         path = Path.cwd() / path
     absolute = Path(os.path.abspath(path))
     for candidate in (absolute, *absolute.parents):
-        if candidate.is_symlink():
-            raise ValueError(f"{label} may not contain symbolic-link components.")
+        try:
+            metadata = candidate.lstat()
+        except FileNotFoundError:
+            continue
+        except OSError as exc:
+            raise ValueError(f"{label} could not be validated safely.") from exc
+        if _is_redirecting(metadata):
+            raise ValueError(
+                f"{label} may not contain symbolic links or reparse points."
+            )
     return absolute
 
 
@@ -89,7 +125,7 @@ def _mime_type(value: Any) -> str:
     return rendered or "application/octet-stream"
 
 
-class DocumentStore(_original_document_store):
+class _DocumentStoreBoundary(_original_document_store):
     """Registry with bounded budgets, truthful flags, and safe storage roots."""
 
     def __init__(
@@ -106,23 +142,53 @@ class DocumentStore(_original_document_store):
         )
         safe_path = _lexical_absolute(selected_path, "DOCUMENT_DB_PATH")
         safe_root = _lexical_absolute(selected_root, "UPLOAD_DIR")
+        safe_path.parent.mkdir(parents=True, exist_ok=True)
+        safe_root.mkdir(parents=True, exist_ok=True)
+        self._boundary_database_parent_identity = _path_identity(
+            safe_path.parent,
+            "DOCUMENT_DB_PATH parent",
+            directory=True,
+        )
+        self._boundary_upload_root_identity = _path_identity(
+            safe_root,
+            "UPLOAD_DIR",
+            directory=True,
+        )
+        self._boundary_database_identity: tuple[int, int] | None = None
         super().__init__(path=safe_path, upload_root=safe_root)
+        self._boundary_database_identity = _path_identity(
+            self.path,
+            "DOCUMENT_DB_PATH",
+            directory=False,
+        )
         self._ensure_storage_paths()
 
     def _ensure_storage_paths(self) -> None:
-        _lexical_absolute(self.path, "DOCUMENT_DB_PATH")
-        _lexical_absolute(self.upload_root, "UPLOAD_DIR")
-        if not self.path.parent.exists() or not self.path.parent.is_dir():
-            raise OSError("DOCUMENT_DB_PATH parent must remain a directory.")
-        if not self.upload_root.exists() or not self.upload_root.is_dir():
-            raise OSError("UPLOAD_DIR must remain a directory.")
-        if self.path.exists():
-            try:
-                mode = self.path.stat(follow_symlinks=False).st_mode
-            except OSError as exc:
-                raise OSError("DOCUMENT_DB_PATH could not be inspected.") from exc
-            if not stat.S_ISREG(mode):
-                raise OSError("DOCUMENT_DB_PATH must remain a regular file.")
+        safe_path = _lexical_absolute(self.path, "DOCUMENT_DB_PATH")
+        safe_root = _lexical_absolute(self.upload_root, "UPLOAD_DIR")
+        if _path_identity(
+            safe_path.parent,
+            "DOCUMENT_DB_PATH parent",
+            directory=True,
+        ) != self._boundary_database_parent_identity:
+            raise OSError("DOCUMENT_DB_PATH parent identity changed after initialization.")
+        if _path_identity(
+            safe_root,
+            "UPLOAD_DIR",
+            directory=True,
+        ) != self._boundary_upload_root_identity:
+            raise OSError("UPLOAD_DIR identity changed after initialization.")
+        expected_database = self._boundary_database_identity
+        if safe_path.exists():
+            current_database = _path_identity(
+                safe_path,
+                "DOCUMENT_DB_PATH",
+                directory=False,
+            )
+            if expected_database is not None and current_database != expected_database:
+                raise OSError("DOCUMENT_DB_PATH identity changed after initialization.")
+        elif expected_database is not None:
+            raise OSError("DOCUMENT_DB_PATH disappeared after initialization.")
 
     def _connect(self):
         self._ensure_storage_paths()
@@ -191,6 +257,8 @@ class DocumentStore(_original_document_store):
         job_store: Optional[Any] = None,
     ) -> int:
         if now is not None:
+            if isinstance(now, bool):
+                raise ValueError("now must be numeric.")
             try:
                 current = float(now)
             except (TypeError, ValueError, OverflowError) as exc:
@@ -202,6 +270,11 @@ class DocumentStore(_original_document_store):
         return super().cleanup_orphans(now=now, job_store=job_store)
 
 
+if not hasattr(_implementation, "_boundary_public_DocumentStore"):
+    _implementation._boundary_public_DocumentStore = _DocumentStoreBoundary
+DocumentStore = _implementation._boundary_public_DocumentStore
+
+
 def get_document_store(
     path: str | Path | None = None,
     upload_root: str | Path | None = None,
diff --git a/tools/rag.py b/tools/rag.py
index 33acd84..d6d7b33 100644
--- a/tools/rag.py
+++ b/tools/rag.py
@@ -717,6 +717,10 @@ def get_rag_layer(
         return instance
 
 
+if not hasattr(_implementation, "_boundary_public_RAGLayer"):
+    _implementation._boundary_public_RAGLayer = RAGLayer
+RAGLayer = _implementation._boundary_public_RAGLayer
+
 _implementation.RAGLayer = RAGLayer
 _implementation.get_rag_layer = get_rag_layer
 _implementation._RAG_INSTANCES = _RAG_INSTANCES

```
