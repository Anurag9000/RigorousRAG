# Pass eleven diagnostic

```text
.............F........                                                   [100%]
=================================== FAILURES ===================================
________________ test_rotation_replace_refuses_replaced_source _________________

tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-0/test_rotation_replace_refuses_0')

    def test_rotation_replace_refuses_replaced_source(tmp_path):
        source = tmp_path / "metrics.jsonl"
        destination = tmp_path / "metrics.jsonl.1"
        source.write_bytes(b"original")
        expected = source.lstat()
        source.unlink()
        source.write_bytes(b"replacement")
    
>       with pytest.raises(OSError, match="source changed"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE OSError

tests/unit/test_logger_boundaries.py:270: Failed
=========================== short test summary info ============================
FAILED tests/unit/test_logger_boundaries.py::test_rotation_replace_refuses_replaced_source - Failed: DID NOT RAISE OSError
1 failed, 21 passed in 17.41s

```

```diff
diff --git a/tests/unit/test_logger_boundaries.py b/tests/unit/test_logger_boundaries.py
index 88cdd17..df6eaa0 100644
--- a/tests/unit/test_logger_boundaries.py
+++ b/tests/unit/test_logger_boundaries.py
@@ -1,5 +1,9 @@
 import json
 import os
+import stat
+from decimal import Decimal
+from fractions import Fraction
+from types import SimpleNamespace
 
 import pytest
 
@@ -179,3 +183,98 @@ def test_malformed_telemetry_integer_environment_uses_default(monkeypatch):
     )
 
     assert value == 17
+
+def test_telemetry_numeric_helpers_require_exact_integer_semantics():
+    class ExactIndex:
+        def __index__(self):
+            return 7
+
+    assert logger._nonnegative_integer(ExactIndex()) == 7
+    assert logger._nonnegative_integer(Decimal("1.5")) == 0
+    assert logger._nonnegative_integer(Fraction(3, 2)) == 0
+    assert logger._nonnegative_integer(True) == 0
+    assert logger._finite_nonnegative(True) == 0.0
+
+
+def test_reparse_metadata_is_never_treated_as_regular(monkeypatch, tmp_path):
+    path = tmp_path / "metrics.jsonl"
+    metadata = SimpleNamespace(
+        st_mode=stat.S_IFREG | 0o600,
+        st_file_attributes=logger._WINDOWS_REPARSE_POINT,
+        st_dev=1,
+        st_ino=2,
+    )
+    monkeypatch.setattr(logger, "_member_stat", lambda *_args: metadata)
+
+    assert logger._regular_or_missing(path) is False
+
+
+def test_append_refuses_visible_path_identity_change_before_write(
+    monkeypatch, tmp_path
+):
+    path = tmp_path / "metrics.jsonl"
+    path.write_bytes(b"before")
+    before = path.lstat()
+    changed = SimpleNamespace(
+        st_mode=before.st_mode,
+        st_file_attributes=0,
+        st_dev=before.st_dev,
+        st_ino=before.st_ino + 1,
+    )
+    observed = iter((before, changed))
+    monkeypatch.setattr(logger, "_member_stat", lambda *_args: next(observed))
+
+    with pytest.raises(OSError, match="identity"):
+        logger._append_line(path, b"after\n")
+
+    assert path.read_bytes() == b"before"
+
+
+def test_process_lock_refuses_visible_identity_change_after_open(
+    monkeypatch, tmp_path
+):
+    destination = tmp_path / "metrics.jsonl"
+    lock_path = tmp_path / ".metrics.jsonl.lock"
+    real_member_stat = logger._member_stat
+    calls = 0
+
+    def changed_lock(path, parent_fd):
+        nonlocal calls
+        calls += 1
+        if calls == 1:
+            return None
+        actual = real_member_stat(path, parent_fd)
+        assert actual is not None
+        return SimpleNamespace(
+            st_mode=actual.st_mode,
+            st_file_attributes=0,
+            st_dev=actual.st_dev,
+            st_ino=actual.st_ino + 1,
+        )
+
+    monkeypatch.setattr(logger, "_member_stat", changed_lock)
+    with pytest.raises(OSError, match="identity"):
+        with logger._process_log_lock(destination):
+            raise AssertionError("lock body must not execute")
+    assert lock_path.exists()
+
+
+def test_rotation_replace_refuses_replaced_source(tmp_path):
+    source = tmp_path / "metrics.jsonl"
+    destination = tmp_path / "metrics.jsonl.1"
+    source.write_bytes(b"original")
+    expected = source.lstat()
+    source.unlink()
+    source.write_bytes(b"replacement")
+
+    with pytest.raises(OSError, match="source changed"):
+        logger._replace_member(
+            source,
+            destination,
+            None,
+            expected_source=expected,
+            expected_destination=None,
+        )
+
+    assert source.read_bytes() == b"replacement"
+    assert not destination.exists()
diff --git a/tests/unit/test_logger_directory_boundary.py b/tests/unit/test_logger_directory_boundary.py
index a1cc835..62747bf 100644
--- a/tests/unit/test_logger_directory_boundary.py
+++ b/tests/unit/test_logger_directory_boundary.py
@@ -1,5 +1,6 @@
 import json
 from contextlib import contextmanager
+from types import SimpleNamespace
 
 import pytest
 
@@ -90,3 +91,24 @@ def test_parent_swap_does_not_redirect_lock_or_append(monkeypatch, tmp_path):
     assert list(outside.iterdir()) == []
     assert (moved / "usage.jsonl").is_file()
     assert json.loads((moved / "usage.jsonl").read_text(encoding="utf-8"))["type"] == "event"
+
+def test_reparse_parent_component_is_detected_without_resolution(
+    monkeypatch, tmp_path
+):
+    parent = tmp_path / "state"
+    parent.mkdir()
+    original_lstat = type(parent).lstat
+
+    def reparse_lstat(self):
+        metadata = original_lstat(self)
+        if self == parent:
+            return SimpleNamespace(
+                st_mode=metadata.st_mode,
+                st_file_attributes=logger._WINDOWS_REPARSE_POINT,
+                st_dev=metadata.st_dev,
+                st_ino=metadata.st_ino,
+            )
+        return metadata
+
+    monkeypatch.setattr(type(parent), "lstat", reparse_lstat)
+    assert logger._has_symlink_component(parent) is True
diff --git a/tools/logger.py b/tools/logger.py
index 32c717e..19aaf9f 100644
--- a/tools/logger.py
+++ b/tools/logger.py
@@ -5,6 +5,7 @@ from __future__ import annotations
 import hashlib
 import json
 import math
+import operator
 import os
 import stat
 import threading
@@ -42,6 +43,7 @@ _MAX_PATH_CHARS = 4096
 _MAX_PRIVATE_HASH_INPUT_CHARS = 100_000
 _MAX_PUBLIC_INTEGER = 1_000_000_000
 _LOG_LOCK = threading.Lock()
+_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
 
 
 def _safe_text(value: Any, *, maximum: int, default: str = "") -> str:
@@ -98,6 +100,8 @@ def _json_safe(value: Any, *, depth: int = 0) -> Any:
 
 
 def _finite_nonnegative(value: Any, *, digits: int = 3) -> float:
+    if isinstance(value, bool):
+        return 0.0
     try:
         numeric = float(value)
     except (TypeError, ValueError, OverflowError):
@@ -111,7 +115,7 @@ def _nonnegative_integer(value: Any) -> int:
     if isinstance(value, bool):
         return 0
     try:
-        numeric = int(value)
+        numeric = int(operator.index(value))
     except (TypeError, ValueError, OverflowError):
         return 0
     return max(0, min(numeric, _MAX_PUBLIC_INTEGER))
@@ -121,6 +125,20 @@ def _contains_ascii_control(value: str) -> bool:
     return any(ord(character) < 32 or ord(character) == 127 for character in value)
 
 
+def _is_redirecting(metadata: Any) -> bool:
+    return stat.S_ISLNK(metadata.st_mode) or bool(
+        int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT
+    )
+
+
+def _is_regular_nonredirecting(metadata: Any) -> bool:
+    return stat.S_ISREG(metadata.st_mode) and not _is_redirecting(metadata)
+
+
+def _same_identity(left: Any, right: Any) -> bool:
+    return _identity(left) == _identity(right)
+
+
 def _absolute_without_resolving(path: Any) -> Path:
     if not isinstance(path, (str, os.PathLike)):
         raise ValueError("Telemetry path must be a filesystem path.")
@@ -140,10 +158,14 @@ def _absolute_without_resolving(path: Any) -> Path:
 def _has_symlink_component(path: Path) -> bool:
     try:
         candidate = _absolute_without_resolving(path)
-        return any(
-            component.is_symlink()
-            for component in (candidate, *candidate.parents)
-        )
+        for component in (candidate, *candidate.parents):
+            try:
+                metadata = component.lstat()
+            except FileNotFoundError:
+                continue
+            if _is_redirecting(metadata):
+                return True
+        return False
     except (OSError, ValueError):
         return True
 
@@ -172,8 +194,8 @@ def _log_directory(path: Path) -> Iterator[tuple[Path, Optional[int]]]:
     if _has_symlink_component(parent):
         raise OSError("Telemetry parent path is unsafe.")
     before = os.stat(parent, follow_symlinks=False)
-    if not stat.S_ISDIR(before.st_mode):
-        raise OSError("Telemetry parent must be a directory.")
+    if not stat.S_ISDIR(before.st_mode) or _is_redirecting(before):
+        raise OSError("Telemetry parent must be a non-redirecting directory.")
 
     if os.name == "nt":  # pragma: no cover - Windows-specific fallback
         try:
@@ -194,14 +216,18 @@ def _log_directory(path: Path) -> Iterator[tuple[Path, Optional[int]]]:
     descriptor = os.open(parent, flags)
     try:
         opened = os.fstat(descriptor)
-        if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != _identity(before):
+        if (
+            not stat.S_ISDIR(opened.st_mode)
+            or _is_redirecting(opened)
+            or not _same_identity(opened, before)
+        ):
             raise OSError("Telemetry parent descriptor identity is invalid.")
         current = os.stat(parent, follow_symlinks=False)
-        if _identity(current) != _identity(opened):
+        if _is_redirecting(current) or not _same_identity(current, opened):
             raise OSError("Telemetry parent changed before publication.")
         yield destination, descriptor
         current = os.stat(parent, follow_symlinks=False)
-        if _identity(current) != _identity(opened):
+        if _is_redirecting(current) or not _same_identity(current, opened):
             raise OSError("Telemetry parent changed during publication.")
     finally:
         os.close(descriptor)
@@ -221,37 +247,74 @@ def _regular_or_missing(path: Path, parent_fd: Optional[int] = None) -> bool:
         metadata = _member_stat(path, parent_fd)
     except OSError:
         return False
-    return metadata is None or stat.S_ISREG(metadata.st_mode)
+    return metadata is None or _is_regular_nonredirecting(metadata)
 
 
-def _unlink_member(path: Path, parent_fd: Optional[int]) -> None:
+def _unlink_member(
+    path: Path,
+    parent_fd: Optional[int],
+    *,
+    expected: Optional[os.stat_result] = None,
+) -> None:
+    current = _member_stat(path, parent_fd)
+    if current is None:
+        return
+    if not _is_regular_nonredirecting(current):
+        raise OSError("Telemetry unlink refused a redirected or non-regular path.")
+    if expected is not None and not _same_identity(current, expected):
+        raise OSError("Telemetry unlink refused a replaced path.")
     if parent_fd is None:
-        path.unlink(missing_ok=True)
+        path.unlink()
         return
-    try:
-        os.unlink(path.name, dir_fd=parent_fd)
-    except FileNotFoundError:
-        pass
+    os.unlink(path.name, dir_fd=parent_fd)
 
 
-def _replace_member(source: Path, destination: Path, parent_fd: Optional[int]) -> None:
+def _replace_member(
+    source: Path,
+    destination: Path,
+    parent_fd: Optional[int],
+    *,
+    expected_source: os.stat_result,
+    expected_destination: Optional[os.stat_result],
+) -> None:
+    current_source = _member_stat(source, parent_fd)
+    current_destination = _member_stat(destination, parent_fd)
+    if (
+        current_source is None
+        or not _is_regular_nonredirecting(current_source)
+        or not _same_identity(current_source, expected_source)
+    ):
+        raise OSError("Telemetry rotation source changed before replacement.")
+    if expected_destination is None:
+        if current_destination is not None:
+            raise OSError("Telemetry rotation destination appeared unexpectedly.")
+    elif (
+        current_destination is None
+        or not _is_regular_nonredirecting(current_destination)
+        or not _same_identity(current_destination, expected_destination)
+    ):
+        raise OSError("Telemetry rotation destination changed before replacement.")
     if parent_fd is None:
         source.replace(destination)
-        return
-    os.replace(
-        source.name,
-        destination.name,
-        src_dir_fd=parent_fd,
-        dst_dir_fd=parent_fd,
-    )
+    else:
+        os.replace(
+            source.name,
+            destination.name,
+            src_dir_fd=parent_fd,
+            dst_dir_fd=parent_fd,
+        )
+    published = _member_stat(destination, parent_fd)
+    if published is None or not _same_identity(published, expected_source):
+        raise OSError("Telemetry rotation publication identity is invalid.")
 
 
 @contextmanager
 def _process_log_lock(path: Path, parent_fd: Optional[int] = None) -> Iterator[None]:
-    """Serialize publication and rotation across service processes."""
+    """Serialize publication and rotation across one identity-stable lock file."""
 
     lock_path = path.with_name(f".{path.name}.lock")
-    if not _regular_or_missing(lock_path, parent_fd):
+    before = _member_stat(lock_path, parent_fd)
+    if before is not None and not _is_regular_nonredirecting(before):
         raise OSError("Telemetry lock path is unsafe.")
     flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
     if parent_fd is None:
@@ -259,22 +322,42 @@ def _process_log_lock(path: Path, parent_fd: Optional[int] = None) -> Iterator[N
     else:
         descriptor = os.open(lock_path.name, flags, 0o600, dir_fd=parent_fd)
     try:
-        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
-            raise OSError("Telemetry lock path must be a regular file.")
+        opened = os.fstat(descriptor)
+        current = _member_stat(lock_path, parent_fd)
+        if (
+            not _is_regular_nonredirecting(opened)
+            or current is None
+            or not _is_regular_nonredirecting(current)
+            or not _same_identity(current, opened)
+            or (before is not None and not _same_identity(before, opened))
+        ):
+            raise OSError("Telemetry lock path identity is invalid.")
         try:
             os.fchmod(descriptor, 0o600)
         except OSError:
             pass
+
+        def verify_visible_lock() -> None:
+            visible = _member_stat(lock_path, parent_fd)
+            if (
+                visible is None
+                or not _is_regular_nonredirecting(visible)
+                or not _same_identity(visible, opened)
+            ):
+                raise OSError("Telemetry lock path changed during publication.")
+
         if os.name == "nt":  # pragma: no cover
             import msvcrt
 
-            if os.fstat(descriptor).st_size < 1:
+            if opened.st_size < 1:
                 os.write(descriptor, b"0")
                 os.fsync(descriptor)
             os.lseek(descriptor, 0, os.SEEK_SET)
             msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
             try:
+                verify_visible_lock()
                 yield
+                verify_visible_lock()
             finally:
                 os.lseek(descriptor, 0, os.SEEK_SET)
                 msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
@@ -283,7 +366,9 @@ def _process_log_lock(path: Path, parent_fd: Optional[int] = None) -> Iterator[N
 
             fcntl.flock(descriptor, fcntl.LOCK_EX)
             try:
+                verify_visible_lock()
                 yield
+                verify_visible_lock()
             finally:
                 fcntl.flock(descriptor, fcntl.LOCK_UN)
     finally:
@@ -294,18 +379,18 @@ def _rotate(path: Path, parent_fd: Optional[int] = None) -> None:
     metadata = _member_stat(path, parent_fd)
     if metadata is None:
         return
-    if not stat.S_ISREG(metadata.st_mode):
-        raise OSError("Telemetry rotation refused a non-regular path.")
+    if not _is_regular_nonredirecting(metadata):
+        raise OSError("Telemetry rotation refused a redirected or non-regular path.")
     if LOG_BACKUPS <= 0:
-        _unlink_member(path, parent_fd)
+        _unlink_member(path, parent_fd, expected=metadata)
         return
 
     oldest = _rotated_path(path, LOG_BACKUPS)
     oldest_metadata = _member_stat(oldest, parent_fd)
     if oldest_metadata is not None:
-        if not stat.S_ISREG(oldest_metadata.st_mode):
-            raise OSError("Telemetry backup path is not a regular file.")
-        _unlink_member(oldest, parent_fd)
+        if not _is_regular_nonredirecting(oldest_metadata):
+            raise OSError("Telemetry backup path is not a safe regular file.")
+        _unlink_member(oldest, parent_fd, expected=oldest_metadata)
 
     for index in range(LOG_BACKUPS - 1, 0, -1):
         source = _rotated_path(path, index)
@@ -313,19 +398,41 @@ def _rotate(path: Path, parent_fd: Optional[int] = None) -> None:
         source_metadata = _member_stat(source, parent_fd)
         if source_metadata is None:
             continue
-        if not stat.S_ISREG(source_metadata.st_mode):
+        if not _is_regular_nonredirecting(source_metadata):
             raise OSError("Telemetry backup rotation encountered an unsafe path.")
         destination_metadata = _member_stat(destination, parent_fd)
-        if destination_metadata is not None and not stat.S_ISREG(destination_metadata.st_mode):
+        if (
+            destination_metadata is not None
+            and not _is_regular_nonredirecting(destination_metadata)
+        ):
             raise OSError("Telemetry backup rotation encountered an unsafe path.")
-        _replace_member(source, destination, parent_fd)
+        _replace_member(
+            source,
+            destination,
+            parent_fd,
+            expected_source=source_metadata,
+            expected_destination=destination_metadata,
+        )
 
-    _replace_member(path, _rotated_path(path, 1), parent_fd)
+    first_backup = _rotated_path(path, 1)
+    first_metadata = _member_stat(first_backup, parent_fd)
+    if first_metadata is not None and not _is_regular_nonredirecting(first_metadata):
+        raise OSError("Telemetry first backup path is unsafe.")
+    _replace_member(
+        path,
+        first_backup,
+        parent_fd,
+        expected_source=metadata,
+        expected_destination=first_metadata,
+    )
 
 
 def _append_line(path: Path, encoded: bytes, parent_fd: Optional[int] = None) -> None:
     if not encoded or len(encoded) > _MAX_EVENT_BYTES:
         raise OSError("Telemetry event exceeds the append limit.")
+    before = _member_stat(path, parent_fd)
+    if before is not None and not _is_regular_nonredirecting(before):
+        raise OSError("Telemetry destination path is unsafe.")
     flags = (
         os.O_WRONLY
         | os.O_CREAT
@@ -338,9 +445,16 @@ def _append_line(path: Path, encoded: bytes, parent_fd: Optional[int] = None) ->
     else:
         descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
     try:
-        metadata = os.fstat(descriptor)
-        if not stat.S_ISREG(metadata.st_mode):
-            raise OSError("Telemetry destination must be a regular file.")
+        opened = os.fstat(descriptor)
+        visible = _member_stat(path, parent_fd)
+        if (
+            not _is_regular_nonredirecting(opened)
+            or visible is None
+            or not _is_regular_nonredirecting(visible)
+            or not _same_identity(visible, opened)
+            or (before is not None and not _same_identity(before, opened))
+        ):
+            raise OSError("Telemetry destination identity is invalid.")
         try:
             os.fchmod(descriptor, 0o600)
         except OSError:
@@ -355,6 +469,13 @@ def _append_line(path: Path, encoded: bytes, parent_fd: Optional[int] = None) ->
             os.fsync(descriptor)
         except OSError:
             pass
+        visible = _member_stat(path, parent_fd)
+        if (
+            visible is None
+            or not _is_regular_nonredirecting(visible)
+            or not _same_identity(visible, opened)
+        ):
+            raise OSError("Telemetry destination changed during append.")
     finally:
         os.close(descriptor)
 
@@ -406,7 +527,7 @@ def log_activity(activity_type: str, details: Dict[str, Any]) -> None:
         with _LOG_LOCK, _log_directory(path) as (destination, parent_fd):
             with _process_log_lock(destination, parent_fd):
                 metadata = _member_stat(destination, parent_fd)
-                if metadata is not None and not stat.S_ISREG(metadata.st_mode):
+                if metadata is not None and not _is_regular_nonredirecting(metadata):
                     return
                 current_size = metadata.st_size if metadata is not None else 0
                 if current_size + len(encoded) > LOG_MAX_BYTES:

```
