"""Correct pass-eleven destructive-operation identity checks after diagnostics."""

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected correction anchor missing in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "tools/logger.py",
    '''def _same_identity(left: Any, right: Any) -> bool:\n    return _identity(left) == _identity(right)\n\n\ndef _absolute_without_resolving''',
    '''def _same_identity(left: Any, right: Any) -> bool:\n    return _identity(left) == _identity(right)\n\n\ndef _snapshot_identity(metadata: Any) -> tuple[int, int, int, int, int, int]:\n    ctime_ns = getattr(metadata, "st_ctime_ns", None)\n    if ctime_ns is None:\n        ctime_ns = int(float(getattr(metadata, "st_ctime", 0.0)) * 1_000_000_000)\n    mtime_ns = getattr(metadata, "st_mtime_ns", None)\n    if mtime_ns is None:\n        mtime_ns = int(float(getattr(metadata, "st_mtime", 0.0)) * 1_000_000_000)\n    return (\n        int(metadata.st_dev),\n        int(metadata.st_ino),\n        int(ctime_ns),\n        int(mtime_ns),\n        int(getattr(metadata, "st_size", -1)),\n        int(metadata.st_mode),\n    )\n\n\ndef _same_snapshot(left: Any, right: Any) -> bool:\n    return _snapshot_identity(left) == _snapshot_identity(right)\n\n\ndef _absolute_without_resolving''',
)
replace_once(
    "tools/logger.py",
    '''    if expected is not None and not _same_identity(current, expected):\n        raise OSError("Telemetry unlink refused a replaced path.")\n''',
    '''    if expected is not None and not _same_snapshot(current, expected):\n        raise OSError("Telemetry unlink refused a replaced path.")\n''',
)
replace_once(
    "tools/logger.py",
    '''        or not _same_identity(current_source, expected_source)\n    ):\n        raise OSError("Telemetry rotation source changed before replacement.")\n''',
    '''        or not _same_snapshot(current_source, expected_source)\n    ):\n        raise OSError("Telemetry rotation source changed before replacement.")\n''',
)
replace_once(
    "tools/logger.py",
    '''        or not _same_identity(current_destination, expected_destination)\n    ):\n        raise OSError("Telemetry rotation destination changed before replacement.")\n''',
    '''        or not _same_snapshot(current_destination, expected_destination)\n    ):\n        raise OSError("Telemetry rotation destination changed before replacement.")\n''',
)
