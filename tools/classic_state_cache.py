import os
import threading

from Searching import AcademicSearchEngine
from tools.internal_search_storage import _storage_signature

_ENGINE_INSTANCE = None
_ENGINE_SIGNATURE = None
_ENGINE_LOCK = threading.Lock()
_DIGEST_CACHE: dict[object, str] = {}


def _close(engine):
    if engine is not None:
        try:
            engine.close()
        except Exception:
            pass


def get_engine():
    global _ENGINE_INSTANCE, _ENGINE_SIGNATURE
    with _ENGINE_LOCK:
        signature = _storage_signature(os.getenv("CLASSIC_STORAGE_DIR", "data"))
        if _ENGINE_INSTANCE is not None and _ENGINE_SIGNATURE == signature:
            return _ENGINE_INSTANCE
        for _ in range(3):
            replacement = AcademicSearchEngine(storage_dir=signature[0])
            after = _storage_signature(signature[0])
            if after != signature:
                _close(replacement)
                signature = after
                continue
            previous = _ENGINE_INSTANCE
            _ENGINE_INSTANCE, _ENGINE_SIGNATURE = replacement, after
            if previous is not replacement:
                _close(previous)
            return replacement
        raise RuntimeError("Classic search state changed repeatedly during engine initialization.")
