import os
import threading

from Searching import AcademicSearchEngine
from tools.internal_search_storage import _storage_signature

_INSTANCE = None
_SIGNATURE = None
_LOCK = threading.Lock()


def current():
    global _INSTANCE, _SIGNATURE
    with _LOCK:
        signature = _storage_signature(os.getenv("CLASSIC_STORAGE_DIR", "data"))
        if _INSTANCE is not None and _SIGNATURE == signature:
            return _INSTANCE
        replacement = AcademicSearchEngine(storage_dir=signature[0])
        after = _storage_signature(signature[0])
        if after != signature:
            try:
                replacement.close()
            except Exception:
                pass
            replacement = AcademicSearchEngine(storage_dir=after[0])
            after = _storage_signature(after[0])
        previous = _INSTANCE
        _INSTANCE, _SIGNATURE = replacement, after
        if previous is not None and previous is not replacement:
            try:
                previous.close()
            except Exception:
                pass
        return replacement
