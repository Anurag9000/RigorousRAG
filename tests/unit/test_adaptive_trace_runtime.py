import pytest

from tools.adaptive_trace_runtime import (
    clear_adaptive_trace_store_cache,
    get_adaptive_trace_store,
)


def test_trace_runtime_is_disabled_without_configuration(monkeypatch):
    clear_adaptive_trace_store_cache()
    monkeypatch.delenv("ADAPTIVE_TRACE_DB_PATH", raising=False)
    assert get_adaptive_trace_store() is None


def test_trace_runtime_is_path_keyed_and_reload_safe(monkeypatch, tmp_path):
    clear_adaptive_trace_store_cache()
    first_path = tmp_path / "first.sqlite3"
    second_path = tmp_path / "second.sqlite3"
    monkeypatch.setenv("ADAPTIVE_TRACE_DB_PATH", str(first_path))
    first = get_adaptive_trace_store()
    assert first is get_adaptive_trace_store()
    second = get_adaptive_trace_store(second_path)
    assert second is get_adaptive_trace_store(second_path)
    assert second is not first
    clear_adaptive_trace_store_cache()
    assert get_adaptive_trace_store() is not first


def test_trace_runtime_rejects_padded_configuration(monkeypatch):
    clear_adaptive_trace_store_cache()
    monkeypatch.setenv("ADAPTIVE_TRACE_DB_PATH", " padded.sqlite3 ")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        get_adaptive_trace_store()
