from tools.evidence_graph_rag_baseline_runtime import (
    clear_graph_rag_baseline_store_cache,
    get_graph_rag_baseline_store,
)


def test_runtime_is_path_scoped(tmp_path):
    clear_graph_rag_baseline_store_cache()
    first = get_graph_rag_baseline_store(tmp_path / "a.sqlite3")
    second = get_graph_rag_baseline_store(tmp_path / "a.sqlite3")
    third = get_graph_rag_baseline_store(tmp_path / "b.sqlite3")
    assert first is second
    assert first is not third
