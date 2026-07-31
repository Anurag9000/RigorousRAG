from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected correction anchor missing in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "tools/hybrid_retrieval.py",
    '''            self.term_counts[document.document_id] = counts\n            self.document_frequency.update(counts)\n''',
    '''            self.term_counts[document.document_id] = counts\n            self.document_frequency.update(counts.keys())\n''',
)

replace_once(
    "tests/unit/test_evaluation_foundation.py",
    '''def test_beir_loader_normalizes_documents_queries_and_qrels(tmp_path):\n    _write_beir(tmp_path)\n    dataset = load_beir_dataset(tmp_path)\n    assert dataset.documents["d1"].title == "Alpha"\n    assert dataset.queries[0].relevant == {"d1": 2.0}\n\n    link = tmp_path / "linked"\n    try:\n        link.symlink_to(tmp_path / "corpus.jsonl")\n    except OSError:\n        pytest.skip("Symbolic links unavailable")\n    with pytest.raises(ValueError, match="links"):\n        load_beir_dataset(link.parent / "linked")\n''',
    '''def test_beir_loader_normalizes_documents_queries_and_qrels(tmp_path):\n    dataset_root = tmp_path / "dataset"\n    _write_beir(dataset_root)\n    dataset = load_beir_dataset(dataset_root)\n    assert dataset.documents["d1"].title == "Alpha"\n    assert dataset.queries[0].relevant == {"d1": 2.0}\n\n    link = tmp_path / "linked-dataset"\n    try:\n        link.symlink_to(dataset_root, target_is_directory=True)\n    except OSError:\n        pytest.skip("Symbolic links unavailable")\n    with pytest.raises(ValueError, match="links"):\n        load_beir_dataset(link)\n''',
)

replace_once(
    "scripts/run_retrieval_benchmarks.py",
    '''            "top_k": args.top_k,\n            "candidate_pool": args.top_k,\n''',
    '''            "top_k": args.top_k,\n            "candidate_pool": [max(args.top_k)],\n''',
)

replace_once(
    "tools/rag_tool.py",
    '''        metadata_owner = metadata.get("owner_id")\n        source_id = metadata.get("doc_id")\n        if metadata_owner != owner or not isinstance(source_id, str):\n            continue\n''',
    '''        try:\n            metadata_owner = metadata.get("owner_id")\n            source_id = metadata.get("doc_id")\n        except Exception:\n            continue\n        if metadata_owner != owner or not isinstance(source_id, str):\n            continue\n''',
)
