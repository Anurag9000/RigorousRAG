from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected patch anchor missing in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + addition.strip() + "\n", encoding="utf-8")


# Keep the legacy scientific implementation's original functions stable across reimports.
replace_once(
    "tools/integrity_boundary.py",
    '''_original_compare_papers = _implementation.compare_papers\n_original_generate_comparison_matrix = _implementation.generate_comparison_matrix\n''',
    '''if not hasattr(_implementation, "_integrity_boundary_original_compare_papers"):\n    _implementation._integrity_boundary_original_compare_papers = (\n        _implementation.compare_papers\n    )\nif not hasattr(\n    _implementation,\n    "_integrity_boundary_original_generate_comparison_matrix",\n):\n    _implementation._integrity_boundary_original_generate_comparison_matrix = (\n        _implementation.generate_comparison_matrix\n    )\n\n_original_compare_papers = (\n    _implementation._integrity_boundary_original_compare_papers\n)\n_original_generate_comparison_matrix = (\n    _implementation._integrity_boundary_original_generate_comparison_matrix\n)\n''',
)
replace_once(
    "tools/integrity_boundary.py",
    '''        image_b64, page_number, caption_text = _implementation._extract_figure_region(\n            source_bytes,\n            figure,\n        )\n''',
    '''        image_b64, page_number, caption_text = _extract_figure_region(\n            source_bytes,\n            figure,\n        )\n''',
)

# Keep the strict final integrity layer anchored to the first boundary implementation.
replace_once(
    "tools/integrity.py",
    '''_MAX_SCIENTIFIC_JSON_CHARS = 100_000\n_original_extract_figure_region = _implementation._extract_figure_region\n_original_check_visual_entailment = _implementation.check_visual_entailment\n_original_compare_papers = _implementation.compare_papers\n_original_generate_comparison_matrix = _implementation.generate_comparison_matrix\n_original_extract_protocol = _implementation.extract_protocol\n_original_run_scientific_debate = _implementation.run_scientific_debate\n_original_detect_conflicts = _implementation.detect_conflicts\n_original_extract_limitations = _implementation.extract_limitations\n''',
    '''_MAX_SCIENTIFIC_JSON_CHARS = 100_000\n\n\ndef _persisted_original(name: str, value: Any) -> Any:\n    if not hasattr(_implementation, name):\n        setattr(_implementation, name, value)\n    return getattr(_implementation, name)\n\n\n_original_extract_figure_region = _persisted_original(\n    "_integrity_final_original_extract_figure_region",\n    _implementation._extract_figure_region,\n)\n_original_check_visual_entailment = _persisted_original(\n    "_integrity_final_original_check_visual_entailment",\n    _implementation.check_visual_entailment,\n)\n_original_compare_papers = _persisted_original(\n    "_integrity_final_original_compare_papers",\n    _implementation.compare_papers,\n)\n_original_generate_comparison_matrix = _persisted_original(\n    "_integrity_final_original_generate_comparison_matrix",\n    _implementation.generate_comparison_matrix,\n)\n_original_extract_protocol = _persisted_original(\n    "_integrity_final_original_extract_protocol",\n    _implementation.extract_protocol,\n)\n_original_run_scientific_debate = _persisted_original(\n    "_integrity_final_original_run_scientific_debate",\n    _implementation.run_scientific_debate,\n)\n_original_detect_conflicts = _persisted_original(\n    "_integrity_final_original_detect_conflicts",\n    _implementation.detect_conflicts,\n)\n_original_extract_limitations = _persisted_original(\n    "_integrity_final_original_extract_limitations",\n    _implementation.extract_limitations,\n)\n''',
)

# Make RAG storage checks reparse-aware and integer parsing exact.
replace_once(
    "tools/rag.py",
    '''import itertools\nimport json\nimport math\nimport os\nimport sys\n''',
    '''import itertools\nimport json\nimport math\nimport operator\nimport os\nimport stat\nimport sys\n''',
)
replace_once(
    "tools/rag.py",
    '''_raw_chroma_path = Path(os.getenv("CHROMA_PATH", "rag_storage"))\nif _raw_chroma_path.is_symlink():\n    raise ValueError("CHROMA_PATH may not be a symbolic link.")\n\nfrom tools import rag_legacy as _implementation\n''',
    '''_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)\n\n\ndef _is_redirecting_path(metadata: Any) -> bool:\n    return stat.S_ISLNK(metadata.st_mode) or bool(\n        int(getattr(metadata, "st_file_attributes", 0))\n        & _WINDOWS_REPARSE_POINT\n    )\n\n\ndef _reject_redirecting_components(path: Path) -> None:\n    for candidate in (path, *path.parents):\n        try:\n            metadata = candidate.lstat()\n        except FileNotFoundError:\n            continue\n        except OSError as exc:\n            raise ValueError("CHROMA_PATH could not be validated.") from exc\n        if _is_redirecting_path(metadata):\n            raise ValueError(\n                "CHROMA_PATH may not contain symbolic links or reparse points."\n            )\n\n\n_raw_chroma_path = Path(os.getenv("CHROMA_PATH", "rag_storage"))\nif not _raw_chroma_path.is_absolute():\n    _raw_chroma_path = Path.cwd() / _raw_chroma_path\n_reject_redirecting_components(Path(os.path.abspath(_raw_chroma_path)))\n\nfrom tools import rag_legacy as _implementation\n''',
)
replace_once(
    "tools/rag.py",
    '''def _bounded_integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:\n    if isinstance(value, bool):\n        raise ValueError(f"{label} must be an integer.")\n    try:\n        numeric = int(value)\n    except (TypeError, ValueError, OverflowError) as exc:\n        raise ValueError(f"{label} must be an integer.") from exc\n    if isinstance(value, float) and not value.is_integer():\n        raise ValueError(f"{label} must be an integer.")\n    if not minimum <= numeric <= maximum:\n        raise ValueError(f"{label} must be between {minimum} and {maximum}.")\n    return numeric\n''',
    '''def _bounded_integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:\n    if isinstance(value, bool):\n        raise ValueError(f"{label} must be an integer.")\n    try:\n        numeric = int(operator.index(value))\n    except (TypeError, ValueError, OverflowError) as exc:\n        raise ValueError(f"{label} must be an integer.") from exc\n    if not minimum <= numeric <= maximum:\n        raise ValueError(f"{label} must be between {minimum} and {maximum}.")\n    return numeric\n''',
)
replace_once(
    "tools/rag.py",
    '''    absolute = Path(os.path.abspath(raw))\n    for candidate in (absolute, *absolute.parents):\n        try:\n            if candidate.is_symlink():\n                raise ValueError(\n                    "CHROMA_PATH may not contain symbolic-link components."\n                )\n        except OSError as exc:\n            raise ValueError("CHROMA_PATH could not be validated.") from exc\n    return str(absolute)\n''',
    '''    absolute = Path(os.path.abspath(raw))\n    _reject_redirecting_components(absolute)\n    return str(absolute)\n''',
)
replace_once(
    "tools/rag.py",
    '''class RAGLayer(_implementation.RAGLayer):\n''',
    '''if not hasattr(_implementation, "_boundary_original_RAGLayer"):\n    _implementation._boundary_original_RAGLayer = _implementation.RAGLayer\n\n_BaseRAGLayer = _implementation._boundary_original_RAGLayer\n\n\nclass RAGLayer(_BaseRAGLayer):\n''',
)
replace_once(
    "tools/rag.py",
    '''_RAG_INSTANCES: Dict[str, RAGLayer] = {}\n_RAG_LOCK = _implementation.threading.Lock()\n''',
    '''if not hasattr(_implementation, "_boundary_rag_instances"):\n    _implementation._boundary_rag_instances = {}\nif not hasattr(_implementation, "_boundary_rag_lock"):\n    _implementation._boundary_rag_lock = _implementation.threading.Lock()\n\n_RAG_INSTANCES: Dict[str, RAGLayer] = _implementation._boundary_rag_instances\n_RAG_LOCK = _implementation._boundary_rag_lock\n''',
)

append_once(
    "tests/unit/test_rag_public_boundaries.py",
    "test_rag_integer_limits_require_the_index_protocol",
    '''def test_rag_integer_limits_require_the_index_protocol():\n    from decimal import Decimal\n    from fractions import Fraction\n\n    class ExactIndex:\n        def __index__(self):\n            return 3\n\n    layer = _layer()\n    assert layer.query("question", owner_id="alice", n_results=ExactIndex()) == []\n    for value in (1.0, Decimal("1"), Fraction(1, 1), Fraction(3, 2)):\n        with pytest.raises(ValueError, match="n_results"):\n            layer.query("question", owner_id="alice", n_results=value)\n\n\ndef test_reparse_chroma_component_is_rejected(monkeypatch, tmp_path):\n    from types import SimpleNamespace\n\n    root = tmp_path / "vectors"\n    root.mkdir()\n    original_lstat = type(root).lstat\n\n    def reparse_lstat(self):\n        metadata = original_lstat(self)\n        if self == root:\n            return SimpleNamespace(\n                st_mode=metadata.st_mode,\n                st_file_attributes=rag_module._WINDOWS_REPARSE_POINT,\n                st_dev=metadata.st_dev,\n                st_ino=metadata.st_ino,\n            )\n        return metadata\n\n    monkeypatch.setattr(type(root), "lstat", reparse_lstat)\n    with pytest.raises(ValueError, match="reparse points"):\n        RAGLayer(persist_directory=str(root))\n''',
)

Path("tests/unit/test_compatibility_reload_boundaries.py").write_text(
    '''import subprocess\nimport sys\nfrom pathlib import Path\n\n\nROOT = Path(__file__).resolve().parents[2]\n\n\ndef _run(script: str):\n    return subprocess.run(\n        [sys.executable, "-c", script],\n        cwd=ROOT,\n        capture_output=True,\n        text=True,\n        timeout=60,\n        check=False,\n    )\n\n\ndef test_integrity_layers_preserve_original_call_chains_across_reimports():\n    result = _run(\n        r"""\nimport importlib\nimport sys\nimport tools\n\nlegacy = importlib.import_module("tools.integrity_legacy")\npublic = importlib.import_module("tools.integrity")\nboundary_original = legacy._integrity_boundary_original_compare_papers\nfinal_original = legacy._integrity_final_original_compare_papers\nfor _ in range(3):\n    for name in ("tools.integrity", "tools.integrity_boundary"):\n        sys.modules.pop(name, None)\n        tools.__dict__.pop(name.rsplit(".", 1)[-1], None)\n    public = importlib.import_module("tools.integrity")\n    assert legacy._integrity_boundary_original_compare_papers is boundary_original\n    assert legacy._integrity_final_original_compare_papers is final_original\n    assert public.compare_papers.__globals__["_original_compare_papers"] is final_original\nassert boundary_original.__module__ == "tools.integrity_legacy"\n"""\n    )\n    assert result.returncode == 0, result.stderr\n\n\ndef test_rag_reimports_do_not_stack_wrappers_or_replace_singleton_state():\n    result = _run(\n        r"""\nimport importlib\nimport sys\nimport tools\n\nlegacy = importlib.import_module("tools.rag_legacy")\npublic = importlib.import_module("tools.rag")\nbase = legacy._boundary_original_RAGLayer\ninstances = legacy._boundary_rag_instances\nlock = legacy._boundary_rag_lock\nfor _ in range(3):\n    sys.modules.pop("tools.rag", None)\n    tools.__dict__.pop("rag", None)\n    public = importlib.import_module("tools.rag")\n    assert legacy._boundary_original_RAGLayer is base\n    assert public.RAGLayer.__mro__[1] is base\n    assert public._RAG_INSTANCES is instances\n    assert public._RAG_LOCK is lock\n"""\n    )\n    assert result.returncode == 0, result.stderr\n''',
    encoding="utf-8",
)
