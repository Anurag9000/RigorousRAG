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


# Classic storage: preserve the legacy base and one public wrapper class.
replace_once(
    "storage.py",
    '''_original_storage_manager = _implementation.StorageManager\n_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400\n''',
    '''if not hasattr(_implementation, "_boundary_original_StorageManager"):\n    _implementation._boundary_original_StorageManager = _implementation.StorageManager\n_original_storage_manager = _implementation._boundary_original_StorageManager\n_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400\n''',
)
replace_once(
    "storage.py",
    "class StorageManager(_original_storage_manager):\n",
    "class _StorageManagerBoundary(_original_storage_manager):\n",
)
replace_once(
    "storage.py",
    '''\n\n_implementation.StorageManager = StorageManager\n_implementation.__doc__ = __doc__\nsys.modules[__name__] = _implementation\n''',
    '''\n\nif not hasattr(_implementation, "_boundary_public_StorageManager"):\n    _implementation._boundary_public_StorageManager = _StorageManagerBoundary\nStorageManager = _implementation._boundary_public_StorageManager\n\n_implementation.StorageManager = StorageManager\n_implementation.__doc__ = __doc__\nsys.modules[__name__] = _implementation\n''',
)

# Retained document registry: preserve class identity and bind roots/database identity.
replace_once(
    "tools/document_store.py",
    '''_original_document_store = _implementation.DocumentStore\n\n\ndef _normalize_registry_environment()''',
    '''if not hasattr(_implementation, "_boundary_original_DocumentStore"):\n    _implementation._boundary_original_DocumentStore = _implementation.DocumentStore\n_original_document_store = _implementation._boundary_original_DocumentStore\n_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)\n\n\ndef _normalize_registry_environment()''',
)
replace_once(
    "tools/document_store.py",
    '''def _contains_ascii_control(value: str) -> bool:\n    return any(ord(character) < 32 or ord(character) == 127 for character in value)\n\n\ndef _lexical_absolute''',
    '''def _contains_ascii_control(value: str) -> bool:\n    return any(ord(character) < 32 or ord(character) == 127 for character in value)\n\n\ndef _is_redirecting(metadata: os.stat_result) -> bool:\n    return stat.S_ISLNK(metadata.st_mode) or bool(\n        int(getattr(metadata, "st_file_attributes", 0))\n        & _WINDOWS_REPARSE_POINT\n    )\n\n\ndef _identity(metadata: os.stat_result) -> tuple[int, int]:\n    return int(metadata.st_dev), int(metadata.st_ino)\n\n\ndef _path_identity(path: Path, label: str, *, directory: bool) -> tuple[int, int]:\n    try:\n        metadata = path.lstat()\n    except OSError as exc:\n        raise OSError(f"{label} could not be inspected safely.") from exc\n    if _is_redirecting(metadata):\n        raise ValueError(f"{label} may not be a symbolic link or reparse point.")\n    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)\n    if not expected:\n        kind = "directory" if directory else "regular file"\n        raise OSError(f"{label} must remain a {kind}.")\n    return _identity(metadata)\n\n\ndef _lexical_absolute''',
)
replace_once(
    "tools/document_store.py",
    '''    absolute = Path(os.path.abspath(path))\n    for candidate in (absolute, *absolute.parents):\n        if candidate.is_symlink():\n            raise ValueError(f"{label} may not contain symbolic-link components.")\n    return absolute\n''',
    '''    absolute = Path(os.path.abspath(path))\n    for candidate in (absolute, *absolute.parents):\n        try:\n            metadata = candidate.lstat()\n        except FileNotFoundError:\n            continue\n        except OSError as exc:\n            raise ValueError(f"{label} could not be validated safely.") from exc\n        if _is_redirecting(metadata):\n            raise ValueError(\n                f"{label} may not contain symbolic links or reparse points."\n            )\n    return absolute\n''',
)
replace_once(
    "tools/document_store.py",
    "class DocumentStore(_original_document_store):\n",
    "class _DocumentStoreBoundary(_original_document_store):\n",
)
replace_once(
    "tools/document_store.py",
    '''        safe_path = _lexical_absolute(selected_path, "DOCUMENT_DB_PATH")\n        safe_root = _lexical_absolute(selected_root, "UPLOAD_DIR")\n        super().__init__(path=safe_path, upload_root=safe_root)\n        self._ensure_storage_paths()\n''',
    '''        safe_path = _lexical_absolute(selected_path, "DOCUMENT_DB_PATH")\n        safe_root = _lexical_absolute(selected_root, "UPLOAD_DIR")\n        safe_path.parent.mkdir(parents=True, exist_ok=True)\n        safe_root.mkdir(parents=True, exist_ok=True)\n        self._boundary_database_parent_identity = _path_identity(\n            safe_path.parent,\n            "DOCUMENT_DB_PATH parent",\n            directory=True,\n        )\n        self._boundary_upload_root_identity = _path_identity(\n            safe_root,\n            "UPLOAD_DIR",\n            directory=True,\n        )\n        self._boundary_database_identity: tuple[int, int] | None = None\n        super().__init__(path=safe_path, upload_root=safe_root)\n        self._boundary_database_identity = _path_identity(\n            self.path,\n            "DOCUMENT_DB_PATH",\n            directory=False,\n        )\n        self._ensure_storage_paths()\n''',
)
replace_once(
    "tools/document_store.py",
    '''    def _ensure_storage_paths(self) -> None:\n        _lexical_absolute(self.path, "DOCUMENT_DB_PATH")\n        _lexical_absolute(self.upload_root, "UPLOAD_DIR")\n        if not self.path.parent.exists() or not self.path.parent.is_dir():\n            raise OSError("DOCUMENT_DB_PATH parent must remain a directory.")\n        if not self.upload_root.exists() or not self.upload_root.is_dir():\n            raise OSError("UPLOAD_DIR must remain a directory.")\n        if self.path.exists():\n            try:\n                mode = self.path.stat(follow_symlinks=False).st_mode\n            except OSError as exc:\n                raise OSError("DOCUMENT_DB_PATH could not be inspected.") from exc\n            if not stat.S_ISREG(mode):\n                raise OSError("DOCUMENT_DB_PATH must remain a regular file.")\n''',
    '''    def _ensure_storage_paths(self) -> None:\n        safe_path = _lexical_absolute(self.path, "DOCUMENT_DB_PATH")\n        safe_root = _lexical_absolute(self.upload_root, "UPLOAD_DIR")\n        if _path_identity(\n            safe_path.parent,\n            "DOCUMENT_DB_PATH parent",\n            directory=True,\n        ) != self._boundary_database_parent_identity:\n            raise OSError("DOCUMENT_DB_PATH parent identity changed after initialization.")\n        if _path_identity(\n            safe_root,\n            "UPLOAD_DIR",\n            directory=True,\n        ) != self._boundary_upload_root_identity:\n            raise OSError("UPLOAD_DIR identity changed after initialization.")\n        expected_database = self._boundary_database_identity\n        if safe_path.exists():\n            current_database = _path_identity(\n                safe_path,\n                "DOCUMENT_DB_PATH",\n                directory=False,\n            )\n            if expected_database is not None and current_database != expected_database:\n                raise OSError("DOCUMENT_DB_PATH identity changed after initialization.")\n        elif expected_database is not None:\n            raise OSError("DOCUMENT_DB_PATH disappeared after initialization.")\n''',
)
replace_once(
    "tools/document_store.py",
    '''        if now is not None:\n            try:\n                current = float(now)\n''',
    '''        if now is not None:\n            if isinstance(now, bool):\n                raise ValueError("now must be numeric.")\n            try:\n                current = float(now)\n''',
)
replace_once(
    "tools/document_store.py",
    '''\n\ndef get_document_store(\n''',
    '''\n\nif not hasattr(_implementation, "_boundary_public_DocumentStore"):\n    _implementation._boundary_public_DocumentStore = _DocumentStoreBoundary\nDocumentStore = _implementation._boundary_public_DocumentStore\n\n\ndef get_document_store(\n''',
)

# RAG: preserve one public wrapper class as well as its base/cache.
replace_once(
    "tools/rag.py",
    '''\n\n_implementation.RAGLayer = RAGLayer\n''',
    '''\n\nif not hasattr(_implementation, "_boundary_public_RAGLayer"):\n    _implementation._boundary_public_RAGLayer = RAGLayer\nRAGLayer = _implementation._boundary_public_RAGLayer\n\n_implementation.RAGLayer = RAGLayer\n''',
)

# Search agent: persistent originals/public classes and exact numeric semantics.
replace_once(
    "search_agent.py",
    '''import itertools\nimport json\nimport math\nimport os\n''',
    '''import itertools\nimport json\nimport math\nimport operator\nimport os\n''',
)
replace_once(
    "search_agent.py",
    '''_original_validate_schema_value = _implementation._validate_schema_value\n_original_tool_execution = _implementation.ToolExecution\n_MAX_IDENTIFIER_CHARS = 200\n''',
    '''if not hasattr(_implementation, "_boundary_original_validate_schema_value"):\n    _implementation._boundary_original_validate_schema_value = (\n        _implementation._validate_schema_value\n    )\nif not hasattr(_implementation, "_boundary_original_ToolExecution"):\n    _implementation._boundary_original_ToolExecution = _implementation.ToolExecution\nif not hasattr(_implementation, "_boundary_original_SearchAgent"):\n    _implementation._boundary_original_SearchAgent = _implementation.SearchAgent\n\n_original_validate_schema_value = (\n    _implementation._boundary_original_validate_schema_value\n)\n_original_tool_execution = _implementation._boundary_original_ToolExecution\n_original_search_agent = _implementation._boundary_original_SearchAgent\n_MAX_IDENTIFIER_CHARS = 200\n''',
)
replace_once(
    "search_agent.py",
    '''    rendered = value.strip()\n    if len(rendered) > _MAX_PROVIDER_FIELD_CHARS:\n''',
    '''    rendered = value.strip()\n    if len(rendered) > _MAX_PROVIDER_FIELD_CHARS:\n''',
)
replace_once(
    "search_agent.py",
    '''    if any(character in rendered for character in ("\\x00", "\\r", "\\n")):\n        raise ValueError(f"{label} contains invalid control characters.")\n''',
    '''    if any(ord(character) < 32 or ord(character) == 127 for character in rendered):\n        raise ValueError(f"{label} contains invalid control characters.")\n''',
)
replace_once(
    "search_agent.py",
    '''        or any(character in selected for character in ("\\x00", "\\r", "\\n"))\n''',
    '''        or any(ord(character) < 32 or ord(character) == 127 for character in selected)\n''',
)
replace_once(
    "search_agent.py",
    '''def _strict_int(value: Any, label: str, *, minimum: int, maximum: int) -> int:\n    if isinstance(value, bool):\n        raise ValueError(f"{label} must be an integer.")\n    try:\n        parsed = int(value)\n    except (TypeError, ValueError, OverflowError) as exc:\n        raise ValueError(f"{label} must be an integer.") from exc\n    if isinstance(value, float) and not value.is_integer():\n        raise ValueError(f"{label} must be an integer.")\n    if not minimum <= parsed <= maximum:\n        raise ValueError(f"{label} must be between {minimum} and {maximum}.")\n    return parsed\n''',
    '''def _strict_int(value: Any, label: str, *, minimum: int, maximum: int) -> int:\n    if isinstance(value, bool):\n        raise ValueError(f"{label} must be an integer.")\n    try:\n        parsed = int(operator.index(value))\n    except (TypeError, ValueError, OverflowError) as exc:\n        raise ValueError(f"{label} must be an integer.") from exc\n    if not minimum <= parsed <= maximum:\n        raise ValueError(f"{label} must be between {minimum} and {maximum}.")\n    return parsed\n''',
)
replace_once(
    "search_agent.py",
    '''def _finite_timeout(value: Any, label: str) -> float:\n    try:\n''',
    '''def _finite_timeout(value: Any, label: str) -> float:\n    if isinstance(value, bool):\n        raise ValueError(f"{label} must be numeric.")\n    try:\n''',
)
replace_once(
    "search_agent.py",
    "class ToolExecution(_original_tool_execution):\n",
    "class _ToolExecutionBoundary(_original_tool_execution):\n",
)
replace_once(
    "search_agent.py",
    "class SearchAgent(_implementation.SearchAgent):\n",
    "class _SearchAgentBoundary(_original_search_agent):\n",
)
replace_once(
    "search_agent.py",
    '''\n\n_implementation._validate_schema_value = _validate_schema_value\n_implementation.ToolExecution = ToolExecution\n_implementation.SearchAgent = SearchAgent\n''',
    '''\n\nif not hasattr(_implementation, "_boundary_public_ToolExecution"):\n    _implementation._boundary_public_ToolExecution = _ToolExecutionBoundary\nif not hasattr(_implementation, "_boundary_public_SearchAgent"):\n    _implementation._boundary_public_SearchAgent = _SearchAgentBoundary\nToolExecution = _implementation._boundary_public_ToolExecution\nSearchAgent = _implementation._boundary_public_SearchAgent\n\n_implementation._validate_schema_value = _validate_schema_value\n_implementation.ToolExecution = ToolExecution\n_implementation.SearchAgent = SearchAgent\n''',
)

append_once(
    "tests/unit/test_document_store_root_paths.py",
    "test_registry_detects_nonlink_root_replacement_and_boolean_clock",
    '''def test_registry_detects_nonlink_root_replacement_and_boolean_clock(tmp_path):\n    parent = tmp_path / "state"\n    database = parent / "documents.sqlite3"\n    uploads = tmp_path / "uploads"\n    store = DocumentStore(database, uploads)\n\n    moved = tmp_path / "state-original"\n    parent.rename(moved)\n    parent.mkdir()\n    assert store.ping() is False\n\n    with pytest.raises(ValueError, match="numeric"):\n        store.cleanup_orphans(now=True, job_store=object())\n\n\ndef test_registry_rejects_windows_reparse_components(monkeypatch, tmp_path):\n    from types import SimpleNamespace\n\n    uploads = tmp_path / "uploads"\n    uploads.mkdir()\n    database = tmp_path / "documents.sqlite3"\n    original_lstat = type(uploads).lstat\n\n    def reparse_lstat(self):\n        metadata = original_lstat(self)\n        if self == uploads:\n            return SimpleNamespace(\n                st_mode=metadata.st_mode,\n                st_file_attributes=0x400,\n                st_dev=metadata.st_dev,\n                st_ino=metadata.st_ino,\n            )\n        return metadata\n\n    monkeypatch.setattr(type(uploads), "lstat", reparse_lstat)\n    with pytest.raises(ValueError, match="reparse points"):\n        DocumentStore(database, uploads)\n''',
)

append_once(
    "tests/unit/test_search_agent_provider_boundaries.py",
    "test_agent_integer_limits_require_index_and_timeouts_reject_booleans",
    '''def test_agent_integer_limits_require_index_and_timeouts_reject_booleans():\n    from decimal import Decimal\n    from fractions import Fraction\n\n    class ExactIndex:\n        def __index__(self):\n            return 3\n\n    agent = SearchAgent(owner_id="alice", max_turns=ExactIndex())\n    assert agent.max_turns == 3\n    for value in (1.0, Decimal("2"), Fraction(2, 1), Fraction(3, 2)):\n        with pytest.raises(ValueError, match="max_turns"):\n            SearchAgent(owner_id="alice", max_turns=value)\n    for name in ("request_timeout", "tool_timeout"):\n        with pytest.raises(ValueError, match=name):\n            SearchAgent(owner_id="alice", **{name: True})\n\n\ndef test_agent_model_and_provider_fields_reject_all_ascii_controls():\n    for model in ("bad\\tmodel", "bad\\x1bmodel", "bad\\x7fmodel"):\n        with pytest.raises(ValueError, match="model"):\n            SearchAgent(owner_id="alice", model=model)\n    for value in ("bad\\tkey", "bad\\x1bkey", "bad\\x7fkey"):\n        with pytest.raises(ValueError, match="control"):\n            SearchAgent(owner_id="alice", api_key=value)\n''',
)

append_once(
    "tests/unit/test_compatibility_reload_boundaries.py",
    "test_stateful_class_wrappers_preserve_public_identity_across_reimports",
    '''def test_stateful_class_wrappers_preserve_public_identity_across_reimports():\n    result = _run(\n        r"""\nimport importlib\nimport sys\nimport tools\n\n# Classic storage.\nlegacy_storage = importlib.import_module("storage_legacy")\npublic_storage = importlib.import_module("storage")\nstorage_base = legacy_storage._boundary_original_StorageManager\nstorage_public = legacy_storage._boundary_public_StorageManager\nfor _ in range(3):\n    sys.modules.pop("storage", None)\n    public_storage = importlib.import_module("storage")\n    assert public_storage.StorageManager is storage_public\n    assert storage_public.__mro__[1] is storage_base\n\n# Document registry.\nlegacy_document = importlib.import_module("tools.document_store_legacy")\npublic_document = importlib.import_module("tools.document_store")\ndocument_base = legacy_document._boundary_original_DocumentStore\ndocument_public = legacy_document._boundary_public_DocumentStore\nfor _ in range(3):\n    sys.modules.pop("tools.document_store", None)\n    tools.__dict__.pop("document_store", None)\n    public_document = importlib.import_module("tools.document_store")\n    assert public_document.DocumentStore is document_public\n    assert document_public.__mro__[1] is document_base\n\n# Search agent.\nlegacy_agent = importlib.import_module("search_agent_legacy")\npublic_agent = importlib.import_module("search_agent")\nagent_base = legacy_agent._boundary_original_SearchAgent\nagent_public = legacy_agent._boundary_public_SearchAgent\nexecution_base = legacy_agent._boundary_original_ToolExecution\nexecution_public = legacy_agent._boundary_public_ToolExecution\nvalidator = legacy_agent._boundary_original_validate_schema_value\nfor _ in range(3):\n    sys.modules.pop("search_agent", None)\n    public_agent = importlib.import_module("search_agent")\n    assert public_agent.SearchAgent is agent_public\n    assert agent_public.__mro__[1] is agent_base\n    assert public_agent.ToolExecution is execution_public\n    assert execution_public.__mro__[1] is execution_base\n    assert legacy_agent._boundary_original_validate_schema_value is validator\n\n# RAG public class, in addition to base/cache checks from pass fourteen.\nlegacy_rag = importlib.import_module("tools.rag_legacy")\npublic_rag = importlib.import_module("tools.rag")\nrag_public = legacy_rag._boundary_public_RAGLayer\nfor _ in range(3):\n    sys.modules.pop("tools.rag", None)\n    tools.__dict__.pop("rag", None)\n    public_rag = importlib.import_module("tools.rag")\n    assert public_rag.RAGLayer is rag_public\n"""\n    )\n    assert result.returncode == 0, result.stderr\n''',
)
