"""Fail-closed local model/tokenizer artifact binding for advanced RAG training.

No network fallback or remote model code is permitted. A directory is recursively hashed
before loading; symlinks, path escapes, excessive file counts and excessive bytes are
rejected. Hugging Face loading is optional and occurs only when an explicit load function is
called with ``local_files_only=True`` and ``trust_remote_code=False``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

_MAX_FILES = 200_000
_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024 * 1024
_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024 * 1024


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def local_tree_sha256(path: str | Path) -> str:
    """Content-address a local artifact tree independent of its absolute directory name."""
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("local artifact root must be a regular directory, not a symlink")
    records = []
    total = 0
    for index, candidate in enumerate(sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())):
        if index >= _MAX_FILES:
            raise ValueError("local artifact tree exceeds file-count safety bound")
        if candidate.is_symlink():
            raise ValueError("local artifact tree may not contain symlinks")
        if candidate.is_dir():
            continue
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("local artifact file escapes artifact root") from exc
        if not resolved.is_file():
            raise ValueError("local artifact tree contains a non-regular filesystem entry")
        size = resolved.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise ValueError("local artifact file exceeds byte safety bound")
        total += size
        if total > _MAX_TOTAL_BYTES:
            raise ValueError("local artifact tree exceeds aggregate byte safety bound")
        records.append({"path": relative, "bytes": size, "sha256": _file_sha(resolved)})
    if not records:
        raise ValueError("local artifact tree is empty")
    return hashlib.sha256(_canonical({"schema": "rigorousrag-local-artifact-tree/v1", "files": records})).hexdigest()


@dataclass(frozen=True)
class LocalArtifactTreeBinding:
    path: str
    expected_sha256: str
    artifact_kind: Literal["causal_lm", "seq2seq_lm", "tokenizer", "sequence_classifier"]

    def __post_init__(self) -> None:
        root = Path(self.path).expanduser().resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("local artifact path must be a non-symlink directory")
        object.__setattr__(self, "path", str(root))
        object.__setattr__(self, "expected_sha256", _sha(self.expected_sha256, "expected_sha256"))
        if self.artifact_kind not in {"causal_lm", "seq2seq_lm", "tokenizer", "sequence_classifier"}:
            raise ValueError("unsupported local artifact kind")

    def verify(self) -> str:
        actual = local_tree_sha256(self.path)
        if actual != self.expected_sha256:
            raise ValueError("local artifact tree digest differs from admitted identity")
        return actual

    @property
    def binding_sha256(self) -> str:
        return hashlib.sha256(_canonical({"schema": "rigorousrag-local-artifact-binding/v1", **asdict(self)})).hexdigest()


def load_local_tokenizer(binding: LocalArtifactTreeBinding) -> Any:
    if binding.artifact_kind != "tokenizer":
        raise ValueError("tokenizer loader requires artifact_kind=tokenizer")
    binding.verify()
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required only when loading a local tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        binding.path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is None:
            raise ValueError("local tokenizer requires an existing pad token or EOS token")
        tokenizer.pad_token = eos
    return tokenizer


def load_local_language_model(binding: LocalArtifactTreeBinding) -> Any:
    if binding.artifact_kind not in {"causal_lm", "seq2seq_lm"}:
        raise ValueError("language-model loader requires causal_lm or seq2seq_lm binding")
    binding.verify()
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    except Exception as exc:
        raise RuntimeError("transformers is required only when loading a local language model") from exc
    model_class = AutoModelForCausalLM if binding.artifact_kind == "causal_lm" else AutoModelForSeq2SeqLM
    return model_class.from_pretrained(
        binding.path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )


def load_local_sequence_classifier(binding: LocalArtifactTreeBinding) -> Any:
    if binding.artifact_kind != "sequence_classifier":
        raise ValueError("sequence-classifier loader requires artifact_kind=sequence_classifier")
    binding.verify()
    try:
        from transformers import AutoModelForSequenceClassification
    except Exception as exc:
        raise RuntimeError("transformers is required only when loading a local sequence classifier") from exc
    return AutoModelForSequenceClassification.from_pretrained(
        binding.path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )


def assert_grounded_artifact_bindings(*, base_model: LocalArtifactTreeBinding, tokenizer: LocalArtifactTreeBinding, base_model_sha256: str, tokenizer_sha256: str) -> None:
    if base_model.expected_sha256 != _sha(base_model_sha256, "base_model_sha256"):
        raise ValueError("base-model local binding differs from grounded training plan")
    if tokenizer.expected_sha256 != _sha(tokenizer_sha256, "tokenizer_sha256"):
        raise ValueError("tokenizer local binding differs from grounded training plan")
    base_model.verify(); tokenizer.verify()


def assert_dynamic_generator_binding(*, generator: LocalArtifactTreeBinding, base_generator_sha256: str) -> None:
    if generator.expected_sha256 != _sha(base_generator_sha256, "base_generator_sha256"):
        raise ValueError("generator local binding differs from dynamic policy training plan")
    generator.verify()


__all__ = [
    "LocalArtifactTreeBinding",
    "assert_dynamic_generator_binding",
    "assert_grounded_artifact_bindings",
    "load_local_language_model",
    "load_local_sequence_classifier",
    "load_local_tokenizer",
    "local_tree_sha256",
]
