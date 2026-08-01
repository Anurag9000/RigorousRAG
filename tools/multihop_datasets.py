"""Strict local adapters for common multi-hop QA benchmark formats."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.multihop_evaluation import MultiHopEvaluationExample, SupportFact

_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_EXAMPLES = 100_000
_MAX_JSON_DEPTH = 40
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_DATASETS = frozenset({"hotpotqa", "2wikimultihopqa", "musique"})


def _redirected(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT
    )


def _regular_path(value: str | os.PathLike[str]) -> Path:
    rendered = os.fspath(value)
    if not isinstance(rendered, str) or not rendered or rendered != rendered.strip():
        raise ValueError("dataset path is invalid.")
    path = Path(rendered).expanduser().absolute()
    for item in (path, *path.parents):
        try:
            metadata = item.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("dataset path could not be validated.") from exc
        if _redirected(metadata):
            raise ValueError("dataset path may not contain symbolic links or reparse points.")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ValueError("dataset file is unavailable.") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("dataset path must identify a regular file.")
    if metadata.st_size <= 0 or metadata.st_size > _MAX_FILE_BYTES:
        raise ValueError("dataset file is empty or exceeds the byte limit.")
    return path


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON number: {value}")


def _depth(value: Any, current: int = 0) -> int:
    if current > _MAX_JSON_DEPTH:
        return current
    if isinstance(value, Mapping):
        return max((_depth(item, current + 1) for item in value.values()), default=current)
    if isinstance(value, list):
        return max((_depth(item, current + 1) for item in value), default=current)
    return current


def _loads(raw: str, label: str) -> Any:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=_parse_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} contains invalid JSON.") from exc
    if _depth(value) > _MAX_JSON_DEPTH:
        raise ValueError(f"{label} exceeds the JSON nesting limit.")
    return value


def _read(path: Path) -> tuple[bytes, str]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise ValueError("dataset file could not be read.") from exc
    if len(raw) > _MAX_FILE_BYTES:
        raise ValueError("dataset file exceeds the byte limit.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("dataset file must be UTF-8.") from exc
    return raw, text


def _text(value: Any, label: str, maximum: int = 20_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = " ".join(value.split())
    if (
        not rendered
        or len(rendered) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    return rendered


def _optional_text(value: Any, label: str, maximum: int = 20_000) -> str | None:
    if value is None:
        return None
    return _text(value, label, maximum)


def _index(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 1_000_000:
        raise ValueError(f"{label} must be a bounded non-negative integer.")
    return value


def _aliases(record: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    primary = record.get("answer")
    if isinstance(primary, str):
        values.append(_text(primary, "answer", 5_000))
    for field in ("answers", "answer_aliases", "aliases"):
        raw = record.get(field)
        if raw is None:
            continue
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list) or len(raw) > 50:
            raise ValueError(f"{field} must be a bounded list of strings.")
        for value in raw:
            rendered = _text(value, field, 5_000)
            if rendered.casefold() not in {item.casefold() for item in values}:
                values.append(rendered)
    if not values:
        raise ValueError("benchmark example has no answer.")
    return tuple(values)


def _support_pair(value: Any, label: str) -> tuple[str, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain [document, sentence_index].")
    return _text(value[0], f"{label}.document", 1_000), _index(value[1], f"{label}.index")


def _hotpot_like(record: Mapping[str, Any], position: int) -> MultiHopEvaluationExample:
    example_id = _text(record.get("_id", record.get("id", f"example-{position}")), "example_id", 500)
    question = _text(record.get("question"), "question")
    supporting = record.get("supporting_facts")
    if not isinstance(supporting, list) or len(supporting) > 200:
        raise ValueError("supporting_facts must be a bounded list.")
    facts: list[SupportFact] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(supporting):
        title, sentence = _support_pair(value, f"supporting_facts[{index}]")
        key = (title, f"sentence:{sentence}")
        if key not in seen:
            seen.add(key)
            facts.append(SupportFact(title, key[1]))
    required_hops = max(1, len({fact.document_id for fact in facts}))
    return MultiHopEvaluationExample(
        example_id=example_id,
        question=question,
        answers=_aliases(record),
        support_facts=tuple(facts),
        required_hops=min(required_hops, 100),
    )


def _musique(record: Mapping[str, Any], position: int) -> MultiHopEvaluationExample:
    example_id = _text(record.get("id", record.get("_id", f"example-{position}")), "example_id", 500)
    question = _text(record.get("question"), "question")
    paragraphs = record.get("paragraphs", [])
    if not isinstance(paragraphs, list) or len(paragraphs) > 10_000:
        raise ValueError("paragraphs must be a bounded list.")
    by_index: dict[int, tuple[str, bool]] = {}
    for offset, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, Mapping):
            raise ValueError("each paragraph must be an object.")
        raw_index = paragraph.get("idx", paragraph.get("index", offset))
        paragraph_index = _index(raw_index, "paragraph index")
        title = _optional_text(paragraph.get("title"), "paragraph title", 1_000)
        document_id = title or f"paragraph-{paragraph_index}"
        supporting = paragraph.get("is_supporting", False)
        if not isinstance(supporting, bool):
            raise ValueError("is_supporting must be a boolean.")
        if paragraph_index in by_index:
            raise ValueError("paragraph indexes must be unique.")
        by_index[paragraph_index] = (document_id, supporting)

    decomposition = record.get("question_decomposition", [])
    if not isinstance(decomposition, list) or len(decomposition) > 100:
        raise ValueError("question_decomposition must be a bounded list.")
    support_indexes: list[int] = []
    for step in decomposition:
        if not isinstance(step, Mapping):
            raise ValueError("question_decomposition entries must be objects.")
        raw_index = step.get("paragraph_support_idx")
        if raw_index is not None:
            support_indexes.append(_index(raw_index, "paragraph_support_idx"))
    support_indexes.extend(index for index, (_title, supporting) in by_index.items() if supporting)
    facts: list[SupportFact] = []
    seen: set[int] = set()
    for paragraph_index in support_indexes:
        if paragraph_index in seen:
            continue
        seen.add(paragraph_index)
        if paragraph_index not in by_index:
            raise ValueError("support paragraph index is not declared in paragraphs.")
        document_id, _supporting = by_index[paragraph_index]
        facts.append(SupportFact(document_id, f"paragraph:{paragraph_index}"))
    required_hops = max(1, len(decomposition) or len(facts))
    return MultiHopEvaluationExample(
        example_id=example_id,
        question=question,
        answers=_aliases(record),
        support_facts=tuple(facts),
        required_hops=min(required_hops, 100),
    )


@dataclass(frozen=True)
class LoadedMultiHopDataset:
    dataset: str
    split: str
    sha256: str
    examples: tuple[MultiHopEvaluationExample, ...]

    def __post_init__(self) -> None:
        if self.dataset not in _DATASETS:
            raise ValueError("dataset is unsupported.")
        object.__setattr__(self, "split", _text(self.split, "split", 100))
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("sha256 is invalid.")
        if not self.examples or len(self.examples) > _MAX_EXAMPLES:
            raise ValueError("examples are empty or exceed the limit.")
        if len({example.example_id for example in self.examples}) != len(self.examples):
            raise ValueError("example identifiers must be unique.")


def load_multihop_dataset(
    path: str | os.PathLike[str],
    *,
    dataset: str,
    split: str,
    max_examples: int = _MAX_EXAMPLES,
) -> LoadedMultiHopDataset:
    """Load HotpotQA, 2WikiMultiHopQA, or MuSiQue without network access."""

    selected = _text(dataset, "dataset", 100).lower().replace("-", "")
    aliases = {
        "hotpotqa": "hotpotqa",
        "2wiki": "2wikimultihopqa",
        "2wikimultihopqa": "2wikimultihopqa",
        "musique": "musique",
    }
    selected = aliases.get(selected, selected)
    if selected not in _DATASETS:
        raise ValueError("dataset must be hotpotqa, 2wikimultihopqa, or musique.")
    if isinstance(max_examples, bool) or not isinstance(max_examples, int):
        raise ValueError("max_examples must be an integer.")
    if not 1 <= max_examples <= _MAX_EXAMPLES:
        raise ValueError(f"max_examples must be between 1 and {_MAX_EXAMPLES}.")
    source = _regular_path(path)
    raw, text = _read(source)
    stripped = text.lstrip()
    records: list[Any]
    if stripped.startswith("[") or stripped.startswith("{") and source.suffix.lower() != ".jsonl":
        parsed = _loads(text, "dataset")
        if isinstance(parsed, Mapping) and "data" in parsed:
            parsed = parsed["data"]
        if not isinstance(parsed, list):
            raise ValueError("dataset JSON root must be a list or contain a data list.")
        records = parsed
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            records.append(_loads(line, f"dataset line {line_number}"))
            if len(records) > max_examples:
                break
    if not records:
        raise ValueError("dataset contains no examples.")
    if len(records) > max_examples:
        records = records[:max_examples]
    examples: list[MultiHopEvaluationExample] = []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("every dataset example must be an object.")
        examples.append(
            _musique(record, position)
            if selected == "musique"
            else _hotpot_like(record, position)
        )
    return LoadedMultiHopDataset(
        dataset=selected,
        split=split,
        sha256=hashlib.sha256(raw).hexdigest(),
        examples=tuple(examples),
    )


__all__ = ["LoadedMultiHopDataset", "load_multihop_dataset"]
