"""Adapters for common RAG benchmark record formats.

These adapters intentionally operate on Python mappings so dataset libraries remain
optional. They normalize heterogeneous rows into a stable evaluation contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    query: str
    answers: Tuple[str, ...] = ()
    relevant_ids: Tuple[str, ...] = ()
    contexts: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _answers(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Mapping):
        for key in ("text", "answer", "answers"):
            if key in value:
                return _answers(value[key])
        return ()
    if isinstance(value, Sequence):
        return tuple(_text(item) for item in value if _text(item))
    return (_text(value),) if _text(value) else ()


def hotpotqa(row: Mapping[str, Any]) -> BenchmarkExample:
    identifier = _text(row.get("_id") or row.get("id"))
    question = _text(row.get("question") or row.get("query"))
    context = row.get("context") or []
    contexts = []
    relevant = []
    supporting = {
        _text(item[0])
        for item in (row.get("supporting_facts") or [])
        if isinstance(item, Sequence) and not isinstance(item, str) and item
    }
    for item in context:
        if isinstance(item, Sequence) and not isinstance(item, str) and len(item) >= 2:
            title = _text(item[0])
            sentences = item[1]
            text = " ".join(_text(sentence) for sentence in sentences) if isinstance(
                sentences, Sequence
            ) and not isinstance(sentences, str) else _text(sentences)
            contexts.append(f"{title}\n{text}".strip())
            if title in supporting:
                relevant.append(title)
    return BenchmarkExample(
        identifier,
        question,
        _answers(row.get("answer")),
        tuple(relevant),
        tuple(contexts),
        {"dataset": "hotpotqa"},
    )


def musique(row: Mapping[str, Any]) -> BenchmarkExample:
    paragraphs = row.get("paragraphs") or row.get("contexts") or []
    contexts = []
    relevant = []
    for index, paragraph in enumerate(paragraphs):
        if isinstance(paragraph, Mapping):
            title = _text(paragraph.get("title") or paragraph.get("idx") or index)
            text = _text(paragraph.get("paragraph_text") or paragraph.get("text"))
            contexts.append(f"{title}\n{text}".strip())
            if paragraph.get("is_supporting") is True:
                relevant.append(title)
        else:
            contexts.append(_text(paragraph))
    return BenchmarkExample(
        _text(row.get("id") or row.get("_id")),
        _text(row.get("question") or row.get("query")),
        _answers(row.get("answer") or row.get("answers")),
        tuple(relevant),
        tuple(contexts),
        {"dataset": "musique"},
    )


def qasper(row: Mapping[str, Any]) -> BenchmarkExample:
    question = row.get("question") or {}
    if isinstance(question, Mapping):
        query = _text(question.get("question") or question.get("text"))
        answers = _answers(question.get("answers"))
        identifier = _text(question.get("question_id") or row.get("id"))
    else:
        query = _text(question)
        answers = _answers(row.get("answers") or row.get("answer"))
        identifier = _text(row.get("id"))
    full_text = row.get("full_text") or row.get("contexts") or []
    contexts = []
    if isinstance(full_text, Mapping):
        for section, paragraphs in full_text.items():
            if isinstance(paragraphs, Sequence) and not isinstance(paragraphs, str):
                contexts.extend(f"{section}\n{_text(p)}".strip() for p in paragraphs)
            else:
                contexts.append(f"{section}\n{_text(paragraphs)}".strip())
    elif isinstance(full_text, Sequence) and not isinstance(full_text, str):
        contexts.extend(_text(item) for item in full_text if _text(item))
    return BenchmarkExample(
        identifier,
        query,
        answers,
        (),
        tuple(contexts),
        {"dataset": "qasper"},
    )


def scifact(row: Mapping[str, Any]) -> BenchmarkExample:
    claim_id = _text(row.get("id") or row.get("claim_id"))
    evidence = row.get("evidence") or {}
    relevant = []
    if isinstance(evidence, Mapping):
        relevant.extend(_text(key) for key in evidence)
    elif isinstance(evidence, Sequence) and not isinstance(evidence, str):
        for item in evidence:
            if isinstance(item, Mapping):
                relevant.append(_text(item.get("doc_id") or item.get("document_id")))
            else:
                relevant.append(_text(item))
    return BenchmarkExample(
        claim_id,
        _text(row.get("claim") or row.get("question") or row.get("query")),
        _answers(row.get("label") or row.get("answer")),
        tuple(item for item in relevant if item),
        (),
        {"dataset": "scifact"},
    )


def miracl(row: Mapping[str, Any]) -> BenchmarkExample:
    positives = row.get("positive_passages") or row.get("positives") or []
    relevant = []
    contexts = []
    for item in positives:
        if isinstance(item, Mapping):
            relevant.append(_text(item.get("docid") or item.get("id")))
            text = _text(item.get("text") or item.get("passage"))
            if text:
                contexts.append(text)
    metadata = {"dataset": "miracl"}
    if row.get("lang"):
        metadata["language"] = _text(row.get("lang"))
    return BenchmarkExample(
        _text(row.get("query_id") or row.get("id")),
        _text(row.get("query") or row.get("question")),
        _answers(row.get("answers")),
        tuple(item for item in relevant if item),
        tuple(contexts),
        metadata,
    )


ADAPTERS: Dict[str, Callable[[Mapping[str, Any]], BenchmarkExample]] = {
    "hotpotqa": hotpotqa,
    "musique": musique,
    "qasper": qasper,
    "scifact": scifact,
    "miracl": miracl,
}


def adapt_record(dataset: str, row: Mapping[str, Any]) -> BenchmarkExample:
    try:
        adapter = ADAPTERS[dataset.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark dataset: {dataset}") from exc
    example = adapter(row)
    if not example.query:
        raise ValueError("benchmark rows must produce a non-empty query.")
    return example


def iter_jsonl(path: str | Path, dataset: str) -> Iterator[BenchmarkExample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"line {line_number} is not a JSON object.")
            yield adapt_record(dataset, row)
