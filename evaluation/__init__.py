"""Bounded retrieval evaluation schemas, BEIR loaders, metrics, and runners."""

from __future__ import annotations



import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


@dataclass(frozen=True)
class EvaluationDocument:
    document_id: str
    text: str
    title: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _identifier(self.document_id, "document_id"))
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 5_000_000:
            raise ValueError("text must contain 1-5,000,000 characters.")
        if not isinstance(self.title, str) or len(self.title) > 10_000:
            raise ValueError("title is invalid.")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    text: str
    relevant: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id"))
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 100_000:
            raise ValueError("query text is invalid.")
        if not isinstance(self.relevant, Mapping):
            raise ValueError("relevant must be a mapping.")
        cleaned: dict[str, float] = {}
        for raw_id, raw_score in list(self.relevant.items())[:100_000]:
            identifier = _identifier(raw_id, "relevant document ID")
            if isinstance(raw_score, bool):
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(score) and score > 0.0:
                cleaned[identifier] = score
        object.__setattr__(self, "relevant", cleaned)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    documents: Mapping[str, EvaluationDocument]
    queries: Sequence[EvaluationQuery]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "dataset name", 200))
        if not isinstance(self.documents, Mapping) or len(self.documents) > 10_000_000:
            raise ValueError("documents must be a bounded mapping.")
        if isinstance(self.queries, (str, bytes, bytearray)) or len(self.queries) > 1_000_000:
            raise ValueError("queries must be a bounded sequence.")
        for identifier, document in self.documents.items():
            if identifier != document.document_id:
                raise ValueError("document mapping keys must match document IDs.")
        if not all(isinstance(query, EvaluationQuery) for query in self.queries):
            raise ValueError("queries contains an invalid value.")


@dataclass(frozen=True)
class RetrievalResult:
    document_id: str
    score: float
    rank: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _identifier(self.document_id, "document_id"))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer.")
        if isinstance(self.score, bool):
            raise ValueError("score must be finite.")
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("score must be finite.")
        object.__setattr__(self, "score", score)


import csv
import json
import os
import stat
from pathlib import Path
from typing import Any, Iterator


_MAX_LINE_BYTES = 20_000_000
_MAX_ROWS = 10_000_000
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _redirecting(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(metadata.st_mode) or bool(int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT)


def _safe_root(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("dataset path must be a filesystem path.")
    rendered = os.fspath(value)
    if not rendered or len(rendered) > 4096 or any(ord(ch) < 32 or ord(ch) == 127 for ch in rendered):
        raise ValueError("dataset path is invalid.")
    path = Path(rendered)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        if _redirecting(component):
            raise ValueError("dataset paths may not contain symbolic links or reparse points.")
    if not path.is_dir():
        raise ValueError("dataset path must be a directory.")
    return path


def _safe_member(root: Path, name: str) -> Path:
    path = root / name
    if _redirecting(path) or not path.is_file():
        raise ValueError(f"Required dataset member {name!r} is unavailable or redirecting.")
    if path.stat().st_size > 5_000_000_000:
        raise ValueError(f"Dataset member {name!r} exceeds the byte limit.")
    return path


def _json_lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for index, raw in enumerate(handle, start=1):
            if index > _MAX_ROWS:
                raise ValueError("Dataset exceeds the row limit.")
            if len(raw) > _MAX_LINE_BYTES:
                raise ValueError("Dataset contains an oversized JSON line.")
            try:
                value = json.loads(raw.decode("utf-8"), parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-standard JSON number")))
            except Exception as exc:
                raise ValueError(f"Invalid JSON object at {path.name}:{index}.") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path.name}:{index}.")
            yield value


def load_beir_dataset(root: str | os.PathLike[str], *, name: str | None = None) -> EvaluationDataset:
    """Load corpus.jsonl, queries.jsonl, and qrels/{test|dev|train}.tsv."""

    directory = _safe_root(root)
    corpus_path = _safe_member(directory, "corpus.jsonl")
    queries_path = _safe_member(directory, "queries.jsonl")
    qrels_directory = directory / "qrels"
    if _redirecting(qrels_directory) or not qrels_directory.is_dir():
        raise ValueError("qrels directory is unavailable or redirecting.")
    qrels_path = next((qrels_directory / candidate for candidate in ("test.tsv", "dev.tsv", "train.tsv") if (qrels_directory / candidate).is_file() and not _redirecting(qrels_directory / candidate)), None)
    if qrels_path is None:
        raise ValueError("No qrels split was found.")

    documents: dict[str, EvaluationDocument] = {}
    for row in _json_lines(corpus_path):
        identifier = row.get("_id")
        title = row.get("title", "")
        text = row.get("text", "")
        if not isinstance(identifier, str) or not isinstance(title, str) or not isinstance(text, str):
            raise ValueError("Corpus rows must contain string _id, title, and text fields.")
        if identifier in documents:
            raise ValueError("Corpus contains duplicate document IDs.")
        rendered = (title.strip() + "\n\n" + text.strip()).strip()
        documents[identifier] = EvaluationDocument(identifier, rendered, title=title, metadata={"source": "beir"})

    query_text: dict[str, str] = {}
    for row in _json_lines(queries_path):
        identifier, text = row.get("_id"), row.get("text")
        if not isinstance(identifier, str) or not isinstance(text, str):
            raise ValueError("Query rows must contain string _id and text fields.")
        if identifier in query_text:
            raise ValueError("Queries contain duplicate IDs.")
        query_text[identifier] = text

    relevance: dict[str, dict[str, float]] = {}
    with qrels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"query-id", "corpus-id", "score"}
        if not expected.issubset(set(reader.fieldnames or [])):
            raise ValueError("qrels header is invalid.")
        for index, row in enumerate(reader, start=1):
            if index > _MAX_ROWS:
                raise ValueError("qrels exceeds the row limit.")
            query_id, corpus_id = row.get("query-id"), row.get("corpus-id")
            if not isinstance(query_id, str) or not isinstance(corpus_id, str):
                raise ValueError("qrels contains invalid identifiers.")
            try:
                score = float(row.get("score", ""))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("qrels contains an invalid score.") from exc
            if score > 0.0:
                relevance.setdefault(query_id, {})[corpus_id] = score

    queries = [EvaluationQuery(identifier, text, relevance.get(identifier, {}), metadata={"source": "beir"}) for identifier, text in query_text.items()]
    dataset_name = name or directory.name or "beir"
    return EvaluationDataset(dataset_name, documents, queries, metadata={"format": "beir", "qrels_split": qrels_path.stem})


import math
from statistics import fmean
from typing import Iterable, Mapping, Sequence



def _top_ids(results: Sequence[RetrievalResult], k: int) -> list[str]:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    ordered = sorted(results, key=lambda item: (item.rank, -item.score, item.document_id))
    seen: set[str] = set()
    identifiers: list[str] = []
    for item in ordered:
        if item.document_id not in seen:
            seen.add(item.document_id)
            identifiers.append(item.document_id)
        if len(identifiers) >= k:
            break
    return identifiers


def precision_at_k(query: EvaluationQuery, results: Sequence[RetrievalResult], k: int) -> float:
    ids = _top_ids(results, k)
    return sum(identifier in query.relevant for identifier in ids) / k


def recall_at_k(query: EvaluationQuery, results: Sequence[RetrievalResult], k: int) -> float:
    if not query.relevant:
        return 0.0
    ids = _top_ids(results, k)
    return sum(identifier in query.relevant for identifier in ids) / len(query.relevant)


def hit_rate_at_k(query: EvaluationQuery, results: Sequence[RetrievalResult], k: int) -> float:
    return float(any(identifier in query.relevant for identifier in _top_ids(results, k)))


def reciprocal_rank(query: EvaluationQuery, results: Sequence[RetrievalResult], k: int) -> float:
    for rank, identifier in enumerate(_top_ids(results, k), start=1):
        if identifier in query.relevant:
            return 1.0 / rank
    return 0.0


def average_precision(query: EvaluationQuery, results: Sequence[RetrievalResult], k: int) -> float:
    if not query.relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, identifier in enumerate(_top_ids(results, k), start=1):
        if identifier in query.relevant:
            hits += 1
            total += hits / rank
    return total / min(len(query.relevant), k)


def ndcg_at_k(query: EvaluationQuery, results: Sequence[RetrievalResult], k: int) -> float:
    gains = [query.relevant.get(identifier, 0.0) for identifier in _top_ids(results, k)]
    dcg = sum((2.0 ** gain - 1.0) / math.log2(index + 2.0) for index, gain in enumerate(gains))
    ideal = sorted(query.relevant.values(), reverse=True)[:k]
    idcg = sum((2.0 ** gain - 1.0) / math.log2(index + 2.0) for index, gain in enumerate(ideal))
    return dcg / idcg if idcg > 0.0 else 0.0


def evaluate_query(query: EvaluationQuery, results: Sequence[RetrievalResult], *, ks: Iterable[int] = (1, 3, 5, 10)) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"precision@{k}"] = precision_at_k(query, results, k)
        metrics[f"recall@{k}"] = recall_at_k(query, results, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(query, results, k)
        metrics[f"mrr@{k}"] = reciprocal_rank(query, results, k)
        metrics[f"map@{k}"] = average_precision(query, results, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(query, results, k)
    return metrics


def aggregate_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    names = sorted({name for row in rows for name in row})
    return {name: fmean(float(row[name]) for row in rows if name in row) for name in names} if rows else {}


def citation_metrics(*, cited_source_ids: Iterable[str], supported_source_ids: Iterable[str], required_source_ids: Iterable[str] = ()) -> dict[str, float]:
    cited = {value for value in cited_source_ids if isinstance(value, str) and value}
    supported = {value for value in supported_source_ids if isinstance(value, str) and value}
    required = {value for value in required_source_ids if isinstance(value, str) and value}
    correct = cited & supported
    precision = len(correct) / len(cited) if cited else 0.0
    recall_base = required or supported
    recall = len(cited & recall_base) / len(recall_base) if recall_base else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "unsupported_citation_rate": len(cited - supported) / len(cited) if cited else 0.0,
        "citation_coverage": len(cited & required) / len(required) if required else recall,
    }


import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


Retriever = Callable[[str, int], Sequence[RetrievalResult] | Sequence[tuple[str, float]]]


def run_retrieval_evaluation(dataset: EvaluationDataset, retriever: Retriever, *, top_k: int = 10) -> dict[str, Any]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 1000:
        raise ValueError("top_k must be between 1 and 1000.")
    rows: list[dict[str, Any]] = []
    metric_rows: list[Mapping[str, float]] = []
    for query in dataset.queries:
        started = time.perf_counter()
        raw_results = retriever(query.text, top_k)
        results: list[RetrievalResult] = []
        for rank, item in enumerate(list(raw_results)[:top_k], start=1):
            if isinstance(item, RetrievalResult):
                results.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                results.append(RetrievalResult(str(item[0]), float(item[1]), rank))
        metrics = evaluate_query(query, results, ks=sorted({1, 3, 5, top_k}))
        metric_rows.append(metrics)
        rows.append({
            "query_id": query.query_id,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 6),
            "retrieved": [item.document_id for item in results],
            "metrics": metrics,
        })
    return {
        "dataset": dataset.name,
        "query_count": len(dataset.queries),
        "document_count": len(dataset.documents),
        "top_k": top_k,
        "metrics": aggregate_metrics(metric_rows),
        "queries": rows,
    }



__all__ = [
    "EvaluationDataset", "EvaluationDocument", "EvaluationQuery", "RetrievalResult",
    "aggregate_metrics", "citation_metrics", "evaluate_query", "load_beir_dataset",
    "run_retrieval_evaluation",
]
