from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected patch anchor missing in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


Path("tools/hybrid_retrieval.py").write_text(r'''"""Bounded hybrid retrieval, fusion, diversity, and offline sparse indexing."""

from __future__ import annotations

import heapq
import itertools
import math
import operator
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/+\-][A-Za-z0-9]+)*")
_MAX_CANDIDATES = 500
_MAX_TEXT_CHARS = 100_000
_MAX_TOKENS_PER_TEXT = 20_000


def _exact_int(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return numeric


def _finite(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return numeric


def tokenize(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise ValueError("Retrieval text must be a string.")
    if len(value) > _MAX_TEXT_CHARS:
        value = value[:_MAX_TEXT_CHARS]
    if any(ord(character) < 32 and character not in "\t\r\n" for character in value):
        raise ValueError("Retrieval text contains invalid control characters.")
    return tuple(
        token.lower()
        for token in itertools.islice(_TOKEN_RE.findall(value), _MAX_TOKENS_PER_TEXT)
    )


def _unit_score(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(numeric, 1.0))


@dataclass(frozen=True)
class RetrievalCandidate:
    candidate_id: str
    text: str
    source_id: str
    dense_score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value, maximum in (
            ("candidate_id", self.candidate_id, 500),
            ("source_id", self.source_id, 500),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > maximum
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
            ):
                raise ValueError(f"{label} is invalid.")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string.")
        if len(self.text) > _MAX_TEXT_CHARS:
            raise ValueError("text exceeds the retrieval candidate limit.")
        object.__setattr__(self, "dense_score", _unit_score(self.dense_score))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")


@dataclass(frozen=True)
class RankedCandidate:
    candidate: RetrievalCandidate
    rank: int
    score: float
    components: Mapping[str, float]


@dataclass(frozen=True)
class SparseDocument:
    document_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _bounded_candidates(values: Iterable[RetrievalCandidate]) -> list[RetrievalCandidate]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("candidates must be an iterable of retrieval candidates.")
    try:
        candidates = list(itertools.islice(iter(values), _MAX_CANDIDATES + 1))
    except Exception as exc:
        raise ValueError("candidates must be safely iterable.") from exc
    if len(candidates) > _MAX_CANDIDATES:
        raise ValueError(f"At most {_MAX_CANDIDATES} candidates may be ranked.")
    deduplicated: dict[str, RetrievalCandidate] = {}
    for candidate in candidates:
        if not isinstance(candidate, RetrievalCandidate):
            raise ValueError("Every candidate must be a RetrievalCandidate.")
        previous = deduplicated.get(candidate.candidate_id)
        if previous is None or candidate.dense_score > previous.dense_score:
            deduplicated[candidate.candidate_id] = candidate
    return list(deduplicated.values())


def _minmax(scores: Mapping[str, float]) -> dict[str, float]:
    finite = {key: float(value) for key, value in scores.items() if math.isfinite(float(value))}
    if not finite:
        return {}
    lower = min(finite.values())
    upper = max(finite.values())
    if math.isclose(lower, upper):
        return {key: (1.0 if upper > 0 else 0.0) for key in finite}
    span = upper - lower
    return {key: (value - lower) / span for key, value in finite.items()}


def bm25_scores(
    query: str,
    candidates: Iterable[RetrievalCandidate],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    query_tokens = tokenize(query)
    values = _bounded_candidates(candidates)
    if not query_tokens or not values:
        return {candidate.candidate_id: 0.0 for candidate in values}
    k1_value = _finite(k1, "k1", minimum=0.01, maximum=10.0)
    b_value = _finite(b, "b", minimum=0.0, maximum=1.0)
    document_tokens = {candidate.candidate_id: tokenize(candidate.text) for candidate in values}
    lengths = [len(tokens) for tokens in document_tokens.values()]
    average_length = sum(lengths) / max(len(lengths), 1)
    frequencies: Counter[str] = Counter()
    for tokens in document_tokens.values():
        frequencies.update(set(tokens))
    query_counts = Counter(query_tokens)
    total_documents = len(values)
    output: dict[str, float] = {}
    for candidate in values:
        tokens = document_tokens[candidate.candidate_id]
        term_counts = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for term, query_frequency in query_counts.items():
            document_frequency = frequencies.get(term, 0)
            if document_frequency <= 0:
                continue
            inverse_document_frequency = math.log(
                1.0 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            frequency = term_counts.get(term, 0)
            if frequency <= 0:
                continue
            normalizer = frequency + k1_value * (
                1.0 - b_value + b_value * length / max(average_length, 1.0)
            )
            score += (
                inverse_document_frequency
                * frequency
                * (k1_value + 1.0)
                / normalizer
                * (1.0 + math.log1p(query_frequency - 1))
            )
        output[candidate.candidate_id] = score
    return output


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    weights: Optional[Sequence[float]] = None,
    constant: int = 60,
) -> dict[str, float]:
    fusion_constant = _exact_int(constant, "constant", minimum=1, maximum=10_000)
    if len(rankings) > 32:
        raise ValueError("At most 32 rankings may be fused.")
    selected_weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(selected_weights) != len(rankings):
        raise ValueError("weights must match rankings.")
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking, raw_weight in zip(rankings, selected_weights):
        weight = _finite(raw_weight, "weight", minimum=0.0, maximum=1000.0)
        seen: set[str] = set()
        for rank, candidate_id in enumerate(itertools.islice(ranking, _MAX_CANDIDATES), start=1):
            if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            scores[candidate_id] += weight / (fusion_constant + rank)
    return dict(scores)


def weighted_score_fusion(
    score_maps: Sequence[Mapping[str, float]],
    *,
    weights: Optional[Sequence[float]] = None,
) -> dict[str, float]:
    if len(score_maps) > 32:
        raise ValueError("At most 32 score maps may be fused.")
    selected_weights = list(weights) if weights is not None else [1.0] * len(score_maps)
    if len(selected_weights) != len(score_maps):
        raise ValueError("weights must match score maps.")
    output: defaultdict[str, float] = defaultdict(float)
    total_weight = 0.0
    for scores, raw_weight in zip(score_maps, selected_weights):
        weight = _finite(raw_weight, "weight", minimum=0.0, maximum=1000.0)
        total_weight += weight
        for candidate_id, score in _minmax(scores).items():
            output[candidate_id] += weight * score
    if total_weight <= 0:
        return {candidate_id: 0.0 for candidate_id in output}
    return {candidate_id: score / total_weight for candidate_id, score in output.items()}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _mmr_order(
    candidates: Sequence[RetrievalCandidate],
    relevance: Mapping[str, float],
    *,
    limit: int,
    lambda_mult: float,
    per_source_limit: int,
) -> list[RetrievalCandidate]:
    requested = _exact_int(limit, "limit", minimum=1, maximum=_MAX_CANDIDATES)
    diversity = _finite(lambda_mult, "lambda_mult", minimum=0.0, maximum=1.0)
    source_limit = _exact_int(
        per_source_limit,
        "per_source_limit",
        minimum=1,
        maximum=_MAX_CANDIDATES,
    )
    remaining = list(candidates)
    token_sets = {candidate.candidate_id: set(tokenize(candidate.text)) for candidate in remaining}
    selected: list[RetrievalCandidate] = []
    source_counts: Counter[str] = Counter()
    while remaining and len(selected) < requested:
        best: Optional[RetrievalCandidate] = None
        best_value = -float("inf")
        for candidate in remaining:
            if source_counts[candidate.source_id] >= source_limit:
                continue
            redundancy = max(
                (_jaccard(token_sets[candidate.candidate_id], token_sets[item.candidate_id]) for item in selected),
                default=0.0,
            )
            value = diversity * relevance.get(candidate.candidate_id, 0.0) - (1.0 - diversity) * redundancy
            if best is None or value > best_value or (
                math.isclose(value, best_value) and candidate.candidate_id < best.candidate_id
            ):
                best = candidate
                best_value = value
        if best is None:
            break
        selected.append(best)
        source_counts[best.source_id] += 1
        remaining.remove(best)
    return selected


Reranker = Callable[[str, Sequence[RetrievalCandidate]], Mapping[str, float]]


def rank_candidates(
    query: str,
    candidates: Iterable[RetrievalCandidate],
    *,
    mode: str = "hybrid",
    limit: int = 5,
    reranker: Optional[Reranker] = None,
    diversity_lambda: float = 0.82,
    per_source_limit: int = 3,
) -> list[RankedCandidate]:
    query_text = query.strip() if isinstance(query, str) else ""
    if not query_text:
        return []
    values = _bounded_candidates(candidates)
    requested = _exact_int(limit, "limit", minimum=1, maximum=_MAX_CANDIDATES)
    if mode not in {"dense", "lexical", "hybrid"}:
        raise ValueError("mode must be dense, lexical, or hybrid.")
    dense = {candidate.candidate_id: candidate.dense_score for candidate in values}
    lexical = bm25_scores(query_text, values)
    dense_ranking = sorted(dense, key=lambda item: (-dense[item], item))
    lexical_ranking = sorted(lexical, key=lambda item: (-lexical[item], item))
    rrf = reciprocal_rank_fusion([dense_ranking, lexical_ranking], weights=[1.0, 1.0])
    weighted = weighted_score_fusion([dense, lexical], weights=[0.55, 0.45])
    if mode == "dense":
        base = _minmax(dense)
    elif mode == "lexical":
        base = _minmax(lexical)
    else:
        normalized_rrf = _minmax(rrf)
        all_ids = set(normalized_rrf) | set(weighted)
        base = {
            candidate_id: 0.55 * normalized_rrf.get(candidate_id, 0.0)
            + 0.45 * weighted.get(candidate_id, 0.0)
            for candidate_id in all_ids
        }
    reranker_scores: Mapping[str, float] = {}
    if reranker is not None:
        try:
            raw_scores = reranker(query_text, values)
            if isinstance(raw_scores, Mapping):
                reranker_scores = _minmax(
                    {
                        candidate_id: float(score)
                        for candidate_id, score in raw_scores.items()
                        if candidate_id in base and not isinstance(score, bool) and math.isfinite(float(score))
                    }
                )
        except Exception:
            reranker_scores = {}
    final = {
        candidate.candidate_id: (
            0.72 * base.get(candidate.candidate_id, 0.0)
            + 0.28 * reranker_scores.get(candidate.candidate_id, 0.0)
            if reranker_scores
            else base.get(candidate.candidate_id, 0.0)
        )
        for candidate in values
    }
    ordered = sorted(values, key=lambda item: (-final.get(item.candidate_id, 0.0), item.candidate_id))
    diversified = _mmr_order(
        ordered,
        final,
        limit=requested,
        lambda_mult=diversity_lambda,
        per_source_limit=per_source_limit,
    )
    return [
        RankedCandidate(
            candidate=candidate,
            rank=index,
            score=max(0.0, min(final.get(candidate.candidate_id, 0.0), 1.0)),
            components={
                "dense": dense.get(candidate.candidate_id, 0.0),
                "lexical": lexical.get(candidate.candidate_id, 0.0),
                "rrf": rrf.get(candidate.candidate_id, 0.0),
                "reranker": reranker_scores.get(candidate.candidate_id, 0.0),
            },
        )
        for index, candidate in enumerate(diversified, start=1)
    ]


class BM25Index:
    """Deterministic in-memory BM25 baseline for offline benchmark datasets."""

    def __init__(
        self,
        documents: Iterable[SparseDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        maximum_documents: int = 100_000,
    ) -> None:
        maximum = _exact_int(maximum_documents, "maximum_documents", minimum=1, maximum=1_000_000)
        if isinstance(documents, (str, bytes, bytearray)):
            raise ValueError("documents must be an iterable of SparseDocument values.")
        values = list(itertools.islice(iter(documents), maximum + 1))
        if len(values) > maximum:
            raise ValueError("The sparse corpus exceeds maximum_documents.")
        self.k1 = _finite(k1, "k1", minimum=0.01, maximum=10.0)
        self.b = _finite(b, "b", minimum=0.0, maximum=1.0)
        self.documents: dict[str, SparseDocument] = {}
        self.tokens: dict[str, tuple[str, ...]] = {}
        self.term_counts: dict[str, Counter[str]] = {}
        self.document_frequency: Counter[str] = Counter()
        for document in values:
            if not isinstance(document, SparseDocument):
                raise ValueError("Every document must be a SparseDocument.")
            if document.document_id in self.documents:
                raise ValueError("Sparse document IDs must be unique.")
            document_tokens = tokenize(document.text)
            self.documents[document.document_id] = document
            self.tokens[document.document_id] = document_tokens
            counts = Counter(document_tokens)
            self.term_counts[document.document_id] = counts
            self.document_frequency.update(counts)
        self.average_length = (
            sum(len(value) for value in self.tokens.values()) / max(len(self.tokens), 1)
        )

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        requested = _exact_int(top_k, "top_k", minimum=1, maximum=10_000)
        query_counts = Counter(tokenize(query))
        total_documents = len(self.documents)
        if not query_counts or total_documents == 0:
            return []
        heap: list[tuple[float, str]] = []
        for document_id, counts in self.term_counts.items():
            length = len(self.tokens[document_id])
            score = 0.0
            for term, query_frequency in query_counts.items():
                frequency = counts.get(term, 0)
                if frequency <= 0:
                    continue
                document_frequency = self.document_frequency.get(term, 0)
                inverse_document_frequency = math.log(
                    1.0 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normalizer = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self.average_length, 1.0)
                )
                score += inverse_document_frequency * frequency * (self.k1 + 1.0) / normalizer
                score *= 1.0 + math.log1p(query_frequency - 1)
            if score <= 0:
                continue
            item = (score, document_id)
            if len(heap) < requested:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        return [(document_id, score) for score, document_id in sorted(heap, reverse=True)]
''', encoding="utf-8")

Path("tools/reranking.py").write_text(r'''"""Safe reranker adapters for hybrid retrieval."""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Mapping, Optional, Sequence

from tools.hybrid_retrieval import RetrievalCandidate, bm25_scores, tokenize


class HeuristicReranker:
    """Dependency-free scientific-text reranker used as the safe default."""

    def __call__(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> Mapping[str, float]:
        lexical = bm25_scores(query, candidates)
        query_tokens = set(tokenize(query))
        output: dict[str, float] = {}
        for candidate in candidates:
            document_tokens = set(tokenize(candidate.text))
            overlap = len(query_tokens & document_tokens) / max(len(query_tokens), 1)
            phrase = 1.0 if query.lower() in candidate.text.lower() else 0.0
            section = candidate.metadata.get("section_title")
            section_overlap = 0.0
            if isinstance(section, str):
                section_tokens = set(tokenize(section))
                section_overlap = len(query_tokens & section_tokens) / max(len(query_tokens), 1)
            output[candidate.candidate_id] = (
                0.45 * lexical.get(candidate.candidate_id, 0.0)
                + 0.30 * overlap
                + 0.15 * section_overlap
                + 0.10 * phrase
            )
        return output


class CrossEncoderReranker:
    """Lazily loaded optional cross-encoder with deterministic safe fallback."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 16,
        maximum_candidates: int = 50,
        model_factory: Optional[Callable[[str], Any]] = None,
        fallback: Optional[Callable[[str, Sequence[RetrievalCandidate]], Mapping[str, float]]] = None,
    ) -> None:
        if (
            not isinstance(model_name, str)
            or not model_name
            or len(model_name) > 300
            or model_name != model_name.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in model_name)
        ):
            raise ValueError("model_name must be a canonical bounded string.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be an integer between 1 and 256.")
        if (
            isinstance(maximum_candidates, bool)
            or not isinstance(maximum_candidates, int)
            or not 1 <= maximum_candidates <= 500
        ):
            raise ValueError("maximum_candidates must be an integer between 1 and 500.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.maximum_candidates = maximum_candidates
        self._factory = model_factory
        self._fallback = fallback or HeuristicReranker()
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                if self._factory is not None:
                    self._model = self._factory(self.model_name)
                else:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)
            return self._model

    def __call__(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> Mapping[str, float]:
        selected = list(candidates[: self.maximum_candidates])
        if not selected:
            return {}
        try:
            model = self._load()
            raw_scores = model.predict(
                [(query, candidate.text) for candidate in selected],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            values = list(raw_scores)
            if len(values) != len(selected):
                raise RuntimeError("Cross-encoder returned an invalid score count.")
            finite = []
            for value in values:
                if isinstance(value, bool):
                    raise RuntimeError("Cross-encoder returned a boolean score.")
                score = float(value)
                if not math.isfinite(score):
                    raise RuntimeError("Cross-encoder returned a non-finite score.")
                finite.append(score)
            lower = min(finite)
            upper = max(finite)
            if math.isclose(lower, upper):
                normalized = [1.0] * len(finite)
            else:
                normalized = [(value - lower) / (upper - lower) for value in finite]
            return {
                candidate.candidate_id: score
                for candidate, score in zip(selected, normalized)
            }
        except Exception:
            return self._fallback(query, selected)


def build_reranker(
    name: str,
    *,
    cross_encoder_model: Optional[str] = None,
    model_factory: Optional[Callable[[str], Any]] = None,
):
    if name == "none":
        return None
    if name == "heuristic":
        return HeuristicReranker()
    if name == "cross_encoder":
        if not cross_encoder_model:
            raise ValueError("cross_encoder_model is required for cross_encoder reranking.")
        return CrossEncoderReranker(
            cross_encoder_model,
            model_factory=model_factory,
        )
    raise ValueError("reranker must be none, heuristic, or cross_encoder.")
''', encoding="utf-8")

Path("evaluation").mkdir(exist_ok=True)
Path("evaluation/__init__.py").write_text(
    '"""Dataset, metric, and experiment support for RigorousRAG."""\n',
    encoding="utf-8",
)
Path("evaluation/datasets.py").write_text(r'''"""Bounded normalized benchmark dataset loaders."""

from __future__ import annotations

import csv
import itertools
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _exact_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _safe_text(value: Any, label: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if len(rendered) > maximum or any(ord(character) < 32 and character not in "\t\r\n" for character in rendered):
        raise ValueError(f"{label} is invalid or too long.")
    if not rendered and not allow_empty:
        raise ValueError(f"{label} may not be empty.")
    return rendered


def _safe_file(path: Path, *, maximum_bytes: int) -> bytes:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or bool(
            int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT
        ):
            raise ValueError("Benchmark paths may not contain links or reparse points.")
    metadata = absolute.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Benchmark inputs must be regular files.")
    if metadata.st_size > maximum_bytes:
        raise ValueError("Benchmark input exceeds the byte limit.")
    data = absolute.read_bytes()
    if len(data) != metadata.st_size or len(data) > maximum_bytes:
        raise ValueError("Benchmark input changed during the read.")
    return data


@dataclass(frozen=True)
class BenchmarkDocument:
    document_id: str
    text: str
    title: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _safe_text(self.document_id, "document_id", 500))
        object.__setattr__(self, "text", _safe_text(self.text, "text", 2_000_000, allow_empty=True))
        object.__setattr__(self, "title", _safe_text(self.title, "title", 10_000, allow_empty=True))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    text: str
    relevant: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _safe_text(self.query_id, "query_id", 500))
        object.__setattr__(self, "text", _safe_text(self.text, "query text", 100_000))
        if not isinstance(self.relevant, Mapping):
            raise ValueError("relevant must be a mapping.")
        cleaned: dict[str, float] = {}
        for raw_id, raw_score in itertools.islice(self.relevant.items(), 100_000):
            document_id = _safe_text(raw_id, "relevant document ID", 500)
            if isinstance(raw_score, bool):
                raise ValueError("Relevance scores must be numeric.")
            score = float(raw_score)
            if not 0.0 <= score <= 1_000_000.0:
                raise ValueError("Relevance scores must be finite and nonnegative.")
            cleaned[document_id] = score
        object.__setattr__(self, "relevant", cleaned)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    documents: Mapping[str, BenchmarkDocument]
    queries: tuple[BenchmarkQuery, ...]


def _json_lines(path: Path, *, maximum_bytes: int, maximum_rows: int) -> list[dict[str, Any]]:
    data = _safe_file(path, maximum_bytes=maximum_bytes)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Benchmark JSONL must be UTF-8.") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(rows) >= maximum_rows:
            raise ValueError("Benchmark JSONL exceeds the row limit.")
        try:
            value = json.loads(
                line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"Non-standard constant {constant}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid benchmark JSON on line {line_number}.") from exc
        if not isinstance(value, dict):
            raise ValueError("Every benchmark JSONL row must be an object.")
        rows.append(value)
    return rows


def load_beir_dataset(
    root: str | os.PathLike[str],
    *,
    split: str = "test",
    maximum_documents: int = 100_000,
    maximum_queries: int = 50_000,
    maximum_file_bytes: int = 1_000_000_000,
) -> BenchmarkDataset:
    document_limit = _exact_int(maximum_documents, "maximum_documents", 1, 1_000_000)
    query_limit = _exact_int(maximum_queries, "maximum_queries", 1, 1_000_000)
    byte_limit = _exact_int(maximum_file_bytes, "maximum_file_bytes", 1, 10_000_000_000)
    split_name = _safe_text(split, "split", 100)
    base = Path(root)
    corpus_rows = _json_lines(base / "corpus.jsonl", maximum_bytes=byte_limit, maximum_rows=document_limit)
    query_rows = _json_lines(base / "queries.jsonl", maximum_bytes=byte_limit, maximum_rows=query_limit)
    documents: dict[str, BenchmarkDocument] = {}
    for row in corpus_rows:
        document_id = row.get("_id", row.get("id"))
        document = BenchmarkDocument(
            document_id=document_id,
            title=row.get("title") if isinstance(row.get("title"), str) else "",
            text=row.get("text") if isinstance(row.get("text"), str) else "",
            metadata={key: value for key, value in row.items() if key not in {"_id", "id", "title", "text"}},
        )
        if document.document_id in documents:
            raise ValueError("Benchmark document IDs must be unique.")
        documents[document.document_id] = document
    query_texts: dict[str, str] = {}
    for row in query_rows:
        query_id = _safe_text(row.get("_id", row.get("id")), "query_id", 500)
        text = _safe_text(row.get("text"), "query text", 100_000)
        if query_id in query_texts:
            raise ValueError("Benchmark query IDs must be unique.")
        query_texts[query_id] = text
    qrels_candidates = [base / "qrels" / f"{split_name}.tsv", base / "qrels" / f"{split_name}.txt"]
    qrels_path = next((path for path in qrels_candidates if path.exists()), None)
    if qrels_path is None:
        raise ValueError("The requested BEIR qrels split does not exist.")
    qrels_data = _safe_file(qrels_path, maximum_bytes=byte_limit)
    try:
        qrels_text = qrels_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("BEIR qrels must be UTF-8.") from exc
    relevant: dict[str, dict[str, float]] = {query_id: {} for query_id in query_texts}
    reader = csv.reader(qrels_text.splitlines(), delimiter="\t")
    for row_number, row in enumerate(reader, start=1):
        if not row:
            continue
        if row_number == 1 and any(value.lower() in {"query-id", "corpus-id", "score"} for value in row):
            continue
        if len(row) < 3:
            raise ValueError("Every qrels row must contain query, document, and score.")
        query_id = _safe_text(row[0], "qrels query ID", 500)
        document_id = _safe_text(row[1], "qrels document ID", 500)
        try:
            score = float(row[2])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Qrels scores must be numeric.") from exc
        if not score >= 0 or not score < float("inf"):
            raise ValueError("Qrels scores must be finite and nonnegative.")
        if query_id in relevant and document_id in documents and score > 0:
            relevant[query_id][document_id] = score
    queries = tuple(
        BenchmarkQuery(query_id=query_id, text=text, relevant=relevant.get(query_id, {}))
        for query_id, text in query_texts.items()
    )
    return BenchmarkDataset(name=base.name or "beir", documents=documents, queries=queries)
''', encoding="utf-8")

Path("evaluation/metrics.py").write_text(r'''"""Deterministic retrieval, ranking, citation, and aggregate metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def _ranked(values: Iterable[str], maximum: int = 100_000) -> list[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("ranked IDs must be an iterable of strings.")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if len(result) >= maximum:
            raise ValueError("ranked IDs exceed the metric limit.")
        if not isinstance(value, str) or not value:
            raise ValueError("ranked IDs must be non-empty strings.")
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _qrels(values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError("qrels must be a mapping.")
    result: dict[str, float] = {}
    for document_id, raw_score in values.items():
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("qrels IDs must be non-empty strings.")
        if isinstance(raw_score, bool):
            raise ValueError("qrels scores must be numeric.")
        score = float(raw_score)
        if not math.isfinite(score) or score < 0:
            raise ValueError("qrels scores must be finite and nonnegative.")
        result[document_id] = score
    return result


def precision_at_k(ranked: Iterable[str], relevant: Mapping[str, float], k: int) -> float:
    values = _ranked(ranked)
    qrels = _qrels(relevant)
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    selected = values[:k]
    return sum(1 for item in selected if qrels.get(item, 0.0) > 0) / k


def recall_at_k(ranked: Iterable[str], relevant: Mapping[str, float], k: int) -> float:
    values = _ranked(ranked)
    qrels = {item for item, score in _qrels(relevant).items() if score > 0}
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not qrels:
        return 0.0
    return len(set(values[:k]) & qrels) / len(qrels)


def reciprocal_rank(ranked: Iterable[str], relevant: Mapping[str, float], k: int | None = None) -> float:
    values = _ranked(ranked)
    qrels = _qrels(relevant)
    selected = values if k is None else values[:k]
    for index, item in enumerate(selected, start=1):
        if qrels.get(item, 0.0) > 0:
            return 1.0 / index
    return 0.0


def average_precision(ranked: Iterable[str], relevant: Mapping[str, float], k: int | None = None) -> float:
    values = _ranked(ranked)
    qrels = {item for item, score in _qrels(relevant).items() if score > 0}
    if not qrels:
        return 0.0
    selected = values if k is None else values[:k]
    hits = 0
    total = 0.0
    for index, item in enumerate(selected, start=1):
        if item in qrels:
            hits += 1
            total += hits / index
    return total / len(qrels)


def ndcg_at_k(ranked: Iterable[str], relevant: Mapping[str, float], k: int) -> float:
    values = _ranked(ranked)
    qrels = _qrels(relevant)
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    dcg = sum((2 ** qrels.get(item, 0.0) - 1) / math.log2(index + 1) for index, item in enumerate(values[:k], start=1))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** score - 1) / math.log2(index + 1) for index, score in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(
    ranked: Iterable[str],
    relevant: Mapping[str, float],
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    values = _ranked(ranked)
    qrels = _qrels(relevant)
    metrics: dict[str, float] = {
        "mrr": reciprocal_rank(values, qrels),
        "map": average_precision(values, qrels),
        "hit_rate": 1.0 if reciprocal_rank(values, qrels) > 0 else 0.0,
    }
    for cutoff in cutoffs:
        metrics[f"precision@{cutoff}"] = precision_at_k(values, qrels, cutoff)
        metrics[f"recall@{cutoff}"] = recall_at_k(values, qrels, cutoff)
        metrics[f"ndcg@{cutoff}"] = ndcg_at_k(values, qrels, cutoff)
    return metrics


def citation_metrics(predicted: Iterable[str], expected: Iterable[str]) -> dict[str, float]:
    predicted_set = set(_ranked(predicted))
    expected_set = set(_ranked(expected))
    overlap = predicted_set & expected_set
    precision = len(overlap) / len(predicted_set) if predicted_set else 0.0
    recall = len(overlap) / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"citation_precision": precision, "citation_recall": recall, "citation_f1": f1}


def aggregate_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Metric rows must be mappings.")
        for name, raw_value in row.items():
            if not isinstance(name, str) or isinstance(raw_value, bool):
                raise ValueError("Metric names and values are invalid.")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("Metric values must be finite.")
            totals[name] += value
            counts[name] += 1
    return {name: totals[name] / counts[name] for name in sorted(totals)}
''', encoding="utf-8")

Path("experiments").mkdir(exist_ok=True)
Path("experiments/__init__.py").write_text(
    '"""Reproducible experiment manifests and result persistence."""\n',
    encoding="utf-8",
)
Path("experiments/manifest.py").write_text(r'''"""Deterministic, resumable experiment matrices."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_ALLOWED_AXES = {
    "retrieval_mode",
    "reranker",
    "top_k",
    "candidate_pool",
    "diversity_lambda",
    "use_hyde",
    "use_multi_query",
    "embedding_model",
    "chunk_size",
    "chunk_overlap",
    "seed",
}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    parameters: Mapping[str, Any]
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _canonical(parameters: Mapping[str, Any]) -> str:
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping.")
    unknown = set(parameters) - _ALLOWED_AXES
    if unknown:
        raise ValueError(f"Unsupported experiment axes: {', '.join(sorted(unknown))}.")
    try:
        encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Experiment parameters must be canonical JSON values.") from exc
    if len(encoded) > 100_000:
        raise ValueError("Experiment parameters exceed the size limit.")
    return encoded


def make_experiment_spec(parameters: Mapping[str, Any], *, tags: Sequence[str] = ()) -> ExperimentSpec:
    encoded = _canonical(parameters)
    clean_tags: list[str] = []
    for tag in tags:
        if not isinstance(tag, str) or not tag or len(tag) > 100 or any(ord(char) < 32 or ord(char) == 127 for char in tag):
            raise ValueError("Experiment tags must be bounded non-empty strings.")
        clean_tags.append(tag)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    return ExperimentSpec(experiment_id=digest, parameters=dict(parameters), tags=tuple(clean_tags))


def build_matrix(axes: Mapping[str, Iterable[Any]], *, maximum_experiments: int = 10_000) -> tuple[ExperimentSpec, ...]:
    if not isinstance(axes, Mapping) or not axes:
        raise ValueError("axes must be a non-empty mapping.")
    if isinstance(maximum_experiments, bool) or not isinstance(maximum_experiments, int) or not 1 <= maximum_experiments <= 1_000_000:
        raise ValueError("maximum_experiments must be a positive bounded integer.")
    unknown = set(axes) - _ALLOWED_AXES
    if unknown:
        raise ValueError(f"Unsupported experiment axes: {', '.join(sorted(unknown))}.")
    names = sorted(axes)
    values: list[list[Any]] = []
    total = 1
    for name in names:
        raw_values = axes[name]
        if isinstance(raw_values, (str, bytes, bytearray)):
            raise ValueError(f"Axis {name} must be an iterable of values.")
        selected = list(itertools.islice(iter(raw_values), maximum_experiments + 1))
        if not selected:
            raise ValueError(f"Axis {name} may not be empty.")
        total *= len(selected)
        if total > maximum_experiments:
            raise ValueError("Experiment matrix exceeds maximum_experiments.")
        values.append(selected)
    return tuple(
        make_experiment_spec(dict(zip(names, combination)))
        for combination in itertools.product(*values)
    )


def write_manifest(path: str | os.PathLike[str], specs: Iterable[ExperimentSpec]) -> None:
    destination = Path(path)
    if destination.exists() and destination.is_symlink():
        raise ValueError("Manifest destination may not be a symbolic link.")
    rows = []
    for spec in specs:
        if not isinstance(spec, ExperimentSpec):
            raise ValueError("Every manifest row must be an ExperimentSpec.")
        rows.append(json.dumps(asdict(spec), sort_keys=True, allow_nan=False))
        if len(rows) > 1_000_000:
            raise ValueError("Manifest exceeds the row limit.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows) + ("\n" if rows else ""))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


class ResultStore:
    """One immutable JSON result per experiment for race-safe resume semantics."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(os.path.abspath(root))
        if self.root.exists() and self.root.is_symlink():
            raise ValueError("Result root may not be a symbolic link.")
        self.root.mkdir(parents=True, exist_ok=True)
        mode = self.root.stat(follow_symlinks=False).st_mode
        if not stat.S_ISDIR(mode):
            raise ValueError("Result root must be a directory.")

    def completed_ids(self) -> set[str]:
        completed: set[str] = set()
        for path in itertools.islice(sorted(self.root.glob("*.json")), 1_000_000):
            if path.is_symlink() or not path.is_file():
                continue
            stem = path.stem
            if len(stem) == 20 and all(character in "0123456789abcdef" for character in stem):
                completed.add(stem)
        return completed

    def pending(self, specs: Iterable[ExperimentSpec]) -> tuple[ExperimentSpec, ...]:
        completed = self.completed_ids()
        return tuple(spec for spec in specs if spec.experiment_id not in completed)

    def write(self, result: ExperimentResult) -> bool:
        if not isinstance(result, ExperimentResult):
            raise ValueError("result must be an ExperimentResult.")
        if len(result.experiment_id) != 20 or any(character not in "0123456789abcdef" for character in result.experiment_id):
            raise ValueError("experiment_id is invalid.")
        metrics: dict[str, float] = {}
        for name, raw_value in result.metrics.items():
            if not isinstance(name, str) or not name or len(name) > 200 or isinstance(raw_value, bool):
                raise ValueError("Result metrics are invalid.")
            value = float(raw_value)
            if not value == value or value in {float("inf"), -float("inf")}:
                raise ValueError("Result metrics must be finite.")
            metrics[name] = value
        payload = json.dumps(
            {"experiment_id": result.experiment_id, "metrics": metrics, "metadata": dict(result.metadata)},
            sort_keys=True,
            allow_nan=False,
        )
        destination = self.root / f"{result.experiment_id}.json"
        try:
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return True
        except FileExistsError:
            return False
''', encoding="utf-8")

Path("evaluation/runner.py").write_text(r'''"""Reusable benchmark execution over normalized datasets."""

from __future__ import annotations

import time
from typing import Callable, Iterable, Mapping, Sequence

from evaluation.datasets import BenchmarkDataset, BenchmarkQuery
from evaluation.metrics import aggregate_metrics, retrieval_metrics
from experiments.manifest import ExperimentResult, ExperimentSpec, ResultStore

Retriever = Callable[[BenchmarkQuery, Mapping[str, object]], Sequence[str]]


def run_benchmark(
    dataset: BenchmarkDataset,
    specs: Iterable[ExperimentSpec],
    retriever: Retriever,
    result_store: ResultStore,
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10),
) -> tuple[ExperimentResult, ...]:
    completed: list[ExperimentResult] = []
    for spec in result_store.pending(specs):
        rows = []
        latencies = []
        for query in dataset.queries:
            started = time.perf_counter()
            ranked = retriever(query, spec.parameters)
            latencies.append(time.perf_counter() - started)
            rows.append(retrieval_metrics(ranked, query.relevant, cutoffs=cutoffs))
        metrics = aggregate_metrics(rows)
        metrics["mean_query_latency_seconds"] = sum(latencies) / max(len(latencies), 1)
        result = ExperimentResult(
            experiment_id=spec.experiment_id,
            metrics=metrics,
            metadata={"dataset": dataset.name, "query_count": len(dataset.queries)},
        )
        result_store.write(result)
        completed.append(result)
    return tuple(completed)
''', encoding="utf-8")

Path("scripts/run_retrieval_benchmarks.py").write_text(r'''#!/usr/bin/env python3
"""Run a resumable BM25 baseline matrix over a BEIR-format dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.datasets import load_beir_dataset
from evaluation.runner import run_benchmark
from experiments.manifest import ResultStore, build_matrix, write_manifest
from tools.hybrid_retrieval import BM25Index, SparseDocument


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--maximum-documents", type=int, default=100_000)
    parser.add_argument("--maximum-queries", type=int, default=50_000)
    args = parser.parse_args()

    dataset = load_beir_dataset(
        args.dataset,
        split=args.split,
        maximum_documents=args.maximum_documents,
        maximum_queries=args.maximum_queries,
    )
    index = BM25Index(
        SparseDocument(
            document_id=document.document_id,
            text=f"{document.title}\n{document.text}",
            metadata=document.metadata,
        )
        for document in dataset.documents.values()
    )
    specs = build_matrix(
        {
            "retrieval_mode": ["lexical"],
            "reranker": ["none"],
            "top_k": args.top_k,
            "candidate_pool": args.top_k,
            "diversity_lambda": [1.0],
            "use_hyde": [False],
            "use_multi_query": [False],
            "seed": [0],
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output / "manifest.jsonl", specs)
    store = ResultStore(args.output / "results")

    def retrieve(query, parameters):
        top_k = int(parameters["top_k"])
        return [document_id for document_id, _score in index.search(query.text, top_k=top_k)]

    results = run_benchmark(dataset, specs, retrieve, store)
    print(f"completed={len(results)} pending={len(store.pending(specs))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''', encoding="utf-8")

# Wire hybrid ranking controls into uploaded-document retrieval.
replace_once(
    "tools/rag_tool.py",
    '''from tools.models import Citation\nfrom tools.rag import get_rag_layer\nfrom tools.security import normalize_owner_id\n''',
    '''from tools.hybrid_retrieval import RetrievalCandidate, rank_candidates\nfrom tools.models import Citation\nfrom tools.rag import get_rag_layer\nfrom tools.reranking import build_reranker\nfrom tools.security import normalize_owner_id\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''                "use_multi_query": {\n                    "type": "boolean",\n                    "description": "Generate a small number of alternative retrieval queries.",\n                    "default": False,\n                },\n''',
    '''                "use_multi_query": {\n                    "type": "boolean",\n                    "description": "Generate a small number of alternative retrieval queries.",\n                    "default": False,\n                },\n                "retrieval_mode": {\n                    "type": "string",\n                    "enum": ["dense", "lexical", "hybrid"],\n                    "description": "Dense, candidate-pool lexical, or fused hybrid ranking.",\n                    "default": "hybrid",\n                },\n                "reranker": {\n                    "type": "string",\n                    "enum": ["none", "heuristic"],\n                    "description": "Optional bounded second-stage reranker.",\n                    "default": "heuristic",\n                },\n                "candidate_pool": {\n                    "type": "integer",\n                    "minimum": 1,\n                    "maximum": 50,\n                    "description": "Dense candidate pool before fusion and diversity selection.",\n                    "default": 20,\n                },\n                "diversity_lambda": {\n                    "type": "number",\n                    "minimum": 0.0,\n                    "maximum": 1.0,\n                    "description": "MMR relevance/diversity trade-off; 1.0 disables redundancy penalty.",\n                    "default": 0.82,\n                },\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''def _safe_attr(value: Any, name: str, default: Any = None) -> Any:\n''',
    '''def _choice(value: Any, label: str, allowed: set[str]) -> str:\n    if not isinstance(value, str) or value not in allowed:\n        raise ValueError(f"{label} must be one of: {', '.join(sorted(allowed))}.")\n    return value\n\n\ndef _unit_float(value: Any, label: str) -> float:\n    if isinstance(value, bool):\n        raise ValueError(f"{label} must be numeric.")\n    try:\n        numeric = float(value)\n    except (TypeError, ValueError, OverflowError) as exc:\n        raise ValueError(f"{label} must be numeric.") from exc\n    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:\n        raise ValueError(f"{label} must be between 0 and 1.")\n    return numeric\n\n\ndef _safe_attr(value: Any, name: str, default: Any = None) -> Any:\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''    expansion_model: str = "gpt-4o-mini",\n    n_results: int = 5,\n) -> List[Citation]:\n''',
    '''    expansion_model: str = "gpt-4o-mini",\n    n_results: int = 5,\n    retrieval_mode: str = "hybrid",\n    reranker: str = "heuristic",\n    candidate_pool: int = 20,\n    diversity_lambda: float = 0.82,\n) -> List[Citation]:\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''    requested = _integer(\n        n_results,\n        "n_results",\n        minimum=1,\n        maximum=_MAX_CITATIONS,\n    )\n\n    rag = get_rag_layer()\n''',
    '''    requested = _integer(\n        n_results,\n        "n_results",\n        minimum=1,\n        maximum=_MAX_CITATIONS,\n    )\n    mode = _choice(retrieval_mode, "retrieval_mode", {"dense", "lexical", "hybrid"})\n    reranker_name = _choice(reranker, "reranker", {"none", "heuristic"})\n    pool = _integer(candidate_pool, "candidate_pool", minimum=1, maximum=_MAX_CITATIONS)\n    pool = max(requested, pool)\n    diversity = _unit_float(diversity_lambda, "diversity_lambda")\n\n    rag = get_rag_layer()\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''    chunks = rag.query(\n        retrieval_query,\n        n_results=requested,\n''',
    '''    chunks = rag.query(\n        retrieval_query,\n        n_results=pool,\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''    citations: List[Citation] = []\n    for chunk in _bounded_chunks(chunks, requested):\n''',
    '''    candidate_chunks = _bounded_chunks(chunks, pool)\n    chunk_map: dict[str, Any] = {}\n    ranking_inputs: List[RetrievalCandidate] = []\n    for chunk in candidate_chunks:\n        raw_chunk_id = _safe_attr(chunk, "id", "")\n        raw_text = _safe_attr(chunk, "text", "")\n        metadata = _metadata(_safe_attr(chunk, "metadata", {}))\n        if not isinstance(raw_chunk_id, str) or not isinstance(raw_text, str):\n            continue\n        chunk_id = raw_chunk_id.strip()\n        source_id = metadata.get("doc_id")\n        if not isinstance(source_id, str):\n            continue\n        try:\n            candidate = RetrievalCandidate(\n                candidate_id=chunk_id,\n                text=raw_text[:100_000],\n                source_id=source_id,\n                dense_score=_finite_score(_safe_attr(chunk, "score", 0.0)),\n                metadata=metadata,\n            )\n        except ValueError:\n            continue\n        chunk_map[chunk_id] = chunk\n        ranking_inputs.append(candidate)\n    ranked = rank_candidates(\n        retrieval_query,\n        ranking_inputs,\n        mode=mode,\n        limit=requested,\n        reranker=build_reranker(reranker_name),\n        diversity_lambda=diversity,\n        per_source_limit=max(1, min(requested, 3)),\n    )\n    ordered_chunks = [\n        chunk_map[item.candidate.candidate_id]\n        for item in ranked\n        if item.candidate.candidate_id in chunk_map\n    ]\n    ranking_scores = {item.candidate.candidate_id: item for item in ranked}\n\n    citations: List[Citation] = []\n    for chunk in ordered_chunks:\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''                    "relevance": round(\n                        _finite_score(_safe_attr(chunk, "score", 0.0)),\n                        6,\n                    ),\n''',
    '''                    "relevance": round(\n                        ranking_scores.get(chunk_id).score\n                        if chunk_id in ranking_scores\n                        else _finite_score(_safe_attr(chunk, "score", 0.0)),\n                        6,\n                    ),\n                    "retrieval_mode": mode,\n                    "reranker": reranker_name,\n                    "rank_components": (\n                        dict(ranking_scores[chunk_id].components)\n                        if chunk_id in ranking_scores\n                        else {}\n                    ),\n''',
)
