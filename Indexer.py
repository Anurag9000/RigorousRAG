"""Bounded sparse scientific-text TF-IDF index."""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from Crawler import Page

STOP_WORDS: set[str] = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "during", "each", "for", "from",
    "further", "had", "has", "have", "having", "how", "if", "in", "into",
    "is", "it", "its", "may", "more", "most", "no", "not", "of", "on",
    "only", "or", "other", "our", "out", "over", "same", "should", "so",
    "some", "such", "than", "that", "the", "their", "then", "there", "these",
    "they", "this", "those", "through", "to", "under", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "why",
    "will", "with", "would", "you", "your",
}

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*", flags=re.UNICODE)
_MAX_TEXT_CHARS = 5_000_000
_MAX_TOKENS_PER_DOCUMENT = 1_000_000
_MAX_TOKEN_CHARS = 500
_MAX_DOCUMENTS = 100_000
_MAX_TERMS = 1_000_000
_MAX_POSTINGS = 5_000_000
_MAX_URL_CHARS = 4096
_MAX_TITLE_CHARS = 500
_MAX_SNIPPET_CHARS = 4000


def _safe_text(value: object, *, limit: int, default: str = "") -> str:
    if isinstance(value, str):
        text = value
    elif value is None:
        text = default
    else:
        try:
            text = str(value)
        except Exception:
            text = default
    return text[: max(int(limit), 0)]


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return numeric


def _bounded_nonnegative_int(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer.")
    if not 0 <= numeric <= maximum:
        raise ValueError(f"{label} is outside the supported range.")
    return numeric


def _mapping_items(
    value: object,
    *,
    label: str,
    maximum: int,
) -> List[Tuple[object, object]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    try:
        items = list(itertools.islice(value.items(), maximum + 1))
    except Exception as exc:
        raise ValueError(f"{label} must be a safely iterable object.") from exc
    if len(items) > maximum:
        raise ValueError(f"{label} contains too many entries.")
    return items


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    tokens: List[str] = []
    for match in TOKEN_PATTERN.finditer(text[:_MAX_TEXT_CHARS]):
        token = match.group(0).casefold().strip("-./")[:_MAX_TOKEN_CHARS]
        if not token or token in STOP_WORDS:
            continue
        if (
            len(token) == 1
            and not token.isdigit()
            and token not in {"x", "y", "z", "r", "p", "q"}
        ):
            continue
        tokens.append(token)
        if len(tokens) >= _MAX_TOKENS_PER_DOCUMENT:
            break
    return tokens


def build_snippet(text: str, max_words: int = 40) -> str:
    words_limit = _bounded_nonnegative_int(max_words, "max_words", 10_000)
    if not isinstance(text, str) or words_limit == 0:
        return ""
    words = text[:_MAX_TEXT_CHARS].split()
    snippet = " ".join(words[:words_limit])[:_MAX_SNIPPET_CHARS]
    return f"{snippet}…" if len(words) > words_limit else snippet


@dataclass
class DocumentMetadata:
    title: str
    snippet: str
    length: int

    def __post_init__(self) -> None:
        self.title = _safe_text(
            self.title,
            limit=_MAX_TITLE_CHARS,
            default="Untitled",
        ).strip() or "Untitled"
        self.snippet = _safe_text(self.snippet, limit=_MAX_SNIPPET_CHARS)
        self.length = _bounded_nonnegative_int(
            self.length,
            "document length",
            _MAX_TOKENS_PER_DOCUMENT,
        )


class InvertedIndex:
    """Sparse log-TF/IDF index with cosine-normalised document vectors."""

    SCHEMA_VERSION = 2

    def __init__(self) -> None:
        self.documents: Dict[str, DocumentMetadata] = {}
        self.index: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.idf: Dict[str, float] = {}
        self.doc_norms: Dict[str, float] = {}

    def clear(self) -> None:
        self.documents.clear()
        self.index = defaultdict(dict)
        self.idf.clear()
        self.doc_norms.clear()

    def build(self, pages: Dict[str, Page]) -> None:
        """Build a complete replacement generation and publish it only on success."""

        page_items = _mapping_items(
            pages,
            label="pages",
            maximum=_MAX_DOCUMENTS,
        )
        term_document_frequency: Counter[str] = Counter()
        document_term_frequency: Dict[str, Counter[str]] = {}
        documents: Dict[str, DocumentMetadata] = {}

        for raw_url, page in page_items:
            if not isinstance(raw_url, str) or not 0 < len(raw_url) <= _MAX_URL_CHARS:
                continue
            if not isinstance(page, Page):
                continue
            body_text = page.text if isinstance(page.text, str) else ""
            title_text = page.title if isinstance(page.title, str) else "Untitled"
            body_tokens = tokenize(body_text)
            title_tokens = tokenize(title_text)
            if not body_tokens and not title_tokens:
                continue
            frequencies = Counter(body_tokens)
            for token in title_tokens:
                frequencies[token] += 2
            document_term_frequency[raw_url] = frequencies
            term_document_frequency.update(frequencies.keys())
            documents[raw_url] = DocumentMetadata(
                title=title_text,
                snippet=build_snippet(body_text),
                length=len(body_tokens),
            )

        total_documents = len(document_term_frequency)
        if total_documents == 0:
            self.documents = {}
            self.index = defaultdict(dict)
            self.idf = {}
            self.doc_norms = {}
            return
        if len(term_document_frequency) > _MAX_TERMS:
            raise ValueError(f"An index may contain at most {_MAX_TERMS} unique terms.")

        idf = {
            term: math.log((1 + total_documents) / (1 + frequency)) + 1.0
            for term, frequency in term_document_frequency.items()
        }
        postings: Dict[str, Dict[str, float]] = defaultdict(dict)
        doc_norms: Dict[str, float] = {}
        total_postings = 0
        for url, frequencies in document_term_frequency.items():
            norm_squared = 0.0
            for term, frequency in frequencies.items():
                total_postings += 1
                if total_postings > _MAX_POSTINGS:
                    raise ValueError(
                        f"An index may contain at most {_MAX_POSTINGS} postings."
                    )
                weight = (1.0 + math.log(frequency)) * idf[term]
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError("Computed posting weight is invalid.")
                postings[term][url] = weight
                norm_squared += weight * weight
            if not math.isfinite(norm_squared) or norm_squared <= 0:
                raise ValueError("Computed document norm is invalid.")
            doc_norms[url] = math.sqrt(norm_squared)

        self.documents = documents
        self.index = defaultdict(dict, {term: dict(values) for term, values in postings.items()})
        self.idf = idf
        self.doc_norms = doc_norms

    def to_dict(self) -> Dict[str, object]:
        document_items = _mapping_items(
            self.documents,
            label="index documents",
            maximum=_MAX_DOCUMENTS,
        )
        term_items = _mapping_items(
            self.index,
            label="index postings",
            maximum=_MAX_TERMS,
        )
        idf_items = _mapping_items(
            self.idf,
            label="index IDF values",
            maximum=_MAX_TERMS,
        )
        total_postings = 0
        serialized_documents: Dict[str, Dict[str, object]] = {}
        for url, metadata in document_items:
            if not isinstance(url, str) or not isinstance(metadata, DocumentMetadata):
                raise ValueError("Index documents contain invalid entries.")
            serialized_documents[url] = {
                "title": metadata.title,
                "snippet": metadata.snippet,
                "length": metadata.length,
            }
        serialized_idf: Dict[str, float] = {}
        for term, value in idf_items:
            if not isinstance(term, str):
                raise ValueError("Index terms must be strings.")
            numeric = _finite_nonnegative(value, "IDF value")
            if numeric <= 0:
                raise ValueError("IDF values must be positive.")
            serialized_idf[term] = numeric
        serialized_postings: Dict[str, Dict[str, float]] = {}
        for term, values in term_items:
            if not isinstance(term, str):
                raise ValueError("Index terms must be strings.")
            posting_items = _mapping_items(
                values,
                label="term postings",
                maximum=_MAX_DOCUMENTS,
            )
            accepted: Dict[str, float] = {}
            for url, weight in posting_items:
                total_postings += 1
                if total_postings > _MAX_POSTINGS:
                    raise ValueError("Index contains too many postings.")
                if not isinstance(url, str) or url not in serialized_documents:
                    raise ValueError("Index postings reference an unknown document.")
                numeric = _finite_nonnegative(weight, "posting weight")
                if numeric <= 0:
                    raise ValueError("Posting weights must be positive.")
                accepted[url] = numeric
            if accepted:
                serialized_postings[term] = accepted
        serialized_norms = {
            url: _finite_nonnegative(value, "document norm")
            for url, value in _mapping_items(
                self.doc_norms,
                label="document norms",
                maximum=_MAX_DOCUMENTS,
            )
            if isinstance(url, str)
        }
        return {
            "schema_version": self.SCHEMA_VERSION,
            "documents": serialized_documents,
            "index": serialized_postings,
            "idf": serialized_idf,
            "doc_norms": serialized_norms,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "InvertedIndex":
        """Load only a bounded, internally consistent, finite sparse index."""

        if not isinstance(payload, dict):
            raise ValueError("Index payload must be an object.")
        raw_version = payload.get("schema_version", 1)
        if isinstance(raw_version, bool):
            raise ValueError("Index schema version is invalid.")
        try:
            version = int(raw_version)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Index schema version is invalid.") from exc
        if isinstance(raw_version, float) and not raw_version.is_integer():
            raise ValueError("Index schema version is invalid.")
        if version not in {1, cls.SCHEMA_VERSION}:
            raise ValueError(f"Unsupported index schema version {version}.")

        document_items = _mapping_items(
            payload.get("documents", {}),
            label="Index documents",
            maximum=_MAX_DOCUMENTS,
        )
        posting_terms = _mapping_items(
            payload.get("index", {}),
            label="Index postings",
            maximum=_MAX_TERMS,
        )
        idf_items = _mapping_items(
            payload.get("idf", {}),
            label="Index IDF values",
            maximum=_MAX_TERMS,
        )

        instance = cls()
        documents: Dict[str, DocumentMetadata] = {}
        for raw_url, metadata in document_items:
            if not isinstance(raw_url, str) or not 0 < len(raw_url) <= _MAX_URL_CHARS:
                continue
            if not isinstance(metadata, Mapping):
                continue
            try:
                title = metadata.get("title", "Untitled")
                snippet = metadata.get("snippet", "")
                length_value = metadata.get("length", 0)
            except Exception as exc:
                raise ValueError("Index document metadata is invalid.") from exc
            length = _bounded_nonnegative_int(
                length_value,
                "document length",
                _MAX_TOKENS_PER_DOCUMENT,
            )
            documents[raw_url] = DocumentMetadata(
                title=title if isinstance(title, str) else "Untitled",
                snippet=snippet if isinstance(snippet, str) else "",
                length=length,
            )

        parsed_idf: Dict[str, float] = {}
        for raw_term, raw_value in idf_items:
            if not isinstance(raw_term, str) or not 0 < len(raw_term) <= _MAX_TOKEN_CHARS:
                continue
            value = _finite_nonnegative(raw_value, "IDF value")
            if value > 0:
                parsed_idf[raw_term] = value

        postings: Dict[str, Dict[str, float]] = defaultdict(dict)
        norm_squares: Dict[str, float] = defaultdict(float)
        total_postings = 0
        for raw_term, raw_postings in posting_terms:
            if raw_term not in parsed_idf:
                continue
            posting_items = _mapping_items(
                raw_postings,
                label="Term postings",
                maximum=_MAX_DOCUMENTS,
            )
            accepted: Dict[str, float] = {}
            for raw_url, raw_weight in posting_items:
                total_postings += 1
                if total_postings > _MAX_POSTINGS:
                    raise ValueError("Persisted index contains too many postings.")
                if raw_url not in documents:
                    continue
                weight = _finite_nonnegative(raw_weight, "posting weight")
                if weight <= 0:
                    continue
                accepted[str(raw_url)] = weight
                norm_squares[str(raw_url)] += weight * weight
            if accepted:
                postings[str(raw_term)] = accepted

        doc_norms = {
            url: math.sqrt(value)
            for url, value in norm_squares.items()
            if math.isfinite(value) and value > 0
        }
        documents = {
            url: metadata
            for url, metadata in documents.items()
            if url in doc_norms
        }
        if document_items and not documents:
            raise ValueError("Persisted index contains no usable document vectors.")

        instance.documents = documents
        instance.index = defaultdict(
            dict,
            {
                term: {
                    url: weight
                    for url, weight in values.items()
                    if url in documents
                }
                for term, values in postings.items()
                if any(url in documents for url in values)
            },
        )
        instance.idf = {
            term: parsed_idf[term]
            for term in instance.index
        }
        instance.doc_norms = doc_norms
        return instance
