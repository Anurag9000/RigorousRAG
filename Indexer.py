"""Bounded sparse scientific-text TF-IDF index."""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping

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

# Unicode word components, retaining numbers and scientific identifiers such as
# IL-6, GPT-4o, H2O, p53, α-synuclein, 10.1038 and ResNet/50.
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


def _finite_nonnegative(value: Any, label: str) -> float:
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
    if not isinstance(text, str) or max_words <= 0:
        return ""
    words = text[:_MAX_TEXT_CHARS].split()
    snippet = " ".join(words[:max_words])[:_MAX_SNIPPET_CHARS]
    return f"{snippet}…" if len(words) > max_words else snippet


@dataclass
class DocumentMetadata:
    title: str
    snippet: str
    length: int


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
        """Rebuild from scratch; repeated calls never retain stale postings."""

        self.clear()
        if not isinstance(pages, dict):
            raise ValueError("pages must be a URL-to-Page mapping.")
        if len(pages) > _MAX_DOCUMENTS:
            raise ValueError(f"An index may contain at most {_MAX_DOCUMENTS} documents.")

        term_document_frequency: Counter[str] = Counter()
        document_term_frequency: Dict[str, Counter[str]] = {}
        for raw_url, page in pages.items():
            if not isinstance(raw_url, str) or not 0 < len(raw_url) <= _MAX_URL_CHARS:
                continue
            if not isinstance(page, Page):
                continue
            body_tokens = tokenize(page.text)
            title_tokens = tokenize(page.title)
            if not body_tokens and not title_tokens:
                continue
            frequencies = Counter(body_tokens)
            for token in title_tokens:
                frequencies[token] += 2
            document_term_frequency[raw_url] = frequencies
            term_document_frequency.update(frequencies.keys())
            self.documents[raw_url] = DocumentMetadata(
                title=str(page.title or "Untitled")[:_MAX_TITLE_CHARS],
                snippet=build_snippet(page.text),
                length=len(body_tokens),
            )

        total_documents = len(document_term_frequency)
        if total_documents == 0:
            return
        if len(term_document_frequency) > _MAX_TERMS:
            raise ValueError(f"An index may contain at most {_MAX_TERMS} unique terms.")

        self.idf = {
            term: math.log((1 + total_documents) / (1 + frequency)) + 1.0
            for term, frequency in term_document_frequency.items()
        }
        total_postings = 0
        for url, frequencies in document_term_frequency.items():
            norm_squared = 0.0
            for term, frequency in frequencies.items():
                total_postings += 1
                if total_postings > _MAX_POSTINGS:
                    raise ValueError(
                        f"An index may contain at most {_MAX_POSTINGS} postings."
                    )
                weight = (1.0 + math.log(frequency)) * self.idf[term]
                self.index[term][url] = weight
                norm_squared += weight * weight
            self.doc_norms[url] = math.sqrt(norm_squared)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "documents": {
                url: asdict(metadata) for url, metadata in self.documents.items()
            },
            "index": {term: dict(postings) for term, postings in self.index.items()},
            "idf": dict(self.idf),
            "doc_norms": dict(self.doc_norms),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "InvertedIndex":
        """Load only a bounded, internally consistent, finite sparse index."""

        if not isinstance(payload, dict):
            raise ValueError("Index payload must be an object.")
        try:
            version = int(payload.get("schema_version", 1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Index schema version is invalid.") from exc
        if version not in {1, cls.SCHEMA_VERSION}:
            raise ValueError(f"Unsupported index schema version {version}.")

        documents_payload = payload.get("documents", {})
        postings_payload = payload.get("index", {})
        idf_payload = payload.get("idf", {})
        if not isinstance(documents_payload, Mapping):
            raise ValueError("Index documents must be an object.")
        if not isinstance(postings_payload, Mapping):
            raise ValueError("Index postings must be an object.")
        if not isinstance(idf_payload, Mapping):
            raise ValueError("Index IDF values must be an object.")
        if len(documents_payload) > _MAX_DOCUMENTS:
            raise ValueError("Persisted index contains too many documents.")
        if len(postings_payload) > _MAX_TERMS or len(idf_payload) > _MAX_TERMS:
            raise ValueError("Persisted index contains too many terms.")

        instance = cls()
        for raw_url, metadata in documents_payload.items():
            if not isinstance(raw_url, str) or not 0 < len(raw_url) <= _MAX_URL_CHARS:
                continue
            if not isinstance(metadata, Mapping):
                continue
            length = _bounded_nonnegative_int(
                metadata.get("length", 0),
                "document length",
                _MAX_TOKENS_PER_DOCUMENT,
            )
            instance.documents[raw_url] = DocumentMetadata(
                title=str(metadata.get("title") or "Untitled")[:_MAX_TITLE_CHARS],
                snippet=str(metadata.get("snippet") or "")[:_MAX_SNIPPET_CHARS],
                length=length,
            )

        parsed_idf: Dict[str, float] = {}
        for raw_term, raw_value in idf_payload.items():
            if not isinstance(raw_term, str) or not 0 < len(raw_term) <= _MAX_TOKEN_CHARS:
                continue
            value = _finite_nonnegative(raw_value, "IDF value")
            if value > 0:
                parsed_idf[raw_term] = value

        norm_squares: Dict[str, float] = defaultdict(float)
        total_postings = 0
        for raw_term, raw_postings in postings_payload.items():
            if raw_term not in parsed_idf or not isinstance(raw_postings, Mapping):
                continue
            accepted: Dict[str, float] = {}
            for raw_url, raw_weight in itertools.islice(
                raw_postings.items(),
                _MAX_DOCUMENTS + 1,
            ):
                total_postings += 1
                if total_postings > _MAX_POSTINGS:
                    raise ValueError("Persisted index contains too many postings.")
                if raw_url not in instance.documents:
                    continue
                weight = _finite_nonnegative(raw_weight, "posting weight")
                if weight <= 0:
                    continue
                accepted[str(raw_url)] = weight
                norm_squares[str(raw_url)] += weight * weight
            if accepted:
                instance.index[str(raw_term)] = accepted
                instance.idf[str(raw_term)] = parsed_idf[str(raw_term)]

        instance.doc_norms = {
            url: math.sqrt(value)
            for url, value in norm_squares.items()
            if math.isfinite(value) and value > 0
        }
        instance.documents = {
            url: metadata
            for url, metadata in instance.documents.items()
            if url in instance.doc_norms
        }
        if documents_payload and not instance.documents:
            raise ValueError("Persisted index contains no usable document vectors.")
        return instance
