"""Sparse scientific-text TF-IDF index."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List

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


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in TOKEN_PATTERN.finditer(text or ""):
        token = match.group(0).casefold().strip("-./")
        if not token or token in STOP_WORDS:
            continue
        if len(token) == 1 and not token.isdigit() and token not in {"x", "y", "z", "r", "p", "q"}:
            continue
        tokens.append(token)
    return tokens


def build_snippet(text: str, max_words: int = 40) -> str:
    if max_words <= 0:
        return ""
    words = (text or "").split()
    snippet = " ".join(words[:max_words])
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
        term_document_frequency: Counter[str] = Counter()
        document_term_frequency: Dict[str, Counter[str]] = {}
        for url, page in pages.items():
            body_tokens = tokenize(page.text)
            title_tokens = tokenize(page.title)
            if not body_tokens and not title_tokens:
                continue
            frequencies = Counter(body_tokens)
            for token in title_tokens:
                frequencies[token] += 2
            document_term_frequency[url] = frequencies
            term_document_frequency.update(frequencies.keys())
            self.documents[url] = DocumentMetadata(
                title=page.title or "Untitled",
                snippet=build_snippet(page.text),
                length=len(body_tokens),
            )

        total_documents = len(document_term_frequency)
        if total_documents == 0:
            return
        self.idf = {
            term: math.log((1 + total_documents) / (1 + frequency)) + 1.0
            for term, frequency in term_document_frequency.items()
        }
        for url, frequencies in document_term_frequency.items():
            norm_squared = 0.0
            for term, frequency in frequencies.items():
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
        if not isinstance(payload, dict):
            raise ValueError("Index payload must be an object.")
        version = int(payload.get("schema_version", 1))
        if version not in {1, cls.SCHEMA_VERSION}:
            raise ValueError(f"Unsupported index schema version {version}.")
        instance = cls()
        documents = payload.get("documents", {})
        if isinstance(documents, dict):
            for url, metadata in documents.items():
                if not isinstance(url, str) or not isinstance(metadata, dict):
                    continue
                instance.documents[url] = DocumentMetadata(
                    title=str(metadata.get("title") or "Untitled"),
                    snippet=str(metadata.get("snippet") or ""),
                    length=max(int(metadata.get("length", 0)), 0),
                )
        postings_payload = payload.get("index", {})
        if isinstance(postings_payload, dict):
            for term, postings in postings_payload.items():
                if not isinstance(term, str) or not isinstance(postings, dict):
                    continue
                instance.index[term] = {
                    str(url): float(weight)
                    for url, weight in postings.items()
                    if str(url) in instance.documents
                }
        idf_payload = payload.get("idf", {})
        if isinstance(idf_payload, dict):
            instance.idf = {
                str(term): float(value) for term, value in idf_payload.items()
            }
        norms_payload = payload.get("doc_norms", {})
        if isinstance(norms_payload, dict):
            instance.doc_norms = {
                str(url): max(float(value), 0.0)
                for url, value in norms_payload.items()
                if str(url) in instance.documents
            }
        return instance
