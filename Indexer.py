"""Index construction and text processing utilities."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List

# Lightweight stop word list to avoid requiring NLTK datasets
STOP_WORDS: set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "hi",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z]{2,}")


def tokenize(text: str) -> List[str]:
    tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
    return [token for token in tokens if token not in STOP_WORDS]


def build_snippet(text: str, max_words: int = 40) -> str:
    words = text.split()
    return " ".join(words[:max_words])


@dataclass
class DocumentMetadata:
    title: str
    snippet: str
    length: int


class InvertedIndex:
    """Sparse TF-IDF index with cosine similarity support."""

    def __init__(self) -> None:
        self.documents: Dict[str, DocumentMetadata] = {}
        self.index: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.idf: Dict[str, float] = {}
        self.doc_norms: Dict[str, float] = {}

    def build(self, pages: Dict[str, "Page"]) -> None:
        term_document_frequency: Counter[str] = Counter()
        document_term_frequency: Dict[str, Counter[str]] = {}

        for url, page in pages.items():
            body_tokens = tokenize(page.text)
            title_tokens = tokenize(page.title)
            if not body_tokens and not title_tokens:
                continue

            tf_counter = Counter(body_tokens)
            for token in title_tokens:
                tf_counter[token] += 2  # Lightweight title boost

            document_term_frequency[url] = tf_counter
            term_document_frequency.update(tf_counter.keys())

            snippet = build_snippet(page.text)
            self.documents[url] = DocumentMetadata(
                title=page.title, snippet=snippet, length=len(body_tokens)
            )

        total_documents = len(document_term_frequency)
        if total_documents == 0:
            return

        for term, df in term_document_frequency.items():
            self.idf[term] = math.log((1 + total_documents) / (1 + df)) + 1

        for url, tf_counter in document_term_frequency.items():
            norm_squared = 0.0
            for term, frequency in tf_counter.items():
                idf = self.idf.get(term)
                if idf is None:
                    continue
                weight = (1 + math.log(frequency)) * idf
                self.index[term][url] = weight
                norm_squared += weight * weight
            self.doc_norms[url] = math.sqrt(norm_squared)
