"""Classic lexical academic search engine and CLI."""

from __future__ import annotations

import argparse
import itertools
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from Crawler import AcademicCrawler, DEFAULT_SEEDS, Page
from Indexer import InvertedIndex, tokenize
from Pagerank import compute_pagerank
from storage import CrawlState, StorageManager

_MAX_QUERY_CHARS = 2000
_MAX_SEEDS = 10_000
_MAX_RESULTS = 100
_MAX_CONTEXT_CHARS = 100_000
_MAX_CONTEXT_HITS = 100
_MAX_DISPLAY_CHARS = 500


def _clean_line(value: object, *, limit: int, default: str = "") -> str:
    try:
        text = str(value if value is not None else default)
    except Exception:
        text = default
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())[:limit]


def _bounded_int(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return numeric


def _bounded_float(value: object, label: str, *, minimum: float, maximum: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be finite and between {minimum} and {maximum}.")
    return numeric


def _bounded_query(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Search queries must be strings.")
    query = value.strip()
    if not query:
        return ""
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError(f"Search queries may contain at most {_MAX_QUERY_CHARS} characters.")
    return query


def _bounded_storage_dir(value: object) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("storage_dir must be a filesystem path.")
    rendered = os.fspath(value)
    if not rendered or len(rendered) > 4096 or "\x00" in rendered:
        raise ValueError("storage_dir is invalid or too long.")
    return rendered


@dataclass
class SearchHit:
    rank: int
    url: str
    title: str
    snippet: str
    score: float
    cosine: float
    pagerank: float
    length: int

    def __post_init__(self) -> None:
        self.rank = _bounded_int(self.rank, "rank", minimum=0, maximum=1_000_000)
        self.url = _clean_line(self.url, limit=4096)
        self.title = _clean_line(self.title, limit=500, default="Untitled") or "Untitled"
        self.snippet = _clean_line(self.snippet, limit=4000)
        self.score = _bounded_float(self.score, "score", minimum=0.0, maximum=1.0)
        self.cosine = _bounded_float(self.cosine, "cosine", minimum=0.0, maximum=1.0)
        self.pagerank = _bounded_float(self.pagerank, "pagerank", minimum=0.0, maximum=1.0)
        self.length = _bounded_int(
            self.length,
            "document length",
            minimum=0,
            maximum=1_000_000,
        )


class AcademicSearchEngine:
    """Crawler, sparse index, authority prior, and persisted search state."""

    def __init__(
        self,
        seeds: Optional[Sequence[str]] = None,
        max_pages: int = 200,
        max_depth: int = 2,
        request_delay: float = 1.0,
        *,
        storage_dir: Optional[str] = None,
        lexical_weight: float = 0.85,
    ) -> None:
        self.lexical_weight = _bounded_float(
            lexical_weight,
            "lexical_weight",
            minimum=0.0,
            maximum=1.0,
        )
        if seeds is None:
            raw_seeds: Iterable[str] = DEFAULT_SEEDS
        else:
            if isinstance(seeds, (str, bytes)):
                raise ValueError("seeds must be a sequence of URLs, not a string.")
            raw_seeds = seeds
        try:
            seed_values = list(itertools.islice(iter(raw_seeds), _MAX_SEEDS + 1))
        except TypeError as exc:
            raise ValueError("seeds must be iterable.") from exc
        if len(seed_values) > _MAX_SEEDS:
            raise ValueError(f"At most {_MAX_SEEDS} seeds may be configured.")
        self.seeds = [value for value in seed_values if isinstance(value, str)]
        self.crawler = AcademicCrawler(
            max_pages=max_pages,
            max_depth=max_depth,
            request_delay=request_delay,
        )
        selected_storage = storage_dir
        if selected_storage is None:
            selected_storage = os.getenv("CLASSIC_STORAGE_DIR", "data")
        self.storage = StorageManager(_bounded_storage_dir(selected_storage))
        state, index, pagerank = self.storage.load_snapshot()
        if not self.storage.snapshot_manifest_path.exists():
            legacy_presence = (
                self.storage.crawl_path.exists(),
                self.storage.index_path.exists(),
                self.storage.pagerank_path.exists(),
            )
            page_urls = set(state.pages)
            legacy_consistent = (
                all(legacy_presence)
                and index is not None
                and set(index.documents).issubset(page_urls)
                and set(pagerank) == page_urls
            )
            if any(legacy_presence) and not legacy_consistent:
                state, index, pagerank = CrawlState.empty(), None, {}
        self.state = state
        self.index = index or InvertedIndex()
        self.pagerank_scores = {
            url: float(value)
            for url, value in pagerank.items()
            if isinstance(url, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
        }
        self.pages: Dict[str, Page] = dict(self.state.pages)

    def close(self) -> None:
        self.crawler.close()

    def __enter__(self) -> "AcademicSearchEngine":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def ready(self) -> bool:
        return bool(
            self.index.documents
            and set(self.index.documents).issubset(self.pages)
            and set(self.pagerank_scores) == set(self.pages)
        )

    def build(self) -> int:
        self.state = self.crawler.crawl(self.seeds, self.state)
        self.pages = dict(self.state.pages)
        page_urls = set(self.pages)
        filtered_graph = {
            url: {
                target
                for target in self.state.graph.get(url, set())
                if target in page_urls
            }
            for url in page_urls
        }
        self.state.graph = filtered_graph
        self.index = InvertedIndex()
        self.index.build(self.pages)
        self.pagerank_scores = compute_pagerank(filtered_graph)
        self.storage.save_snapshot(
            self.state,
            self.index,
            self.pagerank_scores,
        )
        return len(self.pages)

    def search(self, query: str, limit: int = 10) -> List[SearchHit]:
        bounded_query = _bounded_query(query)
        if not bounded_query:
            return []
        requested = _bounded_int(
            limit,
            "limit",
            minimum=1,
            maximum=_MAX_RESULTS,
        )
        tokens = tokenize(bounded_query)
        if not tokens or not self.index.documents:
            return []
        frequencies = Counter(tokens)
        query_vector: Dict[str, float] = {}
        query_norm_squared = 0.0
        for term, frequency in frequencies.items():
            idf = self.index.idf.get(term)
            if idf is None or not math.isfinite(idf) or idf <= 0:
                continue
            weight = (1.0 + math.log(frequency)) * idf
            if not math.isfinite(weight) or weight <= 0:
                continue
            query_vector[term] = weight
            query_norm_squared += weight * weight
        if (
            not query_vector
            or not math.isfinite(query_norm_squared)
            or query_norm_squared <= 0
        ):
            return []
        query_norm = math.sqrt(query_norm_squared)
        dot_products: Dict[str, float] = {}
        for term, query_weight in query_vector.items():
            for url, document_weight in self.index.index.get(term, {}).items():
                if not math.isfinite(document_weight) or document_weight <= 0:
                    continue
                updated = dot_products.get(url, 0.0) + query_weight * document_weight
                if math.isfinite(updated):
                    dot_products[url] = updated

        finite_pagerank = [
            value
            for value in self.pagerank_scores.values()
            if math.isfinite(value) and value >= 0
        ]
        global_max_pagerank = max(finite_pagerank, default=0.0)
        authority_weight = 1.0 - self.lexical_weight
        ranked: List[SearchHit] = []
        for url, dot_product in dot_products.items():
            norm = self.index.doc_norms.get(url, 0.0)
            metadata = self.index.documents.get(url)
            if (
                not math.isfinite(norm)
                or norm <= 0
                or metadata is None
                or url not in self.pages
            ):
                continue
            denominator = norm * query_norm
            if not math.isfinite(denominator) or denominator <= 0:
                continue
            cosine = max(0.0, min(dot_product / denominator, 1.0))
            raw_pagerank = self.pagerank_scores.get(url, 0.0)
            if not math.isfinite(raw_pagerank) or raw_pagerank < 0:
                raw_pagerank = 0.0
            pagerank = (
                raw_pagerank / global_max_pagerank
                if global_max_pagerank > 0
                else 0.0
            )
            score = self.lexical_weight * cosine + authority_weight * pagerank
            if not math.isfinite(score):
                continue
            ranked.append(
                SearchHit(
                    rank=0,
                    url=url,
                    title=metadata.title,
                    snippet=self._query_snippet(url, tokens, metadata.snippet),
                    score=max(0.0, min(score, 1.0)),
                    cosine=cosine,
                    pagerank=max(0.0, min(pagerank, 1.0)),
                    length=metadata.length,
                )
            )
        ranked.sort(
            key=lambda item: (item.score, item.cosine, item.pagerank, item.url),
            reverse=True,
        )
        selected = ranked[:requested]
        for rank, hit in enumerate(selected, start=1):
            hit.rank = rank
        return selected

    def _query_snippet(
        self,
        url: str,
        tokens: Sequence[str],
        fallback: str,
    ) -> str:
        page = self.pages.get(url)
        if page is None or not page.text:
            return _clean_line(fallback, limit=4000)
        lowered = page.text.casefold()
        positions: List[int] = []
        for token in tokens[:1000]:
            position = lowered.find(token)
            if position >= 0:
                positions.append(position)
        if not positions:
            return _clean_line(fallback, limit=4000)
        centre = min(positions)
        start = max(0, centre - 180)
        end = min(len(page.text), centre + 420)
        while start > 0 and not page.text[start - 1].isspace():
            start -= 1
        while end < len(page.text) and not page.text[end - 1].isspace():
            end += 1
        prefix = "…" if start else ""
        suffix = "…" if end < len(page.text) else ""
        return _clean_line(
            f"{prefix}{page.text[start:end].strip()}{suffix}",
            limit=4000,
        )

    def gather_context(
        self,
        hits: Sequence[SearchHit],
        max_chars: int = 6000,
    ) -> List[Dict[str, str]]:
        requested_chars = _bounded_int(
            max_chars,
            "max_chars",
            minimum=1,
            maximum=_MAX_CONTEXT_CHARS,
        )
        if isinstance(hits, (str, bytes)):
            raise ValueError("hits must be a sequence of SearchHit objects.")
        try:
            candidates = list(
                itertools.islice(iter(hits), _MAX_CONTEXT_HITS + 1)
            )
        except TypeError as exc:
            raise ValueError("hits must be iterable.") from exc
        if len(candidates) > _MAX_CONTEXT_HITS:
            raise ValueError(f"At most {_MAX_CONTEXT_HITS} hits may be gathered.")
        valid = [
            (hit, self.pages.get(hit.url))
            for hit in candidates
            if isinstance(hit, SearchHit)
        ]
        valid = [(hit, page) for hit, page in valid if page is not None]
        if not valid:
            return []
        per_document = max(1, requested_chars // len(valid))
        contexts: List[Dict[str, str]] = []
        remaining = requested_chars
        for hit, page in valid:
            assert page is not None
            excerpt = page.text[: min(per_document, remaining)]
            if not excerpt:
                continue
            contexts.append(
                {
                    "url": hit.url,
                    "title": hit.title,
                    "text": excerpt,
                }
            )
            remaining -= len(excerpt)
            if remaining <= 0:
                break
        return contexts

    def interactive_loop(self, limit: int = 10) -> None:
        requested = _bounded_int(
            limit,
            "limit",
            minimum=1,
            maximum=_MAX_RESULTS,
        )
        print("Enter an empty line to exit.\n")
        while True:
            try:
                query = input("Search> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                break
            try:
                matches = self.search(query, limit=requested)
            except ValueError as exc:
                print(f"Invalid query: {exc}\n")
                continue
            if not matches:
                print("No results found.\n")
                continue
            for match in matches:
                print(f"{match.rank}. {match.title}")
                print(f"   {match.url}")
                if match.snippet:
                    print(f"   {match.snippet[:300]}")
                print(
                    f"   score={match.score:.3f} lexical={match.cosine:.3f} "
                    f"authority={match.pagerank:.3f}"
                )
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Academic lexical search engine")
    parser.add_argument("--max-pages", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--results", type=int, default=10)
    parser.add_argument(
        "--storage-dir",
        default=os.getenv("CLASSIC_STORAGE_DIR", "data"),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Crawl and rebuild the persisted index before searching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        with AcademicSearchEngine(
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            request_delay=args.delay,
            storage_dir=args.storage_dir,
        ) as engine:
            if args.rebuild or not engine.ready:
                print("Crawling and rebuilding the academic index...")
                total_pages = engine.build()
                print(f"Indexed {total_pages} pages.\n")
            else:
                print(
                    f"Loaded {len(engine.index.documents)} indexed pages from disk.\n"
                )
            engine.interactive_loop(limit=args.results)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
