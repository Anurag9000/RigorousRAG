"""Classic lexical academic search engine and CLI."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from Crawler import AcademicCrawler, DEFAULT_SEEDS, Page
from Indexer import InvertedIndex, tokenize
from Pagerank import compute_pagerank
from storage import CrawlState, StorageManager


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


class AcademicSearchEngine:
    """Crawler, sparse index, authority prior, and persisted search state."""

    def __init__(
        self,
        seeds: Optional[Sequence[str]] = None,
        max_pages: int = 200,
        max_depth: int = 2,
        request_delay: float = 1.0,
        *,
        storage_dir: str = "data",
        lexical_weight: float = 0.85,
    ) -> None:
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between 0 and 1.")
        self.lexical_weight = lexical_weight
        self.seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
        self.crawler = AcademicCrawler(
            max_pages=max_pages,
            max_depth=max_depth,
            request_delay=request_delay,
        )
        self.storage = StorageManager(storage_dir)
        self.state: CrawlState = self.storage.load_crawl_state()
        self.index = self.storage.load_index() or InvertedIndex()
        self.pagerank_scores: Dict[str, float] = self.storage.load_pagerank()
        self.pages: Dict[str, Page] = dict(self.state.pages)

    @property
    def ready(self) -> bool:
        return bool(self.index.documents)

    def build(self) -> int:
        self.state = self.crawler.crawl(self.seeds, self.state)
        self.pages = dict(self.state.pages)
        # PageRank authority is defined only over pages that were actually fetched.
        page_urls = set(self.pages)
        filtered_graph = {
            url: {target for target in self.state.graph.get(url, set()) if target in page_urls}
            for url in page_urls
        }
        self.state.graph = filtered_graph
        self.storage.save_crawl_state(self.state)
        self.index = InvertedIndex()
        self.index.build(self.pages)
        self.storage.save_index(self.index)
        self.pagerank_scores = compute_pagerank(filtered_graph)
        self.storage.save_pagerank(self.pagerank_scores)
        return len(self.pages)

    def search(self, query: str, limit: int = 10) -> List[SearchHit]:
        limit = max(1, min(int(limit), 100))
        tokens = tokenize(query)
        if not tokens or not self.index.documents:
            return []
        frequencies = Counter(tokens)
        query_vector: Dict[str, float] = {}
        query_norm_squared = 0.0
        for term, frequency in frequencies.items():
            idf = self.index.idf.get(term)
            if idf is None:
                continue
            weight = (1.0 + math.log(frequency)) * idf
            query_vector[term] = weight
            query_norm_squared += weight * weight
        if not query_vector or query_norm_squared <= 0:
            return []
        query_norm = math.sqrt(query_norm_squared)
        dot_products: Dict[str, float] = {}
        for term, query_weight in query_vector.items():
            for url, document_weight in self.index.index.get(term, {}).items():
                dot_products[url] = dot_products.get(url, 0.0) + query_weight * document_weight

        global_max_pagerank = max(self.pagerank_scores.values(), default=0.0)
        authority_weight = 1.0 - self.lexical_weight
        ranked: List[SearchHit] = []
        for url, dot_product in dot_products.items():
            norm = self.index.doc_norms.get(url, 0.0)
            metadata = self.index.documents.get(url)
            if norm <= 0 or metadata is None:
                continue
            cosine = max(0.0, min(dot_product / (norm * query_norm), 1.0))
            raw_pagerank = max(self.pagerank_scores.get(url, 0.0), 0.0)
            pagerank = raw_pagerank / global_max_pagerank if global_max_pagerank else 0.0
            score = self.lexical_weight * cosine + authority_weight * pagerank
            ranked.append(
                SearchHit(
                    rank=0,
                    url=url,
                    title=metadata.title,
                    snippet=self._query_snippet(url, tokens, metadata.snippet),
                    score=score,
                    cosine=cosine,
                    pagerank=pagerank,
                    length=metadata.length,
                )
            )
        ranked.sort(key=lambda item: (item.score, item.cosine, item.pagerank), reverse=True)
        for rank, hit in enumerate(ranked[:limit], start=1):
            hit.rank = rank
        return ranked[:limit]

    def _query_snippet(self, url: str, tokens: Sequence[str], fallback: str) -> str:
        page = self.pages.get(url)
        if page is None or not page.text:
            return fallback
        lowered = page.text.casefold()
        positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
        if not positions:
            return fallback
        centre = min(positions)
        start = max(0, centre - 180)
        end = min(len(page.text), centre + 420)
        while start > 0 and not page.text[start - 1].isspace():
            start -= 1
        while end < len(page.text) and not page.text[end - 1].isspace():
            end += 1
        prefix = "…" if start else ""
        suffix = "…" if end < len(page.text) else ""
        return f"{prefix}{page.text[start:end].strip()}{suffix}"

    def gather_context(
        self,
        hits: Sequence[SearchHit],
        max_chars: int = 6000,
    ) -> List[Dict[str, str]]:
        if max_chars <= 0 or not hits:
            return []
        valid = [(hit, self.pages.get(hit.url)) for hit in hits]
        valid = [(hit, page) for hit, page in valid if page is not None]
        if not valid:
            return []
        per_document = max(1, max_chars // len(valid))
        contexts: List[Dict[str, str]] = []
        remaining = max_chars
        for hit, page in valid:
            assert page is not None
            excerpt = page.text[: min(per_document, remaining)]
            if not excerpt:
                continue
            contexts.append({
                "url": hit.url,
                "title": hit.title,
                "text": excerpt,
            })
            remaining -= len(excerpt)
            if remaining <= 0:
                break
        return contexts

    def interactive_loop(self, limit: int = 10) -> None:
        print("Enter an empty line to exit.\n")
        while True:
            try:
                query = input("Search> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                break
            matches = self.search(query, limit=limit)
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
    parser.add_argument("--storage-dir", default="data")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Crawl and rebuild the persisted index before searching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = AcademicSearchEngine(
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        request_delay=args.delay,
        storage_dir=args.storage_dir,
    )
    if args.rebuild or not engine.ready:
        print("Crawling and rebuilding the academic index...")
        total_pages = engine.build()
        print(f"Indexed {total_pages} pages.\n")
    else:
        print(f"Loaded {len(engine.index.documents)} indexed pages from disk.\n")
    engine.interactive_loop(limit=max(1, min(args.results, 100)))


if __name__ == "__main__":
    main()
