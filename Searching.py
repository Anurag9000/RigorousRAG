"""Command-line entrypoint for the academic-focused search engine."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from Crawler import AcademicCrawler, Page, DEFAULT_SEEDS
from Indexer import InvertedIndex, tokenize
from Pagerank import compute_pagerank
from storage import StorageManager, CrawlState


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
    """High-level orchestration for crawling, indexing, and querying."""

    def __init__(
        self,
        seeds: Optional[Sequence[str]] = None,
        max_pages: int = 200,
        max_depth: int = 2,
        request_delay: float = 1.0,
    ) -> None:
        self.seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
        self.crawler = AcademicCrawler(
            max_pages=max_pages,
            max_depth=max_depth,
            request_delay=request_delay,
        )
        self.storage = StorageManager()
        self.state: CrawlState = self.storage.load_crawl_state()
        stored_index = self.storage.load_index()
        self.index = stored_index if stored_index else InvertedIndex()
        self.pagerank_scores: Dict[str, float] = self.storage.load_pagerank()
        self.pages: Dict[str, Page] = dict(self.state.pages)

    def build(self) -> int:
        """Run the crawler, build the index, and compute PageRank."""
        self.state = self.crawler.crawl(self.seeds, self.state)
        self.storage.save_crawl_state(self.state)
        self.pages = self.state.pages

        self.index = InvertedIndex()
        self.index.build(self.pages)
        self.storage.save_index(self.index)

        self.pagerank_scores = compute_pagerank(self.state.graph)
        self.storage.save_pagerank(self.pagerank_scores)
        return len(self.pages)

    def search(self, query: str, limit: int = 10) -> List[SearchHit]:
        tokens = tokenize(query)
        if not tokens:
            return []

        query_counter = Counter(tokens)
        query_vector: Dict[str, float] = {}
        query_norm_sq = 0.0

        for term, frequency in query_counter.items():
            idf = self.index.idf.get(term)
            if idf is None:
                continue
            weight = (1 + math.log(frequency)) * idf
            query_vector[term] = weight
            query_norm_sq += weight * weight

        if not query_vector or query_norm_sq == 0.0:
            return []

        query_norm = math.sqrt(query_norm_sq)
        raw_scores: Dict[str, float] = {}

        for term, q_weight in query_vector.items():
            postings = self.index.index.get(term)
            if not postings:
                continue
            for url, d_weight in postings.items():
                raw_scores[url] = raw_scores.get(url, 0.0) + q_weight * d_weight

        ranked_hits: List[SearchHit] = []
        for url, dot_product in raw_scores.items():
            doc_norm = self.index.doc_norms.get(url)
            if not doc_norm:
                continue
            cosine_score = dot_product / (doc_norm * query_norm)
            pagerank_score = self.pagerank_scores.get(url, 0.0)
            combined_score = 0.85 * cosine_score + 0.15 * pagerank_score
            metadata = self.index.documents.get(url)

            ranked_hits.append(
                SearchHit(
                    rank=0,  # placeholder, set after sorting
                    url=url,
                    title=metadata.title if metadata else "Untitled",
                    snippet=metadata.snippet if metadata else "",
                    score=combined_score,
                    cosine=cosine_score,
                    pagerank=pagerank_score,
                    length=metadata.length if metadata else 0,
                )
            )

        ranked_hits.sort(key=lambda item: item.score, reverse=True)
        for idx, hit in enumerate(ranked_hits[:limit], start=1):
            hit.rank = idx
        return ranked_hits[:limit]

    def gather_context(
        self, hits: Sequence[SearchHit], max_chars: int = 6000
    ) -> List[Dict[str, str]]:
        """Return lightweight context documents for downstream summarisation."""
        context: List[Dict[str, str]] = []
        consumed = 0
        for hit in hits:
            page = self.pages.get(hit.url)
            if not page:
                continue
            # Safety: ensure we don't divide by zero if hits is empty
            chunk_size = max_chars // max(1, len(hits)) if hits else max_chars
            excerpt = page.text[:chunk_size]
            consumed += len(excerpt)
            context.append(
                {
                    "url": hit.url,
                    "title": hit.title,
                    "text": excerpt,
                }
            )
            if consumed >= max_chars:
                break
        return context

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
                    snippet = match.snippet
                    print(f"   {snippet[:200]}{'...' if len(snippet) > 200 else ''}")
                print(
                    f"   score={match.score:.3f} cosine={match.cosine:.3f} "
                    f"pagerank={match.pagerank:.3f}"
                )
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Academic-focused mini search engine"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=150,
        help="Maximum number of pages to crawl (default: 150)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum crawl depth from each seed URL (default: 2)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between HTTP requests (default: 1.0)",
    )
    parser.add_argument(
        "--results",
        type=int,
        default=10,
        help="Number of search results to display interactively (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = AcademicSearchEngine(
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        request_delay=args.delay,
    )

    print("Crawling trusted academic domains...")
    total_pages = engine.build()
    print(f"Crawl complete. Indexed {total_pages} pages.\n")

    engine.interactive_loop(limit=args.results)


if __name__ == "__main__":
    main()
