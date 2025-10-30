"""Persistent storage helpers for crawl state, index, and ranking data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from Crawler import Page
from Indexer import InvertedIndex


@dataclass
class CrawlState:
    pages: Dict[str, Page]
    graph: Dict[str, Set[str]]
    visited: Set[str]
    frontier: List[Tuple[str, int]]

    @classmethod
    def empty(cls) -> "CrawlState":
        return cls(pages={}, graph={}, visited=set(), frontier=[])


class StorageManager:
    """Handles persistence of crawl state, indexes, and ranking metadata."""

    def __init__(self, base_dir: Path | str = "data") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.crawl_path = self.base_dir / "crawl_state.json"
        self.index_path = self.base_dir / "index.json"
        self.pagerank_path = self.base_dir / "pagerank.json"

    # ---- Crawl state -------------------------------------------------
    def load_crawl_state(self) -> CrawlState:
        if not self.crawl_path.exists():
            return CrawlState.empty()
        data = json.loads(self.crawl_path.read_text(encoding="utf-8"))
        pages = {
            url: Page(
                url=url,
                title=payload.get("title", "Untitled"),
                text=payload.get("text", ""),
                links=payload.get("links", []),
                content_type=payload.get("content_type", ""),
                content_length=payload.get("content_length", 0),
            )
            for url, payload in data.get("pages", {}).items()
        }
        graph = {
            url: set(edges) for url, edges in data.get("graph", {}).items()
        }
        visited = set(data.get("visited", []))
        frontier = [
            (item["url"], item["depth"])
            for item in data.get("frontier", [])
            if "url" in item and "depth" in item
        ]
        return CrawlState(pages=pages, graph=graph, visited=visited, frontier=frontier)

    def save_crawl_state(self, state: CrawlState) -> None:
        data = {
            "pages": {
                url: {
                    "title": page.title,
                    "text": page.text,
                    "links": list(page.links),
                    "content_type": page.content_type,
                    "content_length": page.content_length,
                }
                for url, page in state.pages.items()
            },
            "graph": {url: sorted(edges) for url, edges in state.graph.items()},
            "visited": sorted(state.visited),
            "frontier": [
                {"url": url, "depth": depth} for url, depth in state.frontier
            ],
        }
        self.crawl_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---- Index -------------------------------------------------------
    def load_index(self) -> Optional[InvertedIndex]:
        if not self.index_path.exists():
            return None
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        return InvertedIndex.from_dict(payload)

    def save_index(self, index: InvertedIndex) -> None:
        data = index.to_dict()
        self.index_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---- PageRank ----------------------------------------------------
    def load_pagerank(self) -> Dict[str, float]:
        if not self.pagerank_path.exists():
            return {}
        return {
            url: float(score)
            for url, score in json.loads(
                self.pagerank_path.read_text(encoding="utf-8")
            ).items()
        }

    def save_pagerank(self, pagerank: Dict[str, float]) -> None:
        self.pagerank_path.write_text(
            json.dumps(pagerank, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
