"""Atomic, versioned persistence for crawl and lexical-index state."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    SCHEMA_VERSION = 2

    def __init__(self, base_dir: Path | str = "data") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.crawl_path = self.base_dir / "crawl_state.json"
        self.index_path = self.base_dir / "index.json"
        self.pagerank_path = self.base_dir / "pagerank.json"
        self._lock = threading.RLock()

    def _read_json(self, path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        with self._lock:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                corrupt = path.with_suffix(path.suffix + f".corrupt-{uuid.uuid4().hex[:8]}")
                try:
                    os.replace(path, corrupt)
                except OSError:
                    pass
                return None

    def _write_json(self, path: Path, payload: Any) -> None:
        encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with self._lock:
            try:
                with temp.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)

    def load_crawl_state(self) -> CrawlState:
        payload = self._read_json(self.crawl_path)
        if not isinstance(payload, dict):
            return CrawlState.empty()
        pages_payload = payload.get("pages", {})
        pages: Dict[str, Page] = {}
        if isinstance(pages_payload, dict):
            for url, data in pages_payload.items():
                if not isinstance(url, str) or not isinstance(data, dict):
                    continue
                pages[url] = Page(
                    url=url,
                    title=str(data.get("title") or "Untitled"),
                    text=str(data.get("text") or ""),
                    links=[str(item) for item in data.get("links", []) if isinstance(item, str)],
                    content_type=str(data.get("content_type") or ""),
                    content_length=max(int(data.get("content_length", 0)), 0),
                )
        graph_payload = payload.get("graph", {})
        graph = {
            str(url): {str(edge) for edge in edges if isinstance(edge, str)}
            for url, edges in graph_payload.items()
            if isinstance(url, str) and isinstance(edges, list)
        } if isinstance(graph_payload, dict) else {}
        visited_payload = payload.get("visited", [])
        visited = {
            str(value) for value in visited_payload if isinstance(value, str)
        } if isinstance(visited_payload, list) else set()
        frontier: List[Tuple[str, int]] = []
        for item in payload.get("frontier", []) if isinstance(payload.get("frontier", []), list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            try:
                depth = max(int(item.get("depth", 0)), 0)
            except (TypeError, ValueError):
                continue
            frontier.append((item["url"], depth))
        return CrawlState(pages=pages, graph=graph, visited=visited, frontier=frontier)

    def save_crawl_state(self, state: CrawlState) -> None:
        self._write_json(
            self.crawl_path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "pages": {
                    url: {
                        "title": page.title,
                        "text": page.text,
                        "links": sorted(set(page.links)),
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
            },
        )

    def load_index(self) -> Optional[InvertedIndex]:
        payload = self._read_json(self.index_path)
        if not isinstance(payload, dict):
            return None
        try:
            return InvertedIndex.from_dict(payload)
        except (TypeError, ValueError):
            return None

    def save_index(self, index: InvertedIndex) -> None:
        self._write_json(self.index_path, index.to_dict())

    def load_pagerank(self) -> Dict[str, float]:
        payload = self._read_json(self.pagerank_path)
        if not isinstance(payload, dict):
            return {}
        rankings = payload.get("scores", payload)
        if not isinstance(rankings, dict):
            return {}
        result: Dict[str, float] = {}
        for url, score in rankings.items():
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if isinstance(url, str) and value >= 0:
                result[url] = value
        return result

    def save_pagerank(self, pagerank: Dict[str, float]) -> None:
        self._write_json(
            self.pagerank_path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "scores": {url: float(score) for url, score in pagerank.items()},
            },
        )
