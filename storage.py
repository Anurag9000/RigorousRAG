"""Atomic, versioned persistence for crawl and lexical-index state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

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
    """Persist legacy files and manifest-committed cross-file generations."""

    SCHEMA_VERSION = 3
    _SNAPSHOT_KEYS = ("crawl", "index", "pagerank")

    def __init__(self, base_dir: Path | str = "data") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.crawl_path = self.base_dir / "crawl_state.json"
        self.index_path = self.base_dir / "index.json"
        self.pagerank_path = self.base_dir / "pagerank.json"
        self.snapshot_manifest_path = self.base_dir / "snapshot_manifest.json"
        self.snapshot_lock_path = self.base_dir / ".snapshot.lock"
        self.max_snapshot_file_bytes = max(
            1_000_000,
            min(
                int(os.getenv("CLASSIC_MAX_SNAPSHOT_FILE_BYTES", "250000000")),
                2_000_000_000,
            ),
        )
        self._lock = threading.RLock()

    @staticmethod
    def _encode_json(payload: Any) -> bytes:
        return json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")

    def _fsync_directory(self) -> None:
        try:
            descriptor = os.open(
                self.base_dir,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @contextmanager
    def _snapshot_guard(self) -> Iterator[None]:
        """Serialize manifest reads, publication, and old-generation cleanup.

        The process-local lock protects threads even on platforms where advisory file
        locks are process-scoped. The file lock protects separate service/CLI
        processes that share the classic storage directory.
        """

        with self._lock:
            if self.snapshot_lock_path.is_symlink():
                raise OSError("Snapshot lock path cannot be a symbolic link.")
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.snapshot_lock_path, flags, 0o600)
            handle = os.fdopen(descriptor, "r+b", closefd=True)
            lock_kind = ""
            try:
                if os.name == "nt":  # pragma: no cover - exercised on Windows CI/users
                    import msvcrt

                    handle.seek(0, os.SEEK_END)
                    if handle.tell() == 0:
                        handle.write(b"\0")
                        handle.flush()
                        os.fsync(handle.fileno())
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    lock_kind = "windows"
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    lock_kind = "posix"
                yield
            finally:
                try:
                    if lock_kind == "windows":  # pragma: no cover - Windows only
                        import msvcrt

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    elif lock_kind == "posix":
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()

    def _quarantine(self, path: Path) -> None:
        if not path.exists():
            return
        corrupt = path.with_suffix(path.suffix + f".corrupt-{uuid.uuid4().hex[:8]}")
        try:
            os.replace(path, corrupt)
            self._fsync_directory()
        except OSError:
            pass

    def _read_json(self, path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        with self._lock:
            try:
                if path.stat().st_size > self.max_snapshot_file_bytes:
                    raise ValueError("Persisted JSON exceeds the configured byte limit.")
                return json.loads(path.read_text(encoding="utf-8"))
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                self._quarantine(path)
                return None

    def _write_bytes(self, path: Path, encoded: bytes) -> None:
        if len(encoded) > self.max_snapshot_file_bytes:
            raise ValueError(
                f"Persisted JSON exceeds the {self.max_snapshot_file_bytes}-byte limit."
            )
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with self._lock:
            try:
                with temp.open("xb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, path)
                self._fsync_directory()
            finally:
                temp.unlink(missing_ok=True)

    def _write_json(self, path: Path, payload: Any) -> None:
        self._write_bytes(path, self._encode_json(payload))

    @classmethod
    def _crawl_payload(cls, state: CrawlState) -> Dict[str, Any]:
        return {
            "schema_version": cls.SCHEMA_VERSION,
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
        }

    @classmethod
    def _crawl_from_payload(cls, payload: Any) -> CrawlState:
        if not isinstance(payload, dict):
            return CrawlState.empty()
        try:
            version = int(payload.get("schema_version", 1))
        except (TypeError, ValueError):
            return CrawlState.empty()
        if version not in {1, 2, cls.SCHEMA_VERSION}:
            return CrawlState.empty()
        pages_payload = payload.get("pages", {})
        pages: Dict[str, Page] = {}
        if isinstance(pages_payload, dict):
            for url, data in pages_payload.items():
                if not isinstance(url, str) or not isinstance(data, dict):
                    continue
                try:
                    content_length = max(int(data.get("content_length", 0)), 0)
                except (TypeError, ValueError):
                    continue
                raw_links = data.get("links", [])
                links = (
                    [str(item) for item in raw_links if isinstance(item, str)]
                    if isinstance(raw_links, list)
                    else []
                )
                pages[url] = Page(
                    url=url,
                    title=str(data.get("title") or "Untitled")[:500],
                    text=str(data.get("text") or ""),
                    links=links,
                    content_type=str(data.get("content_type") or "")[:200],
                    content_length=content_length,
                )
        graph_payload = payload.get("graph", {})
        graph = (
            {
                str(url): {str(edge) for edge in edges if isinstance(edge, str)}
                for url, edges in graph_payload.items()
                if isinstance(url, str) and isinstance(edges, list)
            }
            if isinstance(graph_payload, dict)
            else {}
        )
        visited_payload = payload.get("visited", [])
        visited = (
            {str(value) for value in visited_payload if isinstance(value, str)}
            if isinstance(visited_payload, list)
            else set()
        )
        frontier: List[Tuple[str, int]] = []
        raw_frontier = payload.get("frontier", [])
        if isinstance(raw_frontier, list):
            for item in raw_frontier:
                if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                    continue
                try:
                    depth = max(int(item.get("depth", 0)), 0)
                except (TypeError, ValueError):
                    continue
                frontier.append((item["url"], depth))
        return CrawlState(pages=pages, graph=graph, visited=visited, frontier=frontier)

    @classmethod
    def _pagerank_payload(cls, pagerank: Dict[str, float]) -> Dict[str, Any]:
        scores: Dict[str, float] = {}
        for url, score in pagerank.items():
            value = float(score)
            if not math.isfinite(value) or value < 0:
                raise ValueError("PageRank scores must be finite and non-negative.")
            scores[str(url)] = value
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "scores": scores,
        }

    @staticmethod
    def _pagerank_from_payload(payload: Any) -> Dict[str, float]:
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
            if isinstance(url, str) and math.isfinite(value) and value >= 0:
                result[url] = value
        return result

    def load_crawl_state(self) -> CrawlState:
        return self._crawl_from_payload(self._read_json(self.crawl_path))

    def save_crawl_state(self, state: CrawlState) -> None:
        self._write_json(self.crawl_path, self._crawl_payload(state))

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
        return self._pagerank_from_payload(self._read_json(self.pagerank_path))

    def save_pagerank(self, pagerank: Dict[str, float]) -> None:
        self._write_json(self.pagerank_path, self._pagerank_payload(pagerank))

    def _generation_files_exist(self) -> bool:
        patterns = (
            "crawl_state.*.json",
            "index.*.json",
            "pagerank.*.json",
        )
        return any(any(self.base_dir.glob(pattern)) for pattern in patterns)

    def _invalid_snapshot(self) -> Tuple[CrawlState, Optional[InvertedIndex], Dict[str, float]]:
        self._quarantine(self.snapshot_manifest_path)
        return CrawlState.empty(), None, {}

    def _read_manifest_member(
        self,
        *,
        generation: str,
        key: str,
        entry: Any,
    ) -> Optional[Any]:
        expected_names = {
            "crawl": f"crawl_state.{generation}.json",
            "index": f"index.{generation}.json",
            "pagerank": f"pagerank.{generation}.json",
        }
        if key not in expected_names or not isinstance(entry, dict):
            return None
        name = str(entry.get("name") or "")
        if name != expected_names[key] or Path(name).name != name:
            return None
        try:
            expected_bytes = int(entry.get("bytes"))
            expected_count = int(entry.get("count"))
        except (TypeError, ValueError):
            return None
        expected_digest = str(entry.get("sha256") or "")
        if (
            expected_bytes < 0
            or expected_bytes > self.max_snapshot_file_bytes
            or expected_count < 0
            or len(expected_digest) != 64
        ):
            return None
        path = self.base_dir / name
        try:
            if path.is_symlink() or not path.is_file():
                return None
            if path.stat().st_size != expected_bytes:
                return None
            encoded = path.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(encoded).hexdigest() != expected_digest:
            return None
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return None
        return payload

    def load_snapshot(
        self,
    ) -> Tuple[CrawlState, Optional[InvertedIndex], Dict[str, float]]:
        """Load one fully verified generation, or legacy files before first migration."""

        with self._snapshot_guard():
            return self._load_snapshot_locked()

    def _load_snapshot_locked(
        self,
    ) -> Tuple[CrawlState, Optional[InvertedIndex], Dict[str, float]]:
        manifest_existed = self.snapshot_manifest_path.exists()
        manifest = self._read_json(self.snapshot_manifest_path)
        if manifest is None:
            if manifest_existed or self._generation_files_exist():
                return CrawlState.empty(), None, {}
            return self.load_crawl_state(), self.load_index(), self.load_pagerank()
        if not isinstance(manifest, dict):
            return self._invalid_snapshot()
        try:
            version = int(manifest.get("schema_version"))
        except (TypeError, ValueError):
            return self._invalid_snapshot()
        generation = str(manifest.get("generation") or "")
        if version != self.SCHEMA_VERSION or not (
            len(generation) == 32
            and all(character in "0123456789abcdef" for character in generation)
        ):
            return self._invalid_snapshot()
        entries = manifest.get("files")
        if not isinstance(entries, dict) or set(entries) != set(self._SNAPSHOT_KEYS):
            return self._invalid_snapshot()

        payloads: Dict[str, Any] = {}
        for key in self._SNAPSHOT_KEYS:
            payload = self._read_manifest_member(
                generation=generation,
                key=key,
                entry=entries.get(key),
            )
            if payload is None:
                return self._invalid_snapshot()
            payloads[key] = payload

        state = self._crawl_from_payload(payloads["crawl"])
        try:
            index = InvertedIndex.from_dict(payloads["index"])
        except (TypeError, ValueError):
            return self._invalid_snapshot()
        pagerank = self._pagerank_from_payload(payloads["pagerank"])
        counts = {
            "crawl": len(state.pages),
            "index": len(index.documents),
            "pagerank": len(pagerank),
        }
        try:
            manifest_counts = {
                key: int(entries[key]["count"]) for key in self._SNAPSHOT_KEYS
            }
        except (KeyError, TypeError, ValueError):
            return self._invalid_snapshot()
        page_urls = set(state.pages)
        if (
            counts != manifest_counts
            or not set(index.documents).issubset(page_urls)
            or set(pagerank) != page_urls
        ):
            return self._invalid_snapshot()
        return state, index, pagerank

    def save_snapshot(
        self,
        state: CrawlState,
        index: InvertedIndex,
        pagerank: Dict[str, float],
    ) -> str:
        """Write generation files first and atomically publish their manifest last."""

        with self._snapshot_guard():
            return self._save_snapshot_locked(state, index, pagerank)

    def _save_snapshot_locked(
        self,
        state: CrawlState,
        index: InvertedIndex,
        pagerank: Dict[str, float],
    ) -> str:
        page_urls = set(state.pages)
        if not set(index.documents).issubset(page_urls):
            raise ValueError("Index documents must be a subset of persisted crawl pages.")
        if set(pagerank) != page_urls:
            raise ValueError("PageRank keys must exactly match persisted crawl pages.")
        generation = uuid.uuid4().hex
        payloads = {
            "crawl": self._crawl_payload(state),
            "index": index.to_dict(),
            "pagerank": self._pagerank_payload(pagerank),
        }
        names = {
            "crawl": f"crawl_state.{generation}.json",
            "index": f"index.{generation}.json",
            "pagerank": f"pagerank.{generation}.json",
        }
        encoded = {
            key: self._encode_json(payloads[key]) for key in self._SNAPSHOT_KEYS
        }
        counts = {
            "crawl": len(state.pages),
            "index": len(index.documents),
            "pagerank": len(pagerank),
        }
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "generation": generation,
            "files": {
                key: {
                    "name": names[key],
                    "sha256": hashlib.sha256(encoded[key]).hexdigest(),
                    "bytes": len(encoded[key]),
                    "count": counts[key],
                }
                for key in self._SNAPSHOT_KEYS
            },
        }

        for key in self._SNAPSHOT_KEYS:
            self._write_bytes(self.base_dir / names[key], encoded[key])
        # This atomic replace is the cross-file commit point. Before it, the old
        # manifest remains authoritative and the new files are merely unreferenced.
        self._write_json(self.snapshot_manifest_path, manifest)
        current_names = set(names.values())
        for pattern in (
            "crawl_state.*.json",
            "index.*.json",
            "pagerank.*.json",
        ):
            for candidate in self.base_dir.glob(pattern):
                if candidate.name in current_names or candidate.is_symlink():
                    continue
                try:
                    candidate.unlink()
                except OSError:
                    pass
        self._fsync_directory()
        return generation
