"""Bounded, resumable crawler for explicitly allowed public domains."""

from __future__ import annotations

import itertools
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple
from urllib import robotparser
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from tools.security import safe_download
from trusted_sources import ALL_TRUSTED_DOMAINS, ALL_TRUSTED_SEEDS

if TYPE_CHECKING:
    from storage import CrawlState

REQUEST_TIMEOUT = 15.0
ALLOWED_MIME_TYPES = {"text/html", "application/xhtml+xml"}
MAX_CONTENT_LENGTH = 2_500_000
MIN_CONTENT_LENGTH = 512
_MAX_URL_CHARS = 4096
_MAX_QUERY_FIELDS = 200
_MAX_ALLOWED_DOMAINS = 1000
_MAX_SEEDS = 10_000
_MAX_PAGES = 100_000
_MAX_PAGES_PER_DOMAIN = 100_000
_MAX_DEPTH = 20
_MAX_DELAY_SECONDS = 60.0
_MAX_TIMEOUT_SECONDS = 300.0
_MAX_USER_AGENT_CHARS = 500
_MAX_LINKS_PER_PAGE = 2000
_MAX_ANCHORS_INSPECTED = 10_000
_MAX_PAGE_TEXT_CHARS = 5_000_000
_TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}


def _safe_text(value: object, *, limit: int, default: str = "") -> str:
    try:
        text = str(value if value is not None else default)
    except Exception:
        text = default
    return text[:limit]


def _clean_header(value: object, *, default: str) -> str:
    text = _safe_text(value, limit=_MAX_USER_AGENT_CHARS, default=default)
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    return text[:_MAX_USER_AGENT_CHARS] or default


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


_CONTACT_URL = _clean_header(
    os.getenv("CRAWLER_CONTACT_URL"),
    default="https://github.com/Anurag9000/RigorousRAG",
)
DEFAULT_USER_AGENT = _clean_header(
    f"RigorousRAGBot/3.0 (+{_CONTACT_URL})",
    default="RigorousRAGBot/3.0",
)


def _hostname(url: str) -> str:
    if not isinstance(url, str) or len(url) > _MAX_URL_CHARS:
        return ""
    try:
        return (urlparse(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return ""


def is_trusted_domain(url: str, allowed_suffixes: Iterable[str]) -> bool:
    hostname = _hostname(url)
    if not hostname or isinstance(allowed_suffixes, (str, bytes)):
        return False
    try:
        iterator = iter(allowed_suffixes)
    except TypeError:
        return False
    for raw_suffix in itertools.islice(iterator, _MAX_ALLOWED_DOMAINS):
        if not isinstance(raw_suffix, str):
            continue
        suffix = _hostname(
            raw_suffix if "://" in raw_suffix else f"https://{raw_suffix}"
        )
        if suffix and (hostname == suffix or hostname.endswith(f".{suffix}")):
            return True
    return False


def normalize_url(url: str) -> str:
    """Canonicalise one bounded credential-free HTTP(S) URL."""

    if not isinstance(url, str):
        return ""
    value = url.strip()
    if not value or len(value) > _MAX_URL_CHARS:
        return ""
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return ""
    try:
        parsed = urlparse(value)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port
    except ValueError:
        return ""
    if scheme not in {"http", "https"} or not hostname:
        return ""
    if parsed.username is not None or parsed.password is not None:
        return ""

    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = rendered_host
    if port and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        netloc = f"{rendered_host}:{port}"

    path = parsed.path or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    try:
        query_items = [
            (key, item)
            for key, item in parse_qsl(
                parsed.query,
                keep_blank_values=True,
                max_num_fields=_MAX_QUERY_FIELDS,
            )
            if key.casefold() not in _TRACKING_PARAMETERS
        ]
    except ValueError:
        return ""
    query = urlencode(sorted(query_items), doseq=True)
    result = urlunparse((scheme, netloc, path, "", query, ""))
    return result if len(result) <= _MAX_URL_CHARS else ""


@dataclass
class Page:
    url: str
    title: str
    text: str
    links: List[str]
    content_type: str
    content_length: int


class AcademicCrawler:
    """Breadth-first crawler constrained to an explicit host allowlist."""

    def __init__(
        self,
        allowed_domains: Iterable[str] = ALL_TRUSTED_DOMAINS,
        max_pages: int = 250,
        max_pages_per_domain: int = 35,
        max_depth: int = 2,
        request_delay: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = REQUEST_TIMEOUT,
        robots_fail_open: bool = False,
    ) -> None:
        self.max_pages = _bounded_int(
            max_pages,
            "max_pages",
            minimum=1,
            maximum=_MAX_PAGES,
        )
        self.max_pages_per_domain = _bounded_int(
            max_pages_per_domain,
            "max_pages_per_domain",
            minimum=1,
            maximum=_MAX_PAGES_PER_DOMAIN,
        )
        self.max_depth = _bounded_int(
            max_depth,
            "max_depth",
            minimum=0,
            maximum=_MAX_DEPTH,
        )
        self.request_delay = _bounded_float(
            request_delay,
            "request_delay",
            minimum=0.0,
            maximum=_MAX_DELAY_SECONDS,
        )
        self.timeout = _bounded_float(
            timeout,
            "timeout",
            minimum=0.1,
            maximum=_MAX_TIMEOUT_SECONDS,
        )
        if isinstance(allowed_domains, (str, bytes)):
            raise ValueError("allowed_domains must be an iterable of hostnames.")
        try:
            raw_domains = list(
                itertools.islice(iter(allowed_domains), _MAX_ALLOWED_DOMAINS + 1)
            )
        except TypeError as exc:
            raise ValueError("allowed_domains must be iterable.") from exc
        if len(raw_domains) > _MAX_ALLOWED_DOMAINS:
            raise ValueError(
                f"allowed_domains may contain at most {_MAX_ALLOWED_DOMAINS} entries."
            )
        self.allowed_domains: Set[str] = set()
        for domain in raw_domains:
            if not isinstance(domain, str):
                continue
            hostname = _hostname(domain if "://" in domain else f"https://{domain}")
            if hostname:
                self.allowed_domains.add(hostname)
        if not self.allowed_domains:
            raise ValueError("At least one valid allowed domain is required.")

        self.user_agent = _clean_header(user_agent, default=DEFAULT_USER_AGENT)
        self.robots_fail_open = bool(robots_fail_open)
        self.max_frontier_entries = min(max(self.max_pages * 10, 1000), 100_000)
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"User-Agent": self.user_agent})
        self._robots_cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "AcademicCrawler":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _bounded_frontier(
        self,
        frontier: object,
    ) -> deque[Tuple[str, int]]:
        queue: deque[Tuple[str, int]] = deque()
        if isinstance(frontier, (str, bytes)):
            return queue
        try:
            iterator = iter(frontier)  # type: ignore[arg-type]
        except TypeError:
            return queue
        for item in itertools.islice(iterator, self.max_frontier_entries):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            normalized = normalize_url(item[0])
            if not normalized:
                continue
            try:
                depth = max(int(item[1]), 0)
            except (TypeError, ValueError, OverflowError):
                continue
            queue.append((normalized, min(depth, self.max_depth + 1)))
        return queue

    def crawl(
        self,
        seeds: Iterable[str],
        state: Optional["CrawlState"] = None,
    ) -> "CrawlState":
        if state is None:
            from storage import CrawlState as _CrawlState

            state = _CrawlState.empty()

        pages: Dict[str, Page] = {}
        raw_pages = getattr(state, "pages", {})
        if isinstance(raw_pages, dict):
            for raw_url, page in itertools.islice(raw_pages.items(), self.max_pages):
                normalized = normalize_url(raw_url)
                if (
                    not normalized
                    or not is_trusted_domain(normalized, self.allowed_domains)
                    or not isinstance(page, Page)
                ):
                    continue
                links = []
                for link in itertools.islice(page.links or [], _MAX_LINKS_PER_PAGE):
                    normalized_link = normalize_url(link)
                    if normalized_link and is_trusted_domain(
                        normalized_link,
                        self.allowed_domains,
                    ):
                        links.append(normalized_link)
                pages[normalized] = Page(
                    url=normalized,
                    title=_safe_text(page.title, limit=500, default="Untitled") or "Untitled",
                    text=_safe_text(page.text, limit=_MAX_PAGE_TEXT_CHARS),
                    links=sorted(set(links)),
                    content_type=_safe_text(page.content_type, limit=200),
                    content_length=max(int(page.content_length or 0), 0),
                )

        raw_graph = getattr(state, "graph", {})
        graph: Dict[str, Set[str]] = {url: set() for url in pages}
        if isinstance(raw_graph, dict):
            for raw_url, raw_edges in raw_graph.items():
                url = normalize_url(raw_url)
                if url not in pages or isinstance(raw_edges, (str, bytes)):
                    continue
                try:
                    edge_iterator = iter(raw_edges)
                except TypeError:
                    continue
                for raw_edge in itertools.islice(edge_iterator, _MAX_LINKS_PER_PAGE):
                    edge = normalize_url(raw_edge)
                    if edge and is_trusted_domain(edge, self.allowed_domains):
                        graph[url].add(edge)

        visited: Set[str] = set(pages)
        raw_visited = getattr(state, "visited", set())
        if not isinstance(raw_visited, (str, bytes)):
            try:
                visited_iterator = iter(raw_visited)
            except TypeError:
                visited_iterator = iter(())
            for raw_url in itertools.islice(
                visited_iterator,
                self.max_pages + self.max_frontier_entries,
            ):
                normalized = normalize_url(raw_url)
                if normalized:
                    visited.add(normalized)

        queue = self._bounded_frontier(getattr(state, "frontier", []))
        queued = {url for url, _depth in queue}
        if isinstance(seeds, (str, bytes)):
            raise ValueError("seeds must be an iterable of URLs, not a string.")
        try:
            seed_values = list(itertools.islice(iter(seeds), _MAX_SEEDS + 1))
        except TypeError as exc:
            raise ValueError("seeds must be iterable.") from exc
        if len(seed_values) > _MAX_SEEDS:
            raise ValueError(f"seeds may contain at most {_MAX_SEEDS} entries.")
        for seed in seed_values:
            normalised = normalize_url(seed)
            if normalised and normalised not in visited and normalised not in queued:
                if len(queue) >= self.max_frontier_entries:
                    break
                queue.append((normalised, 0))
                queued.add(normalised)

        domain_counts: Dict[str, int] = defaultdict(int)
        for existing_url in pages:
            hostname = _hostname(existing_url)
            if hostname:
                domain_counts[hostname] += 1

        while queue and len(pages) < self.max_pages:
            current_url, depth = queue.popleft()
            queued.discard(current_url)
            if current_url in visited:
                continue
            visited.add(current_url)
            if depth > self.max_depth:
                continue
            if not is_trusted_domain(current_url, self.allowed_domains):
                continue
            if not self._under_domain_quota(current_url, domain_counts):
                continue
            if not self._is_allowed_by_robots(current_url):
                continue

            page = self._fetch_page(current_url)
            if page is None:
                continue
            canonical_url = normalize_url(page.url) or current_url
            if not is_trusted_domain(canonical_url, self.allowed_domains):
                continue
            visited.add(canonical_url)
            if canonical_url in pages:
                continue
            if not self._under_domain_quota(canonical_url, domain_counts):
                continue
            if canonical_url != current_url and not self._is_allowed_by_robots(
                canonical_url
            ):
                continue
            page.url = canonical_url
            pages[canonical_url] = page
            graph.setdefault(canonical_url, set())
            hostname = _hostname(canonical_url)
            if hostname:
                domain_counts[hostname] += 1

            next_depth = depth + 1
            for link in page.links[:_MAX_LINKS_PER_PAGE]:
                if not is_trusted_domain(link, self.allowed_domains):
                    continue
                graph[canonical_url].add(link)
                if (
                    next_depth <= self.max_depth
                    and link not in visited
                    and link not in queued
                    and len(queue) < self.max_frontier_entries
                ):
                    queue.append((link, next_depth))
                    queued.add(link)
            if self.request_delay:
                time.sleep(self.request_delay)

        state.pages = pages
        state.graph = graph
        state.visited = visited
        state.frontier = list(queue)
        return state

    def _fetch_page(self, url: str) -> Optional[Page]:
        try:
            downloaded = safe_download(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                },
                timeout=self.timeout,
                max_bytes=MAX_CONTENT_LENGTH,
                allowed_content_types=ALLOWED_MIME_TYPES,
                session=self.session,
            )
        except Exception:
            return None
        final_url = normalize_url(downloaded.final_url)
        if not final_url or not is_trusted_domain(final_url, self.allowed_domains):
            return None
        content_type_header = _safe_text(
            downloaded.headers.get("Content-Type", ""),
            limit=500,
        )
        content_type = content_type_header.split(";", 1)[0].strip().lower()
        encoding = "utf-8"
        if "charset=" in content_type_header.lower():
            encoding = (
                content_type_header.lower()
                .split("charset=", 1)[1]
                .split(";", 1)[0]
                .strip()
            )
        try:
            html = downloaded.content.decode(encoding, errors="replace")
        except LookupError:
            html = downloaded.content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        title = self._extract_title(soup)
        text = self._extract_text(soup)
        if len(text) < MIN_CONTENT_LENGTH:
            return None
        links = self._extract_links(final_url, soup)
        return Page(
            url=final_url,
            title=title,
            text=text,
            links=links,
            content_type=content_type,
            content_length=len(downloaded.content),
        )

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return _safe_text(soup.title.string, limit=500).strip() or "Untitled"
        heading = soup.find(["h1", "h2"])
        if heading:
            return heading.get_text(" ", strip=True)[:500] or "Untitled"
        return "Untitled"

    @staticmethod
    def _extract_text(soup: BeautifulSoup) -> str:
        for element in soup(
            ["script", "style", "noscript", "header", "footer", "nav", "aside", "svg"]
        ):
            element.decompose()
        return " ".join(soup.get_text(separator=" ", strip=True).split())[
            :_MAX_PAGE_TEXT_CHARS
        ]

    def _extract_links(self, base_url: str, soup: BeautifulSoup) -> List[str]:
        links: Set[str] = set()
        anchors = soup.find_all("a", href=True, limit=_MAX_ANCHORS_INSPECTED)
        for anchor in anchors:
            href = _safe_text(anchor.get("href"), limit=_MAX_URL_CHARS).strip()
            if not href:
                continue
            absolute = normalize_url(urljoin(base_url, href))
            if not absolute or absolute == base_url:
                continue
            if is_trusted_domain(absolute, self.allowed_domains):
                links.add(absolute)
                if len(links) >= _MAX_LINKS_PER_PAGE:
                    break
        return sorted(links)

    def _is_allowed_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots_cache:
            robots_url = urljoin(base, "/robots.txt")
            parser: Optional[robotparser.RobotFileParser] = None
            try:
                downloaded = safe_download(
                    robots_url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "text/plain,*/*;q=0.1",
                    },
                    timeout=self.timeout,
                    max_bytes=512_000,
                    session=self.session,
                )
                text = downloaded.content.decode("utf-8", errors="replace")
                parser = robotparser.RobotFileParser(robots_url)
                parser.parse(text.splitlines())
            except Exception:
                parser = None
            self._robots_cache[base] = parser
        parser = self._robots_cache[base]
        if parser is None:
            return self.robots_fail_open
        try:
            return parser.can_fetch(self.user_agent, url)
        except Exception:
            return self.robots_fail_open

    def _under_domain_quota(
        self,
        url: str,
        domain_counts: Dict[str, int],
    ) -> bool:
        hostname = _hostname(url)
        return bool(hostname) and domain_counts.get(hostname, 0) < self.max_pages_per_domain


DEFAULT_SEEDS: List[str] = ALL_TRUSTED_SEEDS
