"""Bounded, resumable crawler for explicitly allowed public domains."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING
from urllib import robotparser
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from tools.security import safe_download
from trusted_sources import ALL_TRUSTED_DOMAINS, ALL_TRUSTED_SEEDS

if TYPE_CHECKING:
    from storage import CrawlState

DEFAULT_USER_AGENT = (
    "RigorousRAGBot/3.0 "
    f"(+{os.getenv('CRAWLER_CONTACT_URL', 'https://github.com/Anurag9000/RigorousRAG')})"
)
REQUEST_TIMEOUT = 15
ALLOWED_MIME_TYPES = {"text/html", "application/xhtml+xml"}
MAX_CONTENT_LENGTH = 2_500_000
MIN_CONTENT_LENGTH = 512
_TRACKING_PARAMETERS = {
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source",
    "utm_campaign", "utm_content", "utm_medium", "utm_source", "utm_term",
}


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").rstrip(".").lower()


def is_trusted_domain(url: str, allowed_suffixes: Iterable[str]) -> bool:
    hostname = _hostname(url)
    if not hostname:
        return False
    for raw_suffix in allowed_suffixes:
        suffix = _hostname(
            raw_suffix if "://" in raw_suffix else f"https://{raw_suffix}"
        )
        if suffix and (hostname == suffix or hostname.endswith(f".{suffix}")):
            return True
    return False


def normalize_url(url: str) -> str:
    """Canonicalise safe HTTP(S) URLs enough to avoid crawl duplication."""

    parsed = urlparse((url or "").strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    hostname = parsed.hostname.rstrip(".").lower()
    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = hostname
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    path = parsed.path or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in _TRACKING_PARAMETERS
    ]
    query = urlencode(sorted(query_items), doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


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
        timeout: int = REQUEST_TIMEOUT,
        robots_fail_open: bool = False,
    ) -> None:
        if max_pages <= 0 or max_pages_per_domain <= 0:
            raise ValueError("Page limits must be positive.")
        if max_depth < 0 or request_delay < 0 or timeout <= 0:
            raise ValueError("Depth, delay, and timeout values are invalid.")
        self.allowed_domains: Set[str] = {
            _hostname(domain if "://" in domain else f"https://{domain}")
            for domain in allowed_domains
        }
        self.allowed_domains.discard("")
        self.max_pages = max_pages
        self.max_pages_per_domain = max_pages_per_domain
        self.max_depth = max_depth
        self.request_delay = request_delay
        self.user_agent = user_agent
        self.timeout = timeout
        self.robots_fail_open = robots_fail_open
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"User-Agent": user_agent})
        self._robots_cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}

    def crawl(
        self,
        seeds: Iterable[str],
        state: Optional["CrawlState"] = None,
    ) -> "CrawlState":
        if state is None:
            from storage import CrawlState as _CrawlState
            state = _CrawlState.empty()

        pages = dict(state.pages)
        graph = {url: set(edges) for url, edges in state.graph.items()}
        visited = set(state.visited) | set(pages)
        queue: deque[Tuple[str, int]] = deque(
            (normalize_url(url), max(int(depth), 0))
            for url, depth in state.frontier
            if normalize_url(url)
        )
        queued = {url for url, _depth in queue}
        # New source configuration must be respected even when a saved frontier exists.
        for seed in seeds:
            normalised = normalize_url(seed)
            if normalised and normalised not in visited and normalised not in queued:
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
            # Record every attempted URL so rejected URLs cannot be repeatedly requeued.
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
            pages[canonical_url] = page
            graph.setdefault(canonical_url, set())
            hostname = _hostname(canonical_url)
            if hostname:
                domain_counts[hostname] += 1

            next_depth = depth + 1
            for link in page.links:
                if not is_trusted_domain(link, self.allowed_domains):
                    continue
                graph[canonical_url].add(link)
                if (
                    next_depth <= self.max_depth
                    and link not in visited
                    and link not in queued
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
        content_type_header = downloaded.headers.get("Content-Type", "")
        content_type = content_type_header.split(";", 1)[0].strip().lower()
        encoding = "utf-8"
        if "charset=" in content_type_header.lower():
            encoding = content_type_header.lower().split("charset=", 1)[1].split(";", 1)[0].strip()
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
            return soup.title.string.strip()[:500]
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
        return " ".join(soup.get_text(separator=" ", strip=True).split())

    def _extract_links(self, base_url: str, soup: BeautifulSoup) -> List[str]:
        links: Set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href") or "").strip()
            if not href:
                continue
            absolute = normalize_url(urljoin(base_url, href))
            if not absolute or absolute == base_url:
                continue
            if is_trusted_domain(absolute, self.allowed_domains):
                links.add(absolute)
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
                    headers={"User-Agent": self.user_agent, "Accept": "text/plain,*/*;q=0.1"},
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
