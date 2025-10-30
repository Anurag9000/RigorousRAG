"""Crawler for trusted academic, educational, and governmental sources."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from trusted_sources import ALL_TRUSTED_DOMAINS, ALL_TRUSTED_SEEDS

if TYPE_CHECKING:
    from storage import CrawlState

DEFAULT_USER_AGENT = (
    "AcademicSearchBot/2.0 (+https://example.com/academic-search-bot-info)"
)
REQUEST_TIMEOUT = 10
ALLOWED_MIME_TYPES = {"text/html", "application/xhtml+xml"}
MAX_CONTENT_LENGTH = 2_500_000  # ~2.5 MB
MIN_CONTENT_LENGTH = 512


def is_trusted_domain(url: str, allowed_suffixes: Iterable[str]) -> bool:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if not netloc:
        return False
    for suffix in allowed_suffixes:
        if netloc == suffix or netloc.endswith(f".{suffix}"):
            return True
    return False


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    if not parsed.netloc:
        return ""
    normalized = parsed._replace(fragment="")
    return urlunparse(normalized)


@dataclass
class Page:
    url: str
    title: str
    text: str
    links: List[str]
    content_type: str
    content_length: int


class AcademicCrawler:
    """Breadth-first crawler constrained to curated trusted domains."""

    def __init__(
        self,
        allowed_domains: Iterable[str] = ALL_TRUSTED_DOMAINS,
        max_pages: int = 250,
        max_pages_per_domain: int = 35,
        max_depth: int = 2,
        request_delay: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.allowed_domains: Set[str] = set(allowed_domains)
        self.max_pages = max_pages
        self.max_pages_per_domain = max_pages_per_domain
        self.max_depth = max_depth
        self.request_delay = request_delay
        self.user_agent = user_agent
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._robots_cache: Dict[str, robotparser.RobotFileParser] = {}

    def crawl(
        self, seeds: Iterable[str], state: Optional["CrawlState"] = None
    ) -> "CrawlState":
        if state is None:
            from storage import CrawlState as _CrawlState  # Local import to avoid cycle

            state = _CrawlState.empty()

        pages: Dict[str, Page] = dict(state.pages)
        graph: Dict[str, Set[str]] = {
            url: set(edges) for url, edges in state.graph.items()
        }
        visited: Set[str] = set(state.visited) | set(pages.keys())

        queue: deque[Tuple[str, int]] = deque(state.frontier)
        queued: Set[str] = {url for url, _ in queue}
        if not queue:
            for seed in seeds:
                normalized = normalize_url(seed)
                if (
                    normalized
                    and normalized not in visited
                    and normalized not in queued
                ):
                    queue.append((normalized, 0))
                    queued.add(normalized)

        domain_counts: Dict[str, int] = defaultdict(int)
        for url in pages:
            netloc = urlparse(url).netloc.lower()
            if netloc:
                domain_counts[netloc] += 1

        while queue and len(pages) < self.max_pages:
            current_url, depth = queue.popleft()
            queued.discard(current_url)
            if current_url in visited or depth > self.max_depth:
                continue
            if not is_trusted_domain(current_url, self.allowed_domains):
                continue
            if not self._under_domain_quota(current_url, domain_counts):
                continue
            if not self._is_allowed_by_robots(current_url):
                continue

            page = self._fetch_page(current_url)
            visited.add(current_url)

            if not page:
                continue

            pages[current_url] = page
            graph.setdefault(current_url, set())

            netloc = urlparse(current_url).netloc.lower()
            if netloc:
                domain_counts[netloc] += 1

            next_depth = depth + 1
            for link in page.links:
                graph[current_url].add(link)
                if (
                    link not in visited
                    and next_depth <= self.max_depth
                    and is_trusted_domain(link, self.allowed_domains)
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
            response = self.session.get(
                url, timeout=self.timeout, allow_redirects=True
            )
            response.raise_for_status()
        except requests.RequestException:
            return None

        content_type_header = response.headers.get("Content-Type", "")
        content_type = content_type_header.split(";")[0].strip().lower()
        content_length_header = response.headers.get("Content-Length")
        try:
            content_length = int(content_length_header) if content_length_header else 0
        except ValueError:
            content_length = 0

        if content_length and content_length > MAX_CONTENT_LENGTH:
            return None
        if content_length and content_length < MIN_CONTENT_LENGTH:
            return None
        if content_type and content_type not in ALLOWED_MIME_TYPES:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        title = self._extract_title(soup)
        text = self._extract_text(soup)
        links = self._extract_links(url, soup)

        if len(text) < MIN_CONTENT_LENGTH:
            return None

        return Page(
            url=url,
            title=title,
            text=text,
            links=links,
            content_type=content_type,
            content_length=content_length,
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        first_heading = soup.find(["h1", "h2"])
        if first_heading and first_heading.get_text(strip=True):
            return first_heading.get_text(strip=True)
        return "Untitled"

    def _extract_text(self, soup: BeautifulSoup) -> str:
        for element in soup(
            ["script", "style", "noscript", "header", "footer", "nav", "aside"]
        ):
            element.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())

    def _extract_links(self, base_url: str, soup: BeautifulSoup) -> List[str]:
        links: Set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            if not href:
                continue
            absolute = normalize_url(urljoin(base_url, href))
            if not absolute or absolute == base_url:
                continue
            if not is_trusted_domain(absolute, self.allowed_domains):
                continue
            if not self._is_allowed_by_robots(absolute):
                continue
            links.add(absolute)
        return list(links)

    def _is_allowed_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots = self._robots_cache.get(base)
        if not robots:
            robots = robotparser.RobotFileParser()
            robots.set_url(urljoin(base, "/robots.txt"))
            try:
                robots.read()
            except Exception:
                pass
            self._robots_cache[base] = robots
        try:
            return robots.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _under_domain_quota(
        self, url: str, domain_counts: Dict[str, int]
    ) -> bool:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if not netloc:
            return False
        return domain_counts.get(netloc, 0) < self.max_pages_per_domain


DEFAULT_SEEDS: List[str] = ALL_TRUSTED_SEEDS
