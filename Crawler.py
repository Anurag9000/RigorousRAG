"""Bounded, resumable crawler for explicitly allowed public domains."""

from __future__ import annotations

import ipaddress
import itertools
import math
import operator
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Set, Tuple
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
_HOST_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
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
    if isinstance(value, str):
        text = value
    elif value is None:
        text = default
    else:
        try:
            text = str(value)
        except Exception:
            text = default
    return text[: max(int(limit), 0)]


def _safe_getattr(value: object, name: str, default: object) -> object:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _clean_header(value: object, *, default: str) -> str:
    if not isinstance(value, str):
        return default
    normalized = "".join(
        " " if ord(character) < 32 or ord(character) == 127 else character
        for character in value[:_MAX_USER_AGENT_CHARS]
    )
    text = " ".join(normalized.split())
    return text[:_MAX_USER_AGENT_CHARS] or default


def _bounded_int(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return numeric


def _bounded_float(value: object, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be finite and between {minimum} and {maximum}.")
    return numeric


def _nonnegative_int(value: object, maximum: int = 2_000_000_000) -> int:
    if isinstance(value, bool):
        return 0
    try:
        numeric = int(operator.index(value))
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(numeric, maximum))


def _bounded_items(value: object, maximum: int) -> Iterator[object]:
    if isinstance(value, (str, bytes, bytearray)):
        return iter(())
    try:
        return itertools.islice(iter(value), maximum)  # type: ignore[arg-type]
    except Exception:
        return iter(())


def _bounded_collection(value: object, maximum: int, label: str) -> List[object]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable collection.")
    try:
        items = list(itertools.islice(iter(value), maximum + 1))  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{label} must be a safely iterable collection.") from exc
    if len(items) > maximum:
        raise ValueError(f"{label} may contain at most {maximum} entries.")
    return items


def _canonical_hostname(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 253:
        return ""
    if any(
        character.isspace() or ord(character) < 33 or ord(character) == 127
        for character in value
    ):
        return ""
    candidate = value.rstrip(".").lower()
    try:
        return ipaddress.ip_address(candidate).compressed.lower()
    except ValueError:
        pass
    try:
        ascii_host = candidate.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return ""
    if len(ascii_host) > 253:
        return ""
    labels = ascii_host.split(".")
    if not labels or any(not _HOST_LABEL_RE.fullmatch(label) for label in labels):
        return ""
    return ascii_host


def _hostname(url: str) -> str:
    if not isinstance(url, str) or len(url) > _MAX_URL_CHARS:
        return ""
    try:
        return _canonical_hostname(urlparse(url).hostname or "")
    except ValueError:
        return ""


def _allowed_domain(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 253
        or _contains_ascii_control(value)
        or "\\" in value
    ):
        return ""
    try:
        parsed = urlparse(value if "://" in value else f"https://{value}")
        port = parsed.port
    except (ValueError, UnicodeError):
        return ""
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return _canonical_hostname(parsed.hostname or "")


_CONTACT_URL = _clean_header(
    os.getenv("CRAWLER_CONTACT_URL", ""),
    default="https://github.com/Anurag9000/RigorousRAG",
)
DEFAULT_USER_AGENT = _clean_header(
    f"RigorousRAGBot/3.0 (+{_CONTACT_URL})",
    default="RigorousRAGBot/3.0",
)


def is_trusted_domain(url: str, allowed_suffixes: Iterable[str]) -> bool:
    hostname = _hostname(url)
    if not hostname or isinstance(allowed_suffixes, (str, bytes, bytearray)):
        return False
    for raw_suffix in _bounded_items(allowed_suffixes, _MAX_ALLOWED_DOMAINS):
        suffix = _allowed_domain(raw_suffix)
        if suffix and (hostname == suffix or hostname.endswith(f".{suffix}")):
            return True
    return False


def normalize_url(url: str) -> str:
    """Canonicalise one bounded credential-free HTTP(S) URL."""

    if not isinstance(url, str):
        return ""
    value = url
    if (
        not value
        or value != value.strip()
        or len(value) > _MAX_URL_CHARS
        or _contains_ascii_control(value)
        or "\\" in value
    ):
        return ""
    try:
        parsed = urlparse(value)
        scheme = parsed.scheme.lower()
        hostname = _canonical_hostname(parsed.hostname or "")
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
    result = urlunparse(
        (
            scheme,
            netloc,
            path,
            "",
            urlencode(sorted(query_items), doseq=True),
            "",
        )
    )
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
        self.max_pages = _bounded_int(max_pages, "max_pages", 1, _MAX_PAGES)
        self.max_pages_per_domain = _bounded_int(
            max_pages_per_domain,
            "max_pages_per_domain",
            1,
            _MAX_PAGES_PER_DOMAIN,
        )
        self.max_depth = _bounded_int(max_depth, "max_depth", 0, _MAX_DEPTH)
        self.request_delay = _bounded_float(
            request_delay,
            "request_delay",
            0.0,
            _MAX_DELAY_SECONDS,
        )
        self.timeout = _bounded_float(
            timeout,
            "timeout",
            0.1,
            _MAX_TIMEOUT_SECONDS,
        )
        if not isinstance(robots_fail_open, bool):
            raise ValueError("robots_fail_open must be a boolean.")
        domains = _bounded_collection(
            allowed_domains,
            _MAX_ALLOWED_DOMAINS,
            "allowed_domains",
        )
        canonical_domains: Set[str] = set()
        for item in domains:
            if not isinstance(item, str):
                raise ValueError("Every allowed domain must be a hostname string.")
            hostname = _allowed_domain(item)
            if not hostname:
                raise ValueError(
                    "Every allowed domain must be a canonical hostname-only authority."
                )
            canonical_domains.add(hostname)
        if not canonical_domains:
            raise ValueError("At least one valid allowed domain is required.")
        if not isinstance(user_agent, str):
            raise ValueError("user_agent must be a string.")

        self.allowed_domains = canonical_domains
        self.user_agent = _clean_header(user_agent, default=DEFAULT_USER_AGENT)
        self.robots_fail_open = robots_fail_open
        self.max_frontier_entries = min(max(self.max_pages * 10, 1000), 100_000)
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"User-Agent": self.user_agent})
        self._robots_cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    def __enter__(self) -> "AcademicCrawler":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _sanitize_links(self, value: object) -> List[str]:
        links: Set[str] = set()
        for raw_link in _bounded_items(value, _MAX_LINKS_PER_PAGE):
            normalized = normalize_url(raw_link) if isinstance(raw_link, str) else ""
            if normalized and is_trusted_domain(normalized, self.allowed_domains):
                links.add(normalized)
        return sorted(links)

    def _sanitize_page(self, page: object, url: str) -> Optional[Page]:
        if not isinstance(page, Page):
            return None
        normalized = normalize_url(url)
        if not normalized or not is_trusted_domain(normalized, self.allowed_domains):
            return None
        return Page(
            url=normalized,
            title=_safe_text(page.title, limit=500, default="Untitled").strip()
            or "Untitled",
            text=_safe_text(page.text, limit=_MAX_PAGE_TEXT_CHARS),
            links=self._sanitize_links(page.links),
            content_type=_safe_text(page.content_type, limit=200),
            content_length=_nonnegative_int(page.content_length),
        )

    def _bounded_frontier(self, value: object) -> deque[Tuple[str, int]]:
        queue: deque[Tuple[str, int]] = deque()
        for item in _bounded_items(value, self.max_frontier_entries):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            url = normalize_url(item[0]) if isinstance(item[0], str) else ""
            if not url or not is_trusted_domain(url, self.allowed_domains):
                continue
            depth = _nonnegative_int(item[1], self.max_depth + 1)
            queue.append((url, min(depth, self.max_depth + 1)))
        return queue

    def crawl(
        self,
        seeds: Iterable[str],
        state: Optional["CrawlState"] = None,
    ) -> "CrawlState":
        from storage import CrawlState as _CrawlState

        if state is None:
            state = _CrawlState.empty()
        if not isinstance(state, _CrawlState):
            raise ValueError("state must be a CrawlState instance.")

        pages: Dict[str, Page] = {}
        raw_pages = _safe_getattr(state, "pages", {})
        if isinstance(raw_pages, dict):
            try:
                page_items = itertools.islice(raw_pages.items(), self.max_pages)
            except Exception:
                page_items = iter(())
            for raw_url, raw_page in page_items:
                normalized = normalize_url(raw_url) if isinstance(raw_url, str) else ""
                page = self._sanitize_page(raw_page, normalized)
                if page is not None:
                    pages[normalized] = page

        graph: Dict[str, Set[str]] = {url: set() for url in pages}
        raw_graph = _safe_getattr(state, "graph", {})
        if isinstance(raw_graph, dict):
            try:
                graph_items = raw_graph.items()
            except Exception:
                graph_items = ()
            for raw_url, raw_edges in graph_items:
                url = normalize_url(raw_url) if isinstance(raw_url, str) else ""
                if url in pages:
                    graph[url].update(self._sanitize_links(raw_edges))

        visited: Set[str] = set(pages)
        for raw_url in _bounded_items(
            _safe_getattr(state, "visited", set()),
            self.max_pages + self.max_frontier_entries,
        ):
            normalized = normalize_url(raw_url) if isinstance(raw_url, str) else ""
            if normalized and is_trusted_domain(normalized, self.allowed_domains):
                visited.add(normalized)

        queue = self._bounded_frontier(_safe_getattr(state, "frontier", []))
        queued = {url for url, _depth in queue}
        seed_values = _bounded_collection(seeds, _MAX_SEEDS, "seeds")
        for raw_seed in seed_values:
            if not isinstance(raw_seed, str):
                raise ValueError("Every seed must be an HTTP or HTTPS URL string.")
            seed = normalize_url(raw_seed)
            if not seed:
                raise ValueError(
                    "Every seed must be a canonical credential-free HTTP or HTTPS URL."
                )
            if not is_trusted_domain(seed, self.allowed_domains):
                raise ValueError("Every seed must belong to the configured domain allowlist.")
            if seed not in visited and seed not in queued:
                if len(queue) >= self.max_frontier_entries:
                    break
                queue.append((seed, 0))
                queued.add(seed)

        domain_counts: Dict[str, int] = defaultdict(int)
        for url in pages:
            hostname = _hostname(url)
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
            if not self._under_domain_quota(current_url, domain_counts):
                continue
            if not self._is_allowed_by_robots(current_url):
                continue
            fetched = self._fetch_page(current_url)
            if fetched is None:
                continue
            canonical = normalize_url(fetched.url) or current_url
            if not is_trusted_domain(canonical, self.allowed_domains):
                continue
            visited.add(canonical)
            if canonical in pages or not self._under_domain_quota(
                canonical,
                domain_counts,
            ):
                continue
            if canonical != current_url and not self._is_allowed_by_robots(canonical):
                continue
            page = self._sanitize_page(fetched, canonical)
            if page is None:
                continue
            pages[canonical] = page
            graph.setdefault(canonical, set())
            hostname = _hostname(canonical)
            if hostname:
                domain_counts[hostname] += 1
            next_depth = depth + 1
            for link in page.links:
                graph[canonical].add(link)
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
            if not isinstance(downloaded.content, bytes):
                return None
            content = downloaded.content
            final_url = normalize_url(downloaded.final_url)
            headers = downloaded.headers
        except Exception:
            return None
        if not final_url or not is_trusted_domain(final_url, self.allowed_domains):
            return None
        if not isinstance(headers, dict):
            return None
        raw_content_type = headers.get("Content-Type", "")
        content_type_header = _safe_text(raw_content_type, limit=500)
        content_type = content_type_header.split(";", 1)[0].strip().lower()
        if content_type not in ALLOWED_MIME_TYPES:
            return None
        encoding = "utf-8"
        if "charset=" in content_type_header.lower():
            encoding = (
                content_type_header.lower()
                .split("charset=", 1)[1]
                .split(";", 1)[0]
                .strip()
            )
        try:
            html = content.decode(encoding, errors="replace")
        except LookupError:
            html = content.decode("utf-8", errors="replace")
        try:
            soup = BeautifulSoup(html, "html.parser")
            text = self._extract_text(soup)
        except Exception:
            return None
        if len(text) < MIN_CONTENT_LENGTH:
            return None
        return Page(
            final_url,
            self._extract_title(soup),
            text,
            self._extract_links(final_url, soup),
            content_type,
            len(content),
        )

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        try:
            if soup.title and soup.title.string:
                return _safe_text(soup.title.string, limit=500).strip() or "Untitled"
            heading = soup.find(["h1", "h2"])
            return (
                _safe_text(heading.get_text(" ", strip=True), limit=500)
                if heading
                else "Untitled"
            )
        except Exception:
            return "Untitled"

    @staticmethod
    def _extract_text(soup: BeautifulSoup) -> str:
        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav",
                "aside",
                "svg",
            ]
        ):
            element.decompose()
        return " ".join(soup.get_text(separator=" ", strip=True).split())[
            :_MAX_PAGE_TEXT_CHARS
        ]

    def _extract_links(self, base_url: str, soup: BeautifulSoup) -> List[str]:
        links: Set[str] = set()
        try:
            anchors = soup.find_all(
                "a",
                href=True,
                limit=_MAX_ANCHORS_INSPECTED,
            )
        except Exception:
            return []
        for anchor in anchors:
            try:
                href = _safe_text(anchor.get("href"), limit=_MAX_URL_CHARS)
            except Exception:
                continue
            absolute = normalize_url(urljoin(base_url, href)) if href else ""
            if (
                absolute
                and absolute != base_url
                and is_trusted_domain(absolute, self.allowed_domains)
            ):
                links.add(absolute)
                if len(links) >= _MAX_LINKS_PER_PAGE:
                    break
        return sorted(links)

    def _is_allowed_by_robots(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return False
        if not parsed.scheme or not parsed.netloc:
            return False
        if base not in self._robots_cache:
            parser: Optional[robotparser.RobotFileParser] = None
            robots_url = urljoin(base, "/robots.txt")
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
                if not isinstance(downloaded.content, bytes):
                    raise ValueError("Invalid robots response body.")
                parser = robotparser.RobotFileParser(robots_url)
                parser.parse(
                    downloaded.content.decode("utf-8", errors="replace").splitlines()
                )
            except Exception:
                parser = None
            self._robots_cache[base] = parser
        parser = self._robots_cache[base]
        if parser is None:
            return self.robots_fail_open
        try:
            return bool(parser.can_fetch(self.user_agent, url))
        except Exception:
            return self.robots_fail_open

    def _under_domain_quota(
        self,
        url: str,
        domain_counts: Dict[str, int],
    ) -> bool:
        hostname = _hostname(url)
        if not hostname or not isinstance(domain_counts, dict):
            return False
        value = domain_counts.get(hostname, 0)
        count = _nonnegative_int(value, self.max_pages_per_domain)
        return count < self.max_pages_per_domain


DEFAULT_SEEDS: List[str] = list(ALL_TRUSTED_SEEDS)
