from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

from Crawler import AcademicCrawler, Page, is_trusted_domain, normalize_url
from storage import CrawlState


def test_url_normalization_removes_fragments_tracking_and_default_ports():
    assert normalize_url("HTTPS://Example.COM:443/a/?utm_source=x&b=2&a=1#part") == "https://example.com/a?a=1&b=2"
    assert normalize_url("javascript:alert(1)") == ""
    assert normalize_url("http://") == ""


def test_domain_allowlist_uses_hostname_boundaries():
    assert is_trusted_domain("https://journals.example.edu/paper", ["example.edu"])
    assert not is_trusted_domain("https://example.edu.attacker.test", ["example.edu"])


def test_crawl_adds_new_seeds_even_with_saved_frontier():
    crawler = AcademicCrawler(
        allowed_domains=["a.test", "b.test"],
        max_pages=2,
        max_depth=0,
        request_delay=0,
        robots_fail_open=True,
    )
    crawler._is_allowed_by_robots = MagicMock(return_value=True)
    crawler._fetch_page = MagicMock(side_effect=lambda url: Page(url, url, "x" * 600, [], "text/html", 600))
    state = CrawlState.empty()
    state.frontier = [("https://a.test", 0)]
    result = crawler.crawl(["https://b.test"], state)
    assert set(result.pages) == {"https://a.test/", "https://b.test/"}


def test_rejected_url_is_recorded_as_attempted():
    crawler = AcademicCrawler(allowed_domains=["allowed.test"], request_delay=0)
    state = crawler.crawl(["https://blocked.test"])
    assert "https://blocked.test/" in state.visited
    assert not state.pages


def test_fetch_revalidates_final_redirect_host():
    crawler = AcademicCrawler(allowed_domains=["allowed.test"], request_delay=0)
    response = SimpleNamespace(
        final_url="https://attacker.test/page",
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=(b"<html><title>X</title><body>" + b"a" * 700 + b"</body></html>"),
    )
    with patch("Crawler.safe_download", return_value=response):
        assert crawler._fetch_page("https://allowed.test/page") is None


def test_link_extraction_is_deduplicated_and_trusted():
    crawler = AcademicCrawler(allowed_domains=["example.test"])
    soup = BeautifulSoup(
        '<a href="/a?utm_source=x">A</a><a href="/a">A2</a><a href="https://evil.test">E</a>',
        "html.parser",
    )
    assert crawler._extract_links("https://example.test/", soup) == ["https://example.test/a"]
