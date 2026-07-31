from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
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


def test_out_of_allowlist_seed_is_rejected_before_network():
    crawler = AcademicCrawler(allowed_domains=["allowed.test"], request_delay=0)
    crawler._is_allowed_by_robots = MagicMock(
        side_effect=AssertionError("robots lookup must not run")
    )
    crawler._fetch_page = MagicMock(
        side_effect=AssertionError("page fetch must not run")
    )

    with pytest.raises(ValueError, match="allowlist"):
        crawler.crawl(["https://blocked.test"])

    crawler._is_allowed_by_robots.assert_not_called()
    crawler._fetch_page.assert_not_called()
    crawler.close()


def test_fetch_revalidates_final_redirect_host():
    crawler = AcademicCrawler(allowed_domains=["allowed.test"], request_delay=0)
    response = SimpleNamespace(
        final_url="https://attacker.test/page",
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=(b"<html><title>X</title><body>" + b"a" * 700 + b"</body></html>"),
    )
    with patch("Crawler.safe_download", return_value=response):
        assert crawler._fetch_page("https://allowed.test/page") is None


def test_redirected_canonical_target_is_not_fetched_twice():
    crawler = AcademicCrawler(
        allowed_domains=["allowed.test"],
        max_pages=5,
        max_depth=0,
        request_delay=0,
    )
    crawler._is_allowed_by_robots = MagicMock(return_value=True)
    final_url = "https://allowed.test/final"
    crawler._fetch_page = MagicMock(
        return_value=Page(final_url, "Final", "x" * 600, [], "text/html", 600)
    )

    result = crawler.crawl(
        ["https://allowed.test/start", final_url],
        CrawlState.empty(),
    )

    assert crawler._fetch_page.call_count == 1
    assert set(result.pages) == {final_url}
    assert final_url in result.visited


def test_redirected_target_rechecks_final_domain_quota():
    crawler = AcademicCrawler(
        allowed_domains=["a.test", "b.test"],
        max_pages=5,
        max_pages_per_domain=1,
        max_depth=0,
        request_delay=0,
    )
    crawler._is_allowed_by_robots = MagicMock(return_value=True)
    redirected = "https://b.test/new"
    crawler._fetch_page = MagicMock(
        return_value=Page(redirected, "New", "x" * 600, [], "text/html", 600)
    )
    state = CrawlState.empty()
    existing = "https://b.test/existing"
    state.pages[existing] = Page(existing, "Existing", "x" * 600, [], "text/html", 600)

    result = crawler.crawl(["https://a.test/start"], state)

    assert set(result.pages) == {existing}
    assert redirected in result.visited


def test_redirected_target_rechecks_final_robots_policy():
    crawler = AcademicCrawler(
        allowed_domains=["a.test", "b.test"],
        max_pages=5,
        max_depth=0,
        request_delay=0,
    )
    redirected = "https://b.test/private"
    crawler._fetch_page = MagicMock(
        return_value=Page(
            redirected,
            "Private",
            "x" * 600,
            ["https://b.test/child"],
            "text/html",
            600,
        )
    )
    crawler._is_allowed_by_robots = MagicMock(
        side_effect=lambda url: url != redirected
    )

    result = crawler.crawl(["https://a.test/start"], CrawlState.empty())

    assert result.pages == {}
    assert result.graph == {}
    assert redirected in result.visited
    assert [item.args for item in crawler._is_allowed_by_robots.call_args_list] == [
        ("https://a.test/start",),
        (redirected,),
    ]


def test_link_extraction_is_deduplicated_and_trusted():
    crawler = AcademicCrawler(allowed_domains=["example.test"])
    soup = BeautifulSoup(
        '<a href="/a?utm_source=x">A</a><a href="/a">A2</a><a href="https://evil.test">E</a>',
        "html.parser",
    )
    assert crawler._extract_links("https://example.test/", soup) == ["https://example.test/a"]
