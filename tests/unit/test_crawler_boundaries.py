import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

import Crawler
from Crawler import AcademicCrawler, normalize_url


def test_url_normalization_rejects_credentials_controls_and_query_bombs():
    assert normalize_url("https://alice:password@example.test/a") == ""
    assert normalize_url("https://example.test/a\r\nInjected: yes") == ""
    assert normalize_url("https://example.test/?" + "&".join(f"k{i}=v" for i in range(201))) == ""
    assert normalize_url("https://[2606:4700:4700::1111]:443/a/") == (
        "https://[2606:4700:4700::1111]/a"
    )


def test_constructor_rejects_nonfinite_and_malformed_limits():
    with pytest.raises(ValueError, match="finite"):
        AcademicCrawler(allowed_domains=["example.test"], request_delay=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        AcademicCrawler(allowed_domains=["example.test"], timeout=float("inf"))
    with pytest.raises(ValueError, match="integer"):
        AcademicCrawler(allowed_domains=["example.test"], max_pages="bad")
    with pytest.raises(ValueError, match="iterable of hostnames"):
        AcademicCrawler(allowed_domains="example.test")
    with pytest.raises(ValueError, match="At least one"):
        AcademicCrawler(allowed_domains=["not a hostname / path"])


def test_allowed_domain_and_seed_generators_are_bounded(monkeypatch):
    monkeypatch.setattr(Crawler, "_MAX_ALLOWED_DOMAINS", 3)
    with pytest.raises(ValueError, match="at most 3"):
        AcademicCrawler(allowed_domains=(f"{index}.test" for index in itertools.count()))

    crawler = AcademicCrawler(
        allowed_domains=["example.test"],
        request_delay=0,
        robots_fail_open=True,
    )
    monkeypatch.setattr(Crawler, "_MAX_SEEDS", 3)
    with pytest.raises(ValueError, match="at most 3"):
        crawler.crawl((f"https://example.test/{index}" for index in itertools.count()))
    crawler.close()


def test_user_agent_control_characters_are_removed_before_requests():
    crawler = AcademicCrawler(
        allowed_domains=["example.test"],
        request_delay=0,
        user_agent="Agent\r\nInjected: yes",
    )
    response = SimpleNamespace(
        final_url="https://example.test/page",
        headers={"Content-Type": "text/html"},
        content=b"<html><body>" + b"evidence " * 100 + b"</body></html>",
    )
    with patch("Crawler.safe_download", return_value=response) as download:
        assert crawler._fetch_page("https://example.test/page") is not None
    header = download.call_args.kwargs["headers"]["User-Agent"]
    assert "\r" not in header and "\n" not in header
    assert len(header) <= Crawler._MAX_USER_AGENT_CHARS
    crawler.close()


def test_link_extraction_caps_inspected_anchors_and_unique_links(monkeypatch):
    monkeypatch.setattr(Crawler, "_MAX_ANCHORS_INSPECTED", 5)
    monkeypatch.setattr(Crawler, "_MAX_LINKS_PER_PAGE", 3)
    crawler = AcademicCrawler(allowed_domains=["example.test"])
    soup = BeautifulSoup(
        "".join(f'<a href="/{index}">{index}</a>' for index in range(20)),
        "html.parser",
    )

    links = crawler._extract_links("https://example.test/", soup)

    assert len(links) == 3
    assert all(link.startswith("https://example.test/") for link in links)
    crawler.close()


def test_updated_allowlist_drops_persisted_pages_from_removed_domains():
    from Crawler import Page
    from storage import CrawlState

    state = CrawlState(
        pages={
            "https://old.test/": Page(
                "https://old.test/",
                "Old",
                "old evidence",
                [],
                "text/html",
                12,
            )
        },
        graph={"https://old.test/": set()},
        visited={"https://old.test/"},
        frontier=[],
    )
    crawler = AcademicCrawler(
        allowed_domains=["new.test"],
        max_pages=1,
        request_delay=0,
        robots_fail_open=True,
    )

    result = crawler.crawl([], state)

    assert result.pages == {}
    assert result.graph == {}
    crawler.close()
