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
    assert normalize_url(
        "https://example.test/?" + "&".join(f"k{i}=v" for i in range(201))
    ) == ""
    assert normalize_url("https://[2606:4700:4700::1111]:443/a/") == (
        "https://[2606:4700:4700::1111]/a"
    )


def test_constructor_rejects_nonfinite_malformed_and_boolean_limits():
    with pytest.raises(ValueError, match="finite"):
        AcademicCrawler(allowed_domains=["example.test"], request_delay=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        AcademicCrawler(allowed_domains=["example.test"], timeout=float("inf"))
    with pytest.raises(ValueError, match="integer"):
        AcademicCrawler(allowed_domains=["example.test"], max_pages="bad")
    with pytest.raises(ValueError, match="iterable collection"):
        AcademicCrawler(allowed_domains="example.test")
    with pytest.raises(ValueError, match="valid hostname"):
        AcademicCrawler(allowed_domains=["not a hostname / path"])
    with pytest.raises(ValueError, match="boolean"):
        AcademicCrawler(allowed_domains=["example.test"], robots_fail_open="yes")
    with pytest.raises(ValueError, match="numeric"):
        AcademicCrawler(allowed_domains=["example.test"], request_delay=True)
    with pytest.raises(ValueError, match="user_agent must be a string"):
        AcademicCrawler(allowed_domains=["example.test"], user_agent=object())


def test_mixed_allowlist_does_not_silently_drop_invalid_entries():
    with pytest.raises(ValueError, match="valid hostname"):
        AcademicCrawler(allowed_domains=["example.test", "not a host / path"])
    with pytest.raises(ValueError, match="hostname string"):
        AcademicCrawler(allowed_domains=["example.test", object()])


def test_allowed_domain_and_seed_generators_are_bounded(monkeypatch):
    monkeypatch.setattr(Crawler, "_MAX_ALLOWED_DOMAINS", 3)
    with pytest.raises(ValueError, match="at most 3"):
        AcademicCrawler(
            allowed_domains=(f"{index}.test" for index in itertools.count())
        )

    crawler = AcademicCrawler(
        allowed_domains=["example.test"],
        request_delay=0,
        robots_fail_open=True,
    )
    monkeypatch.setattr(Crawler, "_MAX_SEEDS", 3)
    with pytest.raises(ValueError, match="at most 3"):
        crawler.crawl(
            (f"https://example.test/{index}" for index in itertools.count())
        )
    crawler.close()


def test_hostile_allowed_domain_and_seed_iterators_fail_safely():
    class BrokenIterable:
        def __iter__(self):
            raise RuntimeError("private iterator detail")

    with pytest.raises(ValueError, match="safely iterable"):
        AcademicCrawler(allowed_domains=BrokenIterable())

    crawler = AcademicCrawler(allowed_domains=["example.test"])
    with pytest.raises(ValueError, match="safely iterable"):
        crawler.crawl(BrokenIterable())
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


def test_nonbyte_or_malformed_download_response_is_rejected():
    crawler = AcademicCrawler(allowed_domains=["example.test"], request_delay=0)
    for response in (
        SimpleNamespace(
            final_url="https://example.test/page",
            headers={"Content-Type": "text/html"},
            content="not bytes",
        ),
        SimpleNamespace(
            final_url="https://example.test/page",
            headers=object(),
            content=b"evidence " * 100,
        ),
    ):
        with patch("Crawler.safe_download", return_value=response):
            assert crawler._fetch_page("https://example.test/page") is None
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


def test_malformed_persisted_page_fields_are_safely_coerced():
    from Crawler import Page
    from storage import CrawlState

    page = Page(
        "https://example.test/",
        "Title",
        "evidence",
        "https://example.test/character-iteration-must-not-happen",
        object(),
        "not-an-integer",
    )
    state = CrawlState(
        pages={"https://example.test/": page},
        graph={"https://example.test/": "not-an-edge-collection"},
        visited=set(),
        frontier=[],
    )
    crawler = AcademicCrawler(
        allowed_domains=["example.test"],
        max_pages=1,
        request_delay=0,
        robots_fail_open=True,
    )

    result = crawler.crawl([], state)

    stored = result.pages["https://example.test/"]
    assert stored.links == []
    assert stored.content_length == 0
    assert len(stored.content_type) <= 200
    assert result.graph == {"https://example.test/": set()}
    crawler.close()


def test_invalid_state_type_is_rejected_before_state_access():
    crawler = AcademicCrawler(allowed_domains=["example.test"])
    with pytest.raises(ValueError, match="CrawlState"):
        crawler.crawl([], object())
    crawler.close()


def test_invalid_seed_values_are_not_silently_discarded():
    crawler = AcademicCrawler(allowed_domains=["example.test"])
    for seeds in ([object()], ["file:///private/source"], ["https://a:b@example.test"]):
        with pytest.raises(ValueError):
            crawler.crawl(seeds)
    crawler.close()
