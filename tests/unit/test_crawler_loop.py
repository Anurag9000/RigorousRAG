from unittest.mock import MagicMock, patch
from Crawler import AcademicCrawler, Page

class TestCrawlerLoop:
    def test_crawl_basic_loop(self):
        # Use simple domains that match netloc logic
        crawler = AcademicCrawler(max_pages=2, max_depth=1, allowed_domains=["a.com", "b.com"], request_delay=0)
        
        # Manual mocking
        crawler._under_domain_quota = MagicMock(return_value=True)
        crawler._is_allowed_by_robots = MagicMock(return_value=True)
        crawler._fetch_page = MagicMock(side_effect=[
            Page(url="http://a.com", title="A", text="Text"*130, links=["http://b.com"], content_type="text/html", content_length=1000),
            Page(url="http://b.com", title="B", text="Text"*130, links=[], content_type="text/html", content_length=1000)
        ])

        # Run crawl
        state = crawler.crawl(seeds=["http://a.com"])
        
        assert len(state.pages) == 2
        assert crawler._fetch_page.call_count == 2
        assert "http://a.com" in state.visited
        assert "http://b.com" in state.visited

    def test_robots_exclusion(self):
        crawler = AcademicCrawler(allowed_domains=["disallowed.com"], request_delay=0)
        crawler._is_allowed_by_robots = MagicMock(return_value=False)
        state = crawler.crawl(seeds=["http://disallowed.com"])
        assert len(state.pages) == 0
        # In current implementation, if robots check fails, it's not added to visited
        assert "http://disallowed.com" not in state.visited

    @patch('urllib.robotparser.RobotFileParser')
    def test_real_robots_logic(self, mock_rp_class):
        mock_rp = mock_rp_class.return_value
        mock_rp.can_fetch.return_value = True
        
        crawler = AcademicCrawler(request_delay=0)
        allowed = crawler._is_allowed_by_robots("http://example.com/page")
        assert allowed is True
        mock_rp.set_url.assert_called_with("http://example.com/robots.txt")

    def test_domain_quota(self):
        crawler = AcademicCrawler(max_pages_per_domain=100, request_delay=0)
        counts = {"example.com": 101}
        assert crawler._under_domain_quota("http://example.com/p", counts) is False
        assert crawler._under_domain_quota("http://other.com/p", counts) is True

    def test_max_depth_reached(self):
        crawler = AcademicCrawler(max_pages=10, max_depth=0, allowed_domains=["a.com"], request_delay=0)
        crawler._is_allowed_by_robots = MagicMock(return_value=True)
        crawler._under_domain_quota = MagicMock(return_value=True)
        crawler._fetch_page = MagicMock(return_value=Page("http://a.com", "A", "T"*600, ["http://b.com"], "text/html", 1000))
        
        state = crawler.crawl(seeds=["http://a.com"])
        assert len(state.pages) == 1
        # Should not have queued b.com because depth 0
        assert len(state.frontier) == 0
