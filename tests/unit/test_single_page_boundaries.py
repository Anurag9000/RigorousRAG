from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.single_page as single_page


def test_malformed_page_byte_limit_is_rejected_before_network_access():
    with patch(
        "tools.single_page.safe_download",
        side_effect=AssertionError("network should not run"),
    ):
        with pytest.raises(ValueError, match="max_bytes must be an integer"):
            single_page.fetch_single_page(
                "https://example.test/article",
                max_bytes="not-an-integer",
            )
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            single_page.fetch_single_page(
                "https://example.test/article",
                max_bytes=0,
            )


def test_page_byte_limit_is_capped_at_configured_maximum():
    downloaded = SimpleNamespace(
        final_url="https://example.test/article",
        headers={"Content-Type": "text/plain"},
        content=b"evidence",
        status_code=200,
    )
    with patch("tools.single_page.safe_download", return_value=downloaded) as safe:
        page = single_page.fetch_single_page(
            "https://example.test/article",
            max_bytes=10**30,
        )

    assert page.error is None
    assert safe.call_args.kwargs["max_bytes"] == single_page._MAX_PAGE_BYTES


def test_user_agent_control_characters_are_removed_and_length_is_bounded():
    downloaded = SimpleNamespace(
        final_url="https://example.test/article",
        headers={"Content-Type": "text/plain"},
        content=b"evidence",
        status_code=200,
    )
    hostile = "Agent\r\nX-Evil: injected\x00" + ("x" * 5000)
    with patch("tools.single_page.safe_download", return_value=downloaded) as safe:
        page = single_page.fetch_single_page(
            "https://example.test/article",
            user_agent=hostile,
        )

    assert page.error is None
    user_agent = safe.call_args.kwargs["headers"]["User-Agent"]
    assert "\r" not in user_agent
    assert "\n" not in user_agent
    assert "\x00" not in user_agent
    assert len(user_agent) == 1200


def test_empty_page_url_returns_structured_failure_without_network():
    with patch(
        "tools.single_page.safe_download",
        side_effect=AssertionError("network should not run"),
    ):
        page = single_page.fetch_single_page("   ")

    assert page.error == "Page fetch failed (ValueError)."
    assert page.url == ""
    assert page.text == ""
