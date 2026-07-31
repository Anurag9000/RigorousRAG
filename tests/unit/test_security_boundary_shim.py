import json
from decimal import Decimal
from fractions import Fraction

import pytest

from tools import security


def test_api_key_json_rejects_duplicate_and_noncanonical_values(monkeypatch):
    for raw in (
        '{"duplicate":"alice","duplicate":"bob"}',
        '{" padded ":"alice"}',
        '{"bad\\tkey":"alice"}',
        '{"key":" alice "}',
        ' {"key":"alice"}',
    ):
        monkeypatch.setenv("API_KEY_OWNERS_JSON", raw)
        monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
        with pytest.raises(RuntimeError):
            security.parse_api_key_owners()


def test_legacy_api_keys_reject_padding_controls_and_duplicates(monkeypatch):
    monkeypatch.delenv("API_KEY_OWNERS_JSON", raising=False)
    for raw in ("alpha,alpha", "alpha, beta", " alpha", "bad\tkey"):
        monkeypatch.setenv("ALLOWED_API_KEYS", raw)
        with pytest.raises(RuntimeError):
            security.parse_api_key_owners()

    monkeypatch.setenv("ALLOWED_API_KEYS", "alpha,beta")
    owners = security.parse_api_key_owners()
    assert set(owners) == {"alpha", "beta"}
    assert owners["alpha"].startswith("api-")


def test_upload_suffix_rejects_every_ascii_control():
    for filename in (
        "bad\tname.pdf",
        "bad\nname.pdf",
        "bad\x1bname.pdf",
        "bad\x7fname.pdf",
    ):
        with pytest.raises(security.SecurityError):
            security.safe_upload_suffix(filename)

    assert security.safe_upload_suffix("paper.PDF") == ".pdf"


def test_public_url_rejects_padding_backslashes_and_del_before_dns(monkeypatch):
    monkeypatch.setattr(
        security,
        "_resolved_addresses",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("DNS must not run for structurally invalid URLs")
        ),
    )
    for url in (
        " https://example.test/",
        "https://example.test/ ",
        "https://example.test\\attacker.test/",
        "https://example.test/unsafe\x7fpath",
    ):
        with pytest.raises(security.SecurityError):
            security.validate_public_url(url)


def test_hostname_allowlist_requires_hostname_only_authority():
    assert security.hostname_matches(
        "papers.example.test",
        ["example.test"],
    )
    for allowed in (
        "https://example.test/path",
        "https://alice@example.test/",
        "example.test:443",
        "example.test?token=value",
        " example.test",
        "example.test\\attacker.test",
    ):
        assert not security.hostname_matches(
            "papers.example.test",
            [allowed],
        )


def test_request_headers_require_exact_control_free_strings():
    assert security._sanitize_request_headers({"X-Test": "value"}) == {
        "X-Test": "value"
    }
    for headers in (
        {" X-Test": "value"},
        {"X-Test": " padded "},
        {"X-Test": "bad\tvalue"},
        {"X-Test": "bad\x1bvalue"},
        {"X-Test": "bad\x7fvalue"},
    ):
        with pytest.raises(security.SecurityError):
            security._sanitize_request_headers(headers)


def test_response_headers_do_not_stringify_hostile_values():
    bounded = security._bounded_response_headers(
        {
            "X-Good": "value",
            "X-Bad": "bad\x7fvalue",
            object(): "ignored",
            "X-Object": object(),
            "Set-Cookie": "secret=value",
        }
    )
    assert bounded == {"X-Good": "value"}


def test_remote_numeric_limits_require_exact_integer_and_nonboolean_timeout():
    class ExactIndex:
        def __index__(self):
            return 7

    assert security._positive_integer(ExactIndex(), "limit", 10) == 7
    for value in (True, 1.0, Decimal("1.0"), Fraction(3, 2)):
        with pytest.raises(ValueError, match="integer"):
            security._positive_integer(value, "limit", 10)

    with pytest.raises(ValueError, match="numeric"):
        security._positive_timeout(True)
    assert security._positive_timeout(1) == 1.0


def test_boundary_module_is_loaded_once_and_public_functions_are_patched():
    assert security.parse_api_key_owners.__module__ == "tools.security_boundary"
    assert security.validate_public_url.__module__ == "tools.security_boundary"
    assert security.hostname_matches.__module__ == "tools.security_boundary"
