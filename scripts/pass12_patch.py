"""One-shot exact-anchor patch for remediation pass twelve."""

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected patch anchor missing in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + addition.strip() + "\n", encoding="utf-8")


replace_once(
    "tools/security.py",
    '''class SecurityError(ValueError):\n    """Raised when an input violates an explicit security boundary."""\n\n\n@dataclass(frozen=True)\nclass Principal:''',
    '''class SecurityError(ValueError):\n    """Raised when an input violates an explicit security boundary."""\n\n\ndef _contains_ascii_control(value: str) -> bool:\n    return any(ord(character) < 32 or ord(character) == 127 for character in value)\n\n\ndef _strict_json_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:\n    result: Dict[str, Any] = {}\n    for key, value in pairs:\n        if key in result:\n            raise ValueError("Duplicate JSON object key.")\n        result[key] = value\n    return result\n\n\n@dataclass(frozen=True)\nclass Principal:''',
)
replace_once(
    "tools/security.py",
    '''    if (\n        not value\n        or len(value) > _MAX_API_KEY_CHARS\n        or any(character in value for character in ("\\x00", "\\r", "\\n"))\n    ):\n''',
    '''    if (\n        not value\n        or value != value.strip()\n        or len(value) > _MAX_API_KEY_CHARS\n        or _contains_ascii_control(value)\n    ):\n''',
)
replace_once(
    "tools/security.py",
    '''            parsed = json.loads(\n                raw_mapping,\n                parse_constant=lambda value: (_ for _ in ()).throw(\n                    ValueError(f"Non-standard JSON constant {value}")\n                ),\n            )\n''',
    '''            parsed = json.loads(\n                raw_mapping,\n                object_pairs_hook=_strict_json_object,\n                parse_constant=lambda value: (_ for _ in ()).throw(\n                    ValueError(f"Non-standard JSON constant {value}")\n                ),\n            )\n''',
)
replace_once(
    "tools/security.py",
    '''            if not isinstance(owner_id, str):\n                raise RuntimeError("Every configured owner ID must be a string.")\n            result[key] = normalize_owner_id(owner_id)\n''',
    '''            if not isinstance(owner_id, str):\n                raise RuntimeError("Every configured owner ID must be a string.")\n            if owner_id != owner_id.strip():\n                raise RuntimeError("Every configured owner ID must already be canonical.")\n            owner = normalize_owner_id(owner_id)\n            if owner != owner_id:\n                raise RuntimeError("Every configured owner ID must already be canonical.")\n            result[key] = owner\n''',
)
replace_once(
    "tools/security.py",
    '''    for raw_key in raw_legacy.split(","):\n        if not raw_key.strip():\n            continue\n        key = _api_key(raw_key.strip())\n        if len(result) >= _MAX_API_KEYS:\n''',
    '''    for raw_key in raw_legacy.split(","):\n        if raw_key == "":\n            continue\n        if raw_key != raw_key.strip():\n            raise RuntimeError("Legacy API keys must already be canonical.")\n        key = _api_key(raw_key)\n        if key in result:\n            raise RuntimeError("Legacy API keys must be unique.")\n        if len(result) >= _MAX_API_KEYS:\n''',
)
replace_once(
    "tools/security.py",
    '''def safe_upload_suffix(filename: Optional[str]) -> str:\n    if not isinstance(filename, str) or len(filename) > 500 or "\\x00" in filename:\n''',
    '''def safe_upload_suffix(filename: Optional[str]) -> str:\n    if (\n        not isinstance(filename, str)\n        or len(filename) > 500\n        or _contains_ascii_control(filename)\n    ):\n''',
)
replace_once(
    "tools/security.py",
    '''    if any(character.isspace() or ord(character) < 33 for character in value):\n''',
    '''    if any(\n        character.isspace() or ord(character) < 33 or ord(character) == 127\n        for character in value\n    ):\n''',
)
replace_once(
    "tools/security.py",
    '''    value = url.strip()\n    if not value or len(value) > _MAX_URL_CHARS:\n        raise SecurityError(f"URLs must contain 1-{_MAX_URL_CHARS} characters.")\n    if any(ord(character) < 32 or ord(character) == 127 for character in value):\n        raise SecurityError("URLs may not contain control characters.")\n''',
    '''    value = url\n    if (\n        not value\n        or value != value.strip()\n        or len(value) > _MAX_URL_CHARS\n    ):\n        raise SecurityError(\n            f"URLs must contain 1-{_MAX_URL_CHARS} canonical characters."\n        )\n    if _contains_ascii_control(value) or "\\\\" in value:\n        raise SecurityError(\n            "URLs may not contain control characters or backslashes."\n        )\n''',
)
replace_once(
    "tools/security.py",
    '''def hostname_matches(hostname: str, allowed_domains: Iterable[str]) -> bool:\n    host = _canonical_hostname(hostname)\n    if not host or isinstance(allowed_domains, (str, bytes, bytearray)):\n        return False\n    try:\n        candidates = itertools.islice(iter(allowed_domains), _MAX_ALLOWED_DOMAINS)\n    except Exception:\n        return False\n    for raw_domain in candidates:\n        if not isinstance(raw_domain, str):\n            continue\n        try:\n            parsed = urlparse(\n                raw_domain if "://" in raw_domain else f"https://{raw_domain}"\n            )\n        except ValueError:\n            continue\n        domain = _canonical_hostname(parsed.hostname or "")\n        if domain and (host == domain or host.endswith(f".{domain}")):\n            return True\n    return False\n''',
    '''def _allowed_domain(value: Any) -> str:\n    if (\n        not isinstance(value, str)\n        or not value\n        or value != value.strip()\n        or len(value) > 253\n        or _contains_ascii_control(value)\n        or "\\\\" in value\n    ):\n        return ""\n    try:\n        parsed = urlparse(value if "://" in value else f"https://{value}")\n        port = parsed.port\n    except (ValueError, UnicodeError):\n        return ""\n    if (\n        parsed.scheme.lower() not in {"http", "https"}\n        or parsed.username is not None\n        or parsed.password is not None\n        or port is not None\n        or parsed.path not in {"", "/"}\n        or parsed.params\n        or parsed.query\n        or parsed.fragment\n    ):\n        return ""\n    return _canonical_hostname(parsed.hostname or "")\n\n\ndef hostname_matches(hostname: str, allowed_domains: Iterable[str]) -> bool:\n    host = _canonical_hostname(hostname)\n    if not host or isinstance(allowed_domains, (str, bytes, bytearray)):\n        return False\n    try:\n        candidates = itertools.islice(iter(allowed_domains), _MAX_ALLOWED_DOMAINS)\n    except Exception:\n        return False\n    for raw_domain in candidates:\n        domain = _allowed_domain(raw_domain)\n        if domain and (host == domain or host.endswith(f".{domain}")):\n            return True\n    return False\n''',
)
replace_once(
    "tools/security.py",
    '''        name = raw_name.strip()\n        value = raw_value.strip()\n        lowered = name.lower()\n        if not _HEADER_NAME_RE.fullmatch(name):\n''',
    '''        name = raw_name\n        value = raw_value\n        if name != name.strip() or value != value.strip():\n            raise SecurityError(\n                "Remote request headers must already be canonical."\n            )\n        lowered = name.lower()\n        if not _HEADER_NAME_RE.fullmatch(name):\n''',
)
replace_once(
    "tools/security.py",
    '''        if "\\r" in value or "\\n" in value or "\\x00" in value:\n            raise SecurityError(\n                "Remote request headers may not contain control characters."\n            )\n''',
    '''        if _contains_ascii_control(value):\n            raise SecurityError(\n                "Remote request headers may not contain control characters."\n            )\n''',
)
replace_once(
    "tools/security.py",
    '''        try:\n            name = str(raw_name)[:200]\n            value = str(raw_value)[:_MAX_HEADER_VALUE_CHARS]\n        except Exception:\n            continue\n        if name.lower() in _SENSITIVE_RESPONSE_HEADERS:\n            continue\n        bounded[name] = value\n''',
    '''        if not isinstance(raw_name, str) or not isinstance(raw_value, str):\n            continue\n        name = raw_name[:200]\n        value = raw_value[:_MAX_HEADER_VALUE_CHARS]\n        if (\n            not _HEADER_NAME_RE.fullmatch(name)\n            or _contains_ascii_control(value)\n            or name.lower() in _SENSITIVE_RESPONSE_HEADERS\n        ):\n            continue\n        bounded[name] = value\n''',
)

append_once(
    "tests/unit/test_security_complete_boundaries.py",
    "test_duplicate_and_noncanonical_api_key_configuration_is_rejected",
    '''def test_duplicate_and_noncanonical_api_key_configuration_is_rejected(monkeypatch):\n    for raw in (\n        '{"duplicate":"alice","duplicate":"bob"}',\n        '{" padded ":"alice"}',\n        '{"bad\\tkey":"alice"}',\n        '{"key":" alice "}',\n    ):\n        monkeypatch.setenv("API_KEY_OWNERS_JSON", raw)\n        with pytest.raises(RuntimeError):\n            parse_api_key_owners()\n\n    monkeypatch.delenv("API_KEY_OWNERS_JSON", raising=False)\n    for raw in ("alpha,alpha", "alpha, beta", "bad\\tkey"):\n        monkeypatch.setenv("ALLOWED_API_KEYS", raw)\n        with pytest.raises(RuntimeError):\n            parse_api_key_owners()\n\n\ndef test_upload_names_urls_domains_and_headers_reject_ambiguous_inputs(monkeypatch):\n    for filename in ("bad\\tname.pdf", "bad\\x1bname.pdf", "bad\\x7fname.pdf"):\n        with pytest.raises(SecurityError):\n            safe_upload_suffix(filename)\n\n    monkeypatch.setattr(\n        security.socket,\n        "getaddrinfo",\n        lambda *_args, **_kwargs: (_ for _ in ()).throw(\n            AssertionError("DNS must not run for invalid URLs")\n        ),\n    )\n    for url in (\n        " https://example.test/",\n        "https://example.test/ ",\n        "https://example.test\\\\attacker.test/",\n    ):\n        with pytest.raises(SecurityError):\n            validate_public_url(url)\n\n    assert not hostname_matches("papers.example.test", ["https://example.test/path"] )\n    assert not hostname_matches("papers.example.test", ["https://alice@example.test/"])\n    assert not hostname_matches("papers.example.test", ["example.test:443"])\n    assert not hostname_matches("papers.example.test", ["example.test?token=value"])\n\n\ndef test_request_headers_are_canonical_and_response_headers_are_string_only(monkeypatch):\n    monkeypatch.setattr(security.socket, "getaddrinfo", _public_dns)\n    session = FakeSession([])\n    for headers in (\n        {"X-Test": " padded "},\n        {" X-Test": "value"},\n        {"X-Test": "bad\\tvalue"},\n        {"X-Test": "bad\\x1bvalue"},\n        {"X-Test": "bad\\x7fvalue"},\n    ):\n        with pytest.raises(SecurityError):\n            safe_download(\n                "https://example.test/",\n                headers=headers,\n                session=session,\n            )\n    assert session.calls == []\n\n    bounded = security._bounded_response_headers(\n        {\n            "X-Good": "value",\n            "X-Bad": "bad\\x7fvalue",\n            object(): "ignored",\n            "X-Object": object(),\n        }\n    )\n    assert bounded == {"X-Good": "value"}\n''',
)
append_once(
    "tests/unit/test_security.py",
    "test_legacy_api_key_duplicates_and_padding_are_rejected",
    '''def test_legacy_api_key_duplicates_and_padding_are_rejected(monkeypatch):\n    monkeypatch.delenv("API_KEY_OWNERS_JSON", raising=False)\n    for raw in ("alpha,alpha", "alpha, beta", " alpha"):\n        monkeypatch.setenv("ALLOWED_API_KEYS", raw)\n        with pytest.raises(RuntimeError):\n            parse_api_key_owners()\n''',
)
