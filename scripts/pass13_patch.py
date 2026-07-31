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
    "tools/security_boundary.py",
    '''_original_safe_upload_suffix = _implementation.safe_upload_suffix\n_original_validate_public_url = _implementation.validate_public_url\n''',
    '''if not hasattr(_implementation, "_boundary_original_safe_upload_suffix"):\n    _implementation._boundary_original_safe_upload_suffix = (\n        _implementation.safe_upload_suffix\n    )\nif not hasattr(_implementation, "_boundary_original_validate_public_url"):\n    _implementation._boundary_original_validate_public_url = (\n        _implementation.validate_public_url\n    )\n\n_original_safe_upload_suffix = (\n    _implementation._boundary_original_safe_upload_suffix\n)\n_original_validate_public_url = (\n    _implementation._boundary_original_validate_public_url\n)\n''',
)

replace_once(
    "tools/security_boundary.py",
    '''    for raw_domain in candidates:\n        domain = _allowed_domain(raw_domain)\n        if domain and (host == domain or host.endswith(f".{domain}")):\n            return True\n    return False\n''',
    '''    try:\n        for raw_domain in candidates:\n            domain = _allowed_domain(raw_domain)\n            if domain and (host == domain or host.endswith(f".{domain}")):\n                return True\n    except Exception:\n        return False\n    return False\n''',
)

replace_once(
    "tools/security_boundary.py",
    '''    if len(headers) > _implementation._MAX_REQUEST_HEADERS:\n        raise _implementation.SecurityError(\n            f"At most {_implementation._MAX_REQUEST_HEADERS} request headers are allowed."\n        )\n    sanitized: Dict[str, str] = {}\n    for raw_name, raw_value in headers.items():\n        if not isinstance(raw_name, str) or not isinstance(raw_value, str):\n            raise _implementation.SecurityError(\n                "Remote request header names and values must be strings."\n            )\n        if raw_name != raw_name.strip() or raw_value != raw_value.strip():\n            raise _implementation.SecurityError(\n                "Remote request headers must already be canonical."\n            )\n        lowered = raw_name.lower()\n        if not _implementation._HEADER_NAME_RE.fullmatch(raw_name):\n            raise _implementation.SecurityError(\n                "Remote request header names contain invalid characters."\n            )\n        if lowered in _implementation._FORBIDDEN_CALLER_HEADERS:\n            raise _implementation.SecurityError(\n                f"Caller-controlled header '{raw_name}' is not allowed."\n            )\n        if len(raw_value) > _implementation._MAX_HEADER_VALUE_CHARS:\n            raise _implementation.SecurityError(\n                "Remote request header values exceed the size limit."\n            )\n        if _contains_ascii_control(raw_value):\n            raise _implementation.SecurityError(\n                "Remote request headers may not contain control characters."\n            )\n        sanitized[raw_name] = raw_value\n    return sanitized\n''',
    '''    try:\n        header_count = len(headers)\n    except Exception as exc:\n        raise _implementation.SecurityError(\n            "Remote request headers are invalid."\n        ) from exc\n    if header_count > _implementation._MAX_REQUEST_HEADERS:\n        raise _implementation.SecurityError(\n            f"At most {_implementation._MAX_REQUEST_HEADERS} request headers are allowed."\n        )\n    sanitized: Dict[str, str] = {}\n    try:\n        items = headers.items()\n        for raw_name, raw_value in items:\n            if not isinstance(raw_name, str) or not isinstance(raw_value, str):\n                raise _implementation.SecurityError(\n                    "Remote request header names and values must be strings."\n                )\n            if raw_name != raw_name.strip() or raw_value != raw_value.strip():\n                raise _implementation.SecurityError(\n                    "Remote request headers must already be canonical."\n                )\n            lowered = raw_name.lower()\n            if not _implementation._HEADER_NAME_RE.fullmatch(raw_name):\n                raise _implementation.SecurityError(\n                    "Remote request header names contain invalid characters."\n                )\n            if lowered in _implementation._FORBIDDEN_CALLER_HEADERS:\n                raise _implementation.SecurityError(\n                    f"Caller-controlled header '{raw_name}' is not allowed."\n                )\n            if len(raw_value) > _implementation._MAX_HEADER_VALUE_CHARS:\n                raise _implementation.SecurityError(\n                    "Remote request header values exceed the size limit."\n                )\n            if _contains_ascii_control(raw_value):\n                raise _implementation.SecurityError(\n                    "Remote request headers may not contain control characters."\n                )\n            sanitized[raw_name] = raw_value\n    except _implementation.SecurityError:\n        raise\n    except Exception as exc:\n        raise _implementation.SecurityError(\n            "Remote request headers are invalid."\n        ) from exc\n    return sanitized\n''',
)

replace_once(
    "tools/security_boundary.py",
    '''            name = raw_name[:200]\n            value = raw_value[: _implementation._MAX_HEADER_VALUE_CHARS]\n            if (\n                not _implementation._HEADER_NAME_RE.fullmatch(name)\n                or _contains_ascii_control(value)\n                or name.lower() in _implementation._SENSITIVE_RESPONSE_HEADERS\n            ):\n                continue\n            bounded[name] = value\n''',
    '''            if (\n                len(raw_name) > 200\n                or len(raw_value) > _implementation._MAX_HEADER_VALUE_CHARS\n            ):\n                continue\n            name = raw_name\n            value = raw_value\n            if (\n                not _implementation._HEADER_NAME_RE.fullmatch(name)\n                or _contains_ascii_control(value)\n                or name.lower() in _implementation._SENSITIVE_RESPONSE_HEADERS\n            ):\n                continue\n            bounded[name] = value\n''',
)

append_once(
    "tests/unit/test_security_boundary_shim.py",
    "test_boundary_reload_is_idempotent_and_nonrecursive",
    '''def test_boundary_reload_is_idempotent_and_nonrecursive(monkeypatch):\n    import importlib\n\n    from tools import security_boundary\n\n    original_upload = security._boundary_original_safe_upload_suffix\n    original_url = security._boundary_original_validate_public_url\n    for _ in range(3):\n        importlib.reload(security_boundary)\n\n    assert security._boundary_original_safe_upload_suffix is original_upload\n    assert security._boundary_original_validate_public_url is original_url\n    assert security.safe_upload_suffix("paper.PDF") == ".pdf"\n    monkeypatch.setattr(\n        security,\n        "_resolved_addresses",\n        lambda *_args: {security.ipaddress.ip_address("93.184.216.34")},\n    )\n    assert security.validate_public_url("https://example.test/paper") == (\n        "https://example.test/paper"\n    )\n\n\ndef test_hostname_matcher_contains_hostile_iterators():\n    class HostileDomains:\n        def __iter__(self):\n            yield "other.test"\n            raise RuntimeError("private iterator failure")\n\n    assert security.hostname_matches("papers.example.test", HostileDomains()) is False\n\n\ndef test_request_header_mapping_failures_are_generic_and_contained():\n    from collections.abc import Mapping\n\n    class HostileLength(Mapping):\n        def __getitem__(self, key):\n            raise KeyError(key)\n\n        def __iter__(self):\n            return iter(())\n\n        def __len__(self):\n            raise RuntimeError("private length failure")\n\n    with pytest.raises(security.SecurityError, match="headers are invalid") as captured:\n        security._sanitize_request_headers(HostileLength())\n    assert "private length" not in str(captured.value)\n\n\ndef test_response_headers_reject_oversized_names_and_values_without_collisions():\n    long_name = "X-" + "A" * 250\n    long_value = "v" * (security._MAX_HEADER_VALUE_CHARS + 1)\n    bounded = security._bounded_response_headers(\n        {\n            "X-Good": "value",\n            long_name: "ignored",\n            "X-Long": long_value,\n        }\n    )\n    assert bounded == {"X-Good": "value"}\n''',
)
