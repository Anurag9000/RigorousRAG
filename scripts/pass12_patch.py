from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected patch anchor missing in {path}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + addition.strip() + "\n", encoding="utf-8")


replace_once(
    "tools/security.py",
    '''def _api_key(value: Any) -> str:\n    if not isinstance(value, str):\n        raise RuntimeError("Every configured API key must be a string.")\n    if (\n        not value\n        or len(value) > _MAX_API_KEY_CHARS\n        or any(character in value for character in ("\\x00", "\\r", "\\n"))\n    ):\n        raise RuntimeError(\n            f"Every configured API key must contain 1-{_MAX_API_KEY_CHARS} valid characters."\n        )\n    return value\n\n\ndef parse_api_key_owners() -> Dict[str, str]:\n''',
    '''def _api_key(value: Any) -> str:\n    if not isinstance(value, str):\n        raise RuntimeError("Every configured API key must be a string.")\n    if (\n        not value\n        or value != value.strip()\n        or len(value) > _MAX_API_KEY_CHARS\n        or any(ord(character) < 32 or ord(character) == 127 for character in value)\n    ):\n        raise RuntimeError(\n            f"Every configured API key must contain 1-{_MAX_API_KEY_CHARS} canonical valid characters."\n        )\n    return value\n\n\ndef _unique_json_object(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:\n    result: Dict[str, Any] = {}\n    for key, value in pairs:\n        if key in result:\n            raise ValueError("Duplicate JSON object key")\n        result[key] = value\n    return result\n\n\ndef parse_api_key_owners() -> Dict[str, str]:\n''',
)

replace_once(
    "tools/security.py",
    '''            parsed = json.loads(\n                raw_mapping,\n                parse_constant=lambda value: (_ for _ in ()).throw(\n                    ValueError(f"Non-standard JSON constant {value}")\n                ),\n            )\n''',
    '''            parsed = json.loads(\n                raw_mapping,\n                object_pairs_hook=_unique_json_object,\n                parse_constant=lambda value: (_ for _ in ()).throw(\n                    ValueError(f"Non-standard JSON constant {value}")\n                ),\n            )\n''',
)

replace_once(
    "tools/security.py",
    '''            if not isinstance(owner_id, str):\n                raise RuntimeError("Every configured owner ID must be a string.")\n            result[key] = normalize_owner_id(owner_id)\n''',
    '''            if not isinstance(owner_id, str):\n                raise RuntimeError("Every configured owner ID must be a string.")\n            normalized_owner = normalize_owner_id(owner_id)\n            if owner_id != normalized_owner:\n                raise RuntimeError("Every configured owner ID must already be canonical.")\n            result[key] = normalized_owner\n''',
)

replace_once(
    "tools/security.py",
    '''    for raw_key in raw_legacy.split(","):\n        if not raw_key.strip():\n            continue\n        key = _api_key(raw_key.strip())\n        if len(result) >= _MAX_API_KEYS:\n            raise RuntimeError(\n                f"ALLOWED_API_KEYS may contain at most {_MAX_API_KEYS} keys."\n            )\n        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]\n        result[key] = f"api-{digest}"\n''',
    '''    for raw_key in raw_legacy.split(","):\n        if raw_key == "":\n            continue\n        key = _api_key(raw_key)\n        if key in result:\n            raise RuntimeError("ALLOWED_API_KEYS may not contain duplicate keys.")\n        if len(result) >= _MAX_API_KEYS:\n            raise RuntimeError(\n                f"ALLOWED_API_KEYS may contain at most {_MAX_API_KEYS} keys."\n            )\n        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]\n        result[key] = f"api-{digest}"\n''',
)

replace_once(
    "tools/security.py",
    '''def safe_upload_suffix(filename: Optional[str]) -> str:\n    if not isinstance(filename, str) or len(filename) > 500 or "\\x00" in filename:\n        raise SecurityError("Upload filenames must contain at most 500 valid characters.")\n''',
    '''def safe_upload_suffix(filename: Optional[str]) -> str:\n    if (\n        not isinstance(filename, str)\n        or len(filename) > 500\n        or any(ord(character) < 32 or ord(character) == 127 for character in filename)\n    ):\n        raise SecurityError("Upload filenames must contain at most 500 valid characters.")\n''',
)

replace_once(
    "tools/security.py",
    '''    if any(character.isspace() or ord(character) < 33 for character in value):\n        return ""\n''',
    '''    if any(\n        character.isspace() or ord(character) < 33 or ord(character) == 127\n        for character in value\n    ):\n        return ""\n''',
)

replace_once(
    "tools/security.py",
    '''def validate_public_url(url: str) -> str:\n    if not isinstance(url, str):\n        raise SecurityError("URLs must be strings.")\n    value = url.strip()\n    if not value or len(value) > _MAX_URL_CHARS:\n        raise SecurityError(f"URLs must contain 1-{_MAX_URL_CHARS} characters.")\n    if any(ord(character) < 32 or ord(character) == 127 for character in value):\n        raise SecurityError("URLs may not contain control characters.")\n''',
    '''def validate_public_url(url: str) -> str:\n    if not isinstance(url, str):\n        raise SecurityError("URLs must be strings.")\n    value = url\n    if (\n        not value\n        or value != value.strip()\n        or len(value) > _MAX_URL_CHARS\n    ):\n        raise SecurityError(f"URLs must contain 1-{_MAX_URL_CHARS} canonical characters.")\n    if (\n        "\\\\" in value\n        or any(\n            character.isspace() or ord(character) < 32 or ord(character) == 127\n            for character in value\n        )\n    ):\n        raise SecurityError("URLs may not contain whitespace, controls, or backslashes.")\n''',
)

append_once(
    "tests/unit/test_security_complete_boundaries.py",
    "test_api_key_configuration_rejects_duplicates_padding_and_controls",
    '''def test_api_key_configuration_rejects_duplicates_padding_and_controls(monkeypatch):\n    for raw in (\n        '{"duplicate":"alice","duplicate":"bob"}',\n        json.dumps({" padded": "alice"}),\n        json.dumps({"bad\\x1bkey": "alice"}),\n        json.dumps({"key": " alice "}),\n    ):\n        monkeypatch.setenv("API_KEY_OWNERS_JSON", raw)\n        with pytest.raises(RuntimeError):\n            parse_api_key_owners()\n\n    monkeypatch.delenv("API_KEY_OWNERS_JSON", raising=False)\n    for raw in ("alpha, beta", "alpha,alpha", "alpha\\x7f,beta"):\n        monkeypatch.setenv("ALLOWED_API_KEYS", raw)\n        with pytest.raises(RuntimeError):\n            parse_api_key_owners()\n\n\ndef test_upload_names_and_public_urls_reject_all_ambiguous_characters(monkeypatch):\n    for filename in ("paper\\n.pdf", "paper\\x1b.pdf", "paper\\x7f.pdf"):\n        with pytest.raises(SecurityError):\n            safe_upload_suffix(filename)\n\n    for url in (\n        " https://example.test/",\n        "https://example.test/ ",\n        "https://example.test/a b",\n        "https://example.test\\\\attacker.test/",\n        "https://example\\x7f.test/",\n    ):\n        with pytest.raises(SecurityError):\n            validate_public_url(url)\n\n    monkeypatch.setattr(security.socket, "getaddrinfo", _public_dns)\n    assert validate_public_url("https://EXAMPLE.test/a%20b#fragment") == (\n        "https://example.test/a%20b"\n    )\n''',
)

append_once(
    "tests/unit/test_security.py",
    "test_legacy_api_key_list_requires_canonical_unique_entries",
    '''def test_legacy_api_key_list_requires_canonical_unique_entries(monkeypatch):\n    monkeypatch.delenv("API_KEY_OWNERS_JSON", raising=False)\n    for value in ("alpha, beta", "alpha,alpha", "alpha\\n,beta"):\n        monkeypatch.setenv("ALLOWED_API_KEYS", value)\n        with pytest.raises(RuntimeError):\n            parse_api_key_owners()\n''',
)
