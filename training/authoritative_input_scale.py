"""Fail-fast format/size guard for authoritative local dataset import CLIs.

Promotion-grade publishers may reuse semantic adapters that support both monolithic JSON and
streaming JSONL/TREC. Production CLIs call this guard before semantic parsing: monolithic JSON
is allowed only below an explicit convenience bound; large corpora must use a streaming format.
The guard recognizes the repository's supported local source/format field aliases while leaving
the definitive semantic format allowlist to the downstream importer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_MONOLITHIC_JSON_BYTES = 128 * 1024 * 1024
_STREAMING_FORMATS = frozenset({"jsonl", "trec"})
_SOURCE_KEYS = ("source_path", "path", "source")
_FORMAT_KEYS = ("input_format", "format")


def _read_config(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(
        path,
        label="authoritative import config",
        must_exist=True,
        require_file=True,
    )
    size = source.stat().st_size
    if size <= 0 or size > _MAX_CONFIG_BYTES:
        raise ValueError("authoritative import config exceeds byte safety bound")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("authoritative import config is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("authoritative import config must contain an object")
    return value


def _first_key(value: Mapping[str, Any], candidates: tuple[str, ...]) -> str | None:
    matches = [key for key in candidates if key in value]
    if len(matches) > 1:
        raise ValueError(
            "authoritative import source object contains ambiguous equivalent fields: "
            + ",".join(matches)
        )
    return matches[0] if matches else None


def _walk(value: Any) -> None:
    if isinstance(value, Mapping):
        source_key = _first_key(value, _SOURCE_KEYS)
        format_key = _first_key(value, _FORMAT_KEYS)
        if source_key is not None and format_key is not None:
            raw_path = value[source_key]
            raw_format = value[format_key]
            if not isinstance(raw_path, (str, Path)) or not isinstance(raw_format, str):
                raise ValueError("authoritative source path/format fields have invalid types")
            selected_format = raw_format.strip().lower()
            source = safe_advanced_path(
                raw_path,
                label="authoritative import source",
                must_exist=True,
                require_file=True,
            )
            if selected_format == "json":
                if source.stat().st_size > _MAX_MONOLITHIC_JSON_BYTES:
                    raise ValueError(
                        "monolithic JSON source exceeds the authoritative whole-document "
                        f"limit of {_MAX_MONOLITHIC_JSON_BYTES} bytes; convert the local "
                        "artifact to JSONL before promotion-grade import"
                    )
            elif selected_format not in _STREAMING_FORMATS:
                # The semantic importer owns the definitive format allowlist. Unknown formats
                # are deliberately left for that strict downstream validator.
                pass
        for child in value.values():
            _walk(child)
    elif isinstance(value, list):
        for child in value:
            _walk(child)


def assert_authoritative_input_scale(config_path: str | Path) -> None:
    _walk(_read_config(config_path))


__all__ = ["assert_authoritative_input_scale"]
