"""Fail-fast format/size guard for authoritative local dataset import CLIs.

Several legacy semantic adapters support both JSON arrays and streaming JSONL/TREC.  The
promotion-grade publishers reuse those semantics, but a multi-gigabyte JSON array would still
require whole-document parsing in the legacy helper.  Production CLIs call this guard before
semantic parsing: monolithic JSON is allowed only below an explicit bounded convenience size;
large corpora must use a streaming format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_MONOLITHIC_JSON_BYTES = 128 * 1024 * 1024
_STREAMING_FORMATS = frozenset({"jsonl", "trec"})


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


def _walk(value: Any) -> None:
    if isinstance(value, Mapping):
        if "source_path" in value and "input_format" in value:
            raw_path = value["source_path"]
            raw_format = value["input_format"]
            if not isinstance(raw_path, (str, Path)) or not isinstance(raw_format, str):
                raise ValueError("source_path/input_format fields have invalid types")
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
                # The semantic importer owns the definitive format allowlist.  Do not silently
                # bless an unknown format here; simply leave it for the strict downstream
                # schema validator.
                pass
        for child in value.values():
            _walk(child)
    elif isinstance(value, list):
        for child in value:
            _walk(child)


def assert_authoritative_input_scale(config_path: str | Path) -> None:
    _walk(_read_config(config_path))


__all__ = ["assert_authoritative_input_scale"]
