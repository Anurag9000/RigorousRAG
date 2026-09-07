#!/usr/bin/env python3
"""Checksum-pinned upstream dataset acquisition for RigorousRAG.

RigorousRAG's scientific import/materialization authorities intentionally operate on exact local
bytes. This upstream authority closes the acquisition leg without weakening that contract:
only explicitly declared HTTPS mirrors are fetched, every stream is bounded, bytes are accepted
only when their SHA-256 exactly matches the manifest, writes are atomic, existing matching files
are reused, and no archive extraction or implicit dataset discovery is performed.

The output of this command is still only an admitted local byte source. Grounded/dynamic/
benchmark/corpus/qrels governance CLIs remain responsible for licensing, schema conversion,
provenance and scientific publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "rigorousrag-authoritative-dataset-fetch/v1"
RECEIPT_SCHEMA = "rigorousrag-authoritative-dataset-fetch-receipt/v1"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HEX = frozenset("0123456789abcdef")
_MAX_ITEMS = 10_000
_DEFAULT_MAX_BYTES = 1_000_000_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected) or selected == "0" * 64:
        raise ValueError(f"{label} must be a non-placeholder SHA-256 digest")
    return selected


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > 300 or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.expanduser().absolute()
    parts = absolute.parts
    if not parts:
        raise ValueError(f"{label} path is invalid")
    current = Path(parts[0])
    if current.exists() and current.is_symlink():
        raise ValueError(f"{label} traverses a symlink: {current}")
    for part in parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{label} traverses a symlink: {current}")


def _safe_output(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a repository-relative path")
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError(f"{label} must be repository-relative")
    candidate = (_REPO_ROOT / raw).absolute()
    try:
        candidate.relative_to(_REPO_ROOT.absolute())
    except Exception as exc:
        raise ValueError(f"{label} escapes repository root") from exc
    _reject_symlink_components(candidate, label)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(_REPO_ROOT.resolve())
    except Exception as exc:
        raise ValueError(f"{label} resolves outside repository root") from exc
    if resolved.exists() and (resolved.is_symlink() or not resolved.is_file()):
        raise ValueError(f"{label} must be a regular file path")
    return resolved


def _https_urls(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > 32:
        raise ValueError(f"{label} must contain 1..32 HTTPS mirrors")
    result: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{label} entries must be non-empty strings")
        parsed = urllib.parse.urlsplit(raw.strip())
        if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError(f"{label} supports credential-free HTTPS URLs only")
        result.append(raw.strip())
    return tuple(result)


def _positive_int(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1,{maximum}]")
    return value


def _read_config(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    raw_path = Path(path).expanduser()
    _reject_symlink_components(raw_path, "dataset fetch config")
    selected = raw_path.resolve(strict=True)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("dataset fetch config must be a regular non-symlink file")
    try:
        raw = json.loads(selected.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError("dataset fetch config is not strict JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("dataset fetch config must contain an object")
    allowed = {"schema", "downloads", "receipt", "timeout_seconds"}
    if set(raw) - allowed or raw.get("schema") != SCHEMA:
        raise ValueError(f"dataset fetch config must use schema {SCHEMA!r} and closed fields")
    downloads = raw.get("downloads")
    if not isinstance(downloads, list) or not downloads or len(downloads) > _MAX_ITEMS:
        raise ValueError(f"downloads must contain 1..{_MAX_ITEMS} entries")
    return selected, raw


def _fetch_one(item: Mapping[str, Any], timeout: int) -> Mapping[str, Any]:
    allowed = {"id", "urls", "sha256", "output", "max_bytes"}
    if set(item) != allowed:
        raise ValueError(f"download item fields must be exactly {sorted(allowed)}")
    item_id = _identifier(item["id"], "download id")
    expected = _sha(item["sha256"], f"{item_id}.sha256")
    urls = _https_urls(item["urls"], f"{item_id}.urls")
    maximum = _positive_int(item["max_bytes"], f"{item_id}.max_bytes", _DEFAULT_MAX_BYTES)
    output = _safe_output(item["output"], f"{item_id}.output")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        actual = _sha_file(output)
        if actual != expected:
            raise ValueError(f"existing output for {item_id} has SHA-256 {actual}, expected {expected}; refusing overwrite")
        return {"id": item_id, "output": output.relative_to(_REPO_ROOT).as_posix(), "sha256": actual, "bytes": output.stat().st_size, "reused": True, "mirror": None}

    failures: list[str] = []
    for url in urls:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}-", suffix=".part", dir=str(output.parent))
        total = 0
        digest = hashlib.sha256()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "RigorousRAG-authoritative-dataset-fetch/1"})
            with urllib.request.urlopen(request, timeout=timeout) as response, os.fdopen(descriptor, "wb") as handle:
                final_url = urllib.parse.urlsplit(response.geturl())
                if final_url.scheme.lower() != "https":
                    raise ValueError("dataset fetch redirect left HTTPS")
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > maximum:
                        raise ValueError(f"download {item_id} exceeded max_bytes={maximum}")
                    digest.update(block)
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            actual = digest.hexdigest()
            if actual != expected:
                raise ValueError(f"SHA-256 mismatch: expected {expected}, got {actual}")
            os.replace(temporary_name, output)
            try:
                directory = os.open(output.parent, os.O_RDONLY)
            except Exception:
                directory = None
            if directory is not None:
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return {"id": item_id, "output": output.relative_to(_REPO_ROOT).as_posix(), "sha256": actual, "bytes": total, "reused": False, "mirror": url}
        except Exception as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    raise RuntimeError(f"all declared mirrors failed for {item_id}: " + " | ".join(failures))


def run_config(path: str | Path) -> Mapping[str, Any]:
    config_path, raw = _read_config(path)
    timeout = _positive_int(raw.get("timeout_seconds", 120), "timeout_seconds", 600)
    rows: list[Mapping[str, Any]] = []
    ids: set[str] = set()
    outputs: set[str] = set()
    for index, item in enumerate(raw["downloads"]):
        if not isinstance(item, Mapping):
            raise ValueError(f"downloads[{index}] must be an object")
        item_id = _identifier(item.get("id"), f"downloads[{index}].id")
        output = str(item.get("output") or "")
        if item_id in ids or output in outputs:
            raise ValueError("download IDs and output paths must be unique")
        ids.add(item_id)
        outputs.add(output)
        rows.append(_fetch_one(item, timeout))
    receipt_path = _safe_output(raw.get("receipt", "artifacts/dataset_fetch/receipt.json"), "receipt")
    payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "config": str(config_path),
        "config_sha256": _sha_file(config_path),
        "downloads": rows,
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{receipt_path.name}-", suffix=".tmp", dir=str(receipt_path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, receipt_path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help=f"{SCHEMA} JSON manifest")
    result = run_config(parser.parse_args(argv).config)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RECEIPT_SCHEMA", "SCHEMA", "main", "run_config"]
