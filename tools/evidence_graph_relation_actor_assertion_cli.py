"""Operator CLI for provisioning and verifying signed review actor assertions."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from tools.evidence_graph_relation_actor_assertion import (
    sign_review_actor_assertion,
    verify_review_actor_assertion,
)

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MIN_KEY_BYTES = 32
_MAX_KEY_BYTES = 4096
_MAX_PATH = 4096


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _safe_path(value: str | os.PathLike[str], label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if _redirecting(info):
            raise ValueError(f"{label} may not contain redirects.")
    return absolute


def _read_key(path: str | os.PathLike[str]) -> bytes:
    selected = _safe_path(path, "key path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(selected, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or not _MIN_KEY_BYTES <= info.st_size <= _MAX_KEY_BYTES
        ):
            raise ValueError("key file has an unsupported size or type.")
        payload = os.read(descriptor, _MAX_KEY_BYTES + 1)
        if not _MIN_KEY_BYTES <= len(payload) <= _MAX_KEY_BYTES:
            raise ValueError("key file has an unsupported size.")
        return payload
    finally:
        os.close(descriptor)


def _duration(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("lifetime must be numeric.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("lifetime must be numeric.") from exc
    if not math.isfinite(selected) or not 60.0 <= selected <= 86_400.0:
        raise ValueError("lifetime must be between 60 and 86,400 seconds.")
    return selected


def _atomic_create_json(path: str | os.PathLike[str], value: dict[str, Any]) -> Path:
    destination = _safe_path(path, "output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_info = destination.parent.lstat()
    if _redirecting(parent_info) or not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError("output parent must be a regular directory.")
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("output assertion already exists.")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            destination.lstat()
        except FileNotFoundError:
            os.replace(temporary_path, destination)
        else:
            raise FileExistsError("output assertion appeared concurrently.")
        directory = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_relation_actor_assertion_cli",
        description=(
            "Create or verify short-lived HMAC-signed relation-review actor "
            "assertions. Key material is never printed."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sign = commands.add_parser("sign")
    sign.add_argument("--actor-id", required=True)
    sign.add_argument("--issuer", required=True)
    sign.add_argument("--key-path", required=True)
    sign.add_argument("--output", required=True)
    sign.add_argument("--lifetime-seconds", type=float, default=900.0)
    sign.add_argument("--nonce")

    verify = commands.add_parser("verify")
    verify.add_argument("--assertion-path", required=True)
    verify.add_argument("--key-path", required=True)
    verify.add_argument("--expected-issuer", required=True)
    verify.add_argument("--clock-skew-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "sign":
            issued_at = time.time()
            lifetime = _duration(args.lifetime_seconds)
            payload = sign_review_actor_assertion(
                actor_id=args.actor_id,
                issuer=args.issuer,
                issued_at=issued_at,
                expires_at=issued_at + lifetime,
                nonce=args.nonce or secrets.token_hex(16),
                key=_read_key(args.key_path),
            )
            destination = _atomic_create_json(args.output, payload)
            _print(
                {
                    "status": "created",
                    "output": str(destination),
                    "actor_id": payload["actor_id"],
                    "issuer": payload["issuer"],
                    "issued_at": payload["issued_at"],
                    "expires_at": payload["expires_at"],
                    "nonce": payload["nonce"],
                    "key_material_returned": False,
                }
            )
            return 0
        if args.command == "verify":
            value = verify_review_actor_assertion(
                assertion_path=args.assertion_path,
                key_path=args.key_path,
                expected_issuer=args.expected_issuer,
                clock_skew_seconds=args.clock_skew_seconds,
            )
            _print(
                {
                    "status": "valid",
                    "actor_id": value.actor_id,
                    "issuer": value.issuer,
                    "issued_at": value.issued_at,
                    "expires_at": value.expires_at,
                    "nonce": value.nonce,
                    "assertion_digest": value.assertion_digest,
                    "signature_digest": value.signature_digest,
                    "verified_at": value.verified_at,
                    "key_material_returned": False,
                }
            )
            return 0
        raise ValueError("unsupported actor assertion command.")
    except (FileExistsError, OSError, PermissionError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
