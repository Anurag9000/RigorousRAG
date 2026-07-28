"""Security primitives shared by the HTTP, crawler, and tool layers."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, TypeAlias
from urllib.parse import urljoin, urlparse

import requests

IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address

DEFAULT_MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_REMOTE_DOWNLOAD_BYTES", str(5_000_000)))
DEFAULT_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50_000_000)))
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("REMOTE_REQUEST_TIMEOUT_SECONDS", "15"))
MAX_REDIRECTS = max(0, min(int(os.getenv("MAX_REMOTE_REDIRECTS", "4")), 20))

_OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_ALLOWED_REMOTE_METHODS = {"GET", "HEAD", "POST"}
_FORBIDDEN_CALLER_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_SENSITIVE_REDIRECT_HEADERS = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
    "api-key",
}


class SecurityError(ValueError):
    """Raised when an input violates an explicit security boundary."""


@dataclass(frozen=True)
class Principal:
    owner_id: str
    authenticated: bool


@dataclass(frozen=True)
class DownloadedResponse:
    final_url: str
    status_code: int
    headers: Mapping[str, str]
    content: bytes


def normalize_owner_id(owner_id: str) -> str:
    owner_id = (owner_id or "").strip()
    if not _OWNER_RE.fullmatch(owner_id):
        raise SecurityError(
            "Owner identifiers must be 1-128 characters containing only letters, "
            "numbers, '.', '_' or '-'."
        )
    return owner_id


def parse_api_key_owners() -> Dict[str, str]:
    """Load API-key-to-owner mappings without deriving identity from headers."""

    raw_mapping = os.getenv("API_KEY_OWNERS_JSON", "").strip()
    if raw_mapping:
        try:
            parsed = json.loads(raw_mapping)
        except json.JSONDecodeError as exc:
            raise RuntimeError("API_KEY_OWNERS_JSON must contain valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("API_KEY_OWNERS_JSON must be a JSON object.")
        result: Dict[str, str] = {}
        for api_key, owner_id in parsed.items():
            if not isinstance(api_key, str) or not api_key:
                raise RuntimeError("Every configured API key must be a non-empty string.")
            if not isinstance(owner_id, str):
                raise RuntimeError("Every configured owner ID must be a string.")
            result[api_key] = normalize_owner_id(owner_id)
        return result

    result: Dict[str, str] = {}
    for key in os.getenv("ALLOWED_API_KEYS", "").split(","):
        key = key.strip()
        if not key:
            continue
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        result[key] = f"api-{digest}"
    return result


def safe_upload_suffix(filename: Optional[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        raise SecurityError(
            f"Unsupported upload type '{suffix or 'none'}'. "
            f"Allowed: {', '.join(sorted(_ALLOWED_UPLOAD_SUFFIXES))}."
        )
    return suffix


def generated_upload_name(filename: Optional[str]) -> str:
    return safe_upload_suffix(filename)


def _resolved_addresses(hostname: str, port: Optional[int]) -> set[IPAddress]:
    try:
        infos = socket.getaddrinfo(hostname, port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SecurityError(f"Could not resolve remote hostname '{hostname}'.") from exc
    addresses: set[IPAddress] = set()
    for info in infos:
        try:
            addresses.add(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    if not addresses:
        raise SecurityError(f"Remote hostname '{hostname}' resolved to no usable address.")
    return addresses


def _is_public_address(address: IPAddress) -> bool:
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def validate_public_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise SecurityError("Only http:// and https:// URLs are allowed.")
    if not parsed.hostname:
        raise SecurityError("The URL must contain a hostname.")
    if parsed.username or parsed.password:
        raise SecurityError("Credentials embedded in URLs are not allowed.")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        raise SecurityError("Localhost destinations are not allowed.")
    try:
        addresses: set[IPAddress] = {ipaddress.ip_address(host)}
    except ValueError:
        addresses = _resolved_addresses(host, parsed.port)
    if any(not _is_public_address(address) for address in addresses):
        raise SecurityError("Private, local, reserved, and link-local destinations are blocked.")
    return parsed.geturl()


def hostname_matches(hostname: str, allowed_domains: Iterable[str]) -> bool:
    host = (hostname or "").rstrip(".").lower()
    for raw_domain in allowed_domains:
        parsed = urlparse(raw_domain if "://" in raw_domain else f"https://{raw_domain}")
        domain = (parsed.hostname or "").rstrip(".").lower()
        if domain and (host == domain or host.endswith(f".{domain}")):
            return True
    return False


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    return scheme, (parsed.hostname or "").rstrip(".").lower(), port


def _sanitize_request_headers(headers: Optional[Mapping[str, str]]) -> Dict[str, str]:
    sanitized: Dict[str, str] = {}
    for raw_name, raw_value in (headers or {}).items():
        name = str(raw_name).strip()
        value = str(raw_value).strip()
        lowered = name.lower()
        if not name or lowered in _FORBIDDEN_CALLER_HEADERS:
            raise SecurityError(f"Caller-controlled header '{name or 'empty'}' is not allowed.")
        if "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            raise SecurityError("Remote request headers may not contain line breaks.")
        sanitized[name] = value
    return sanitized


def _strip_cross_origin_secrets(headers: Mapping[str, str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if (
            lowered in _SENSITIVE_REDIRECT_HEADERS
            or lowered.endswith("-api-key")
            or lowered.endswith("-token")
        ):
            continue
        result[name] = value
    return result


def _socket_from_response(response: requests.Response) -> Any:
    """Best-effort extraction of the connected socket from urllib3/httplib layers."""

    raw = getattr(response, "raw", None)
    candidates = [
        getattr(getattr(raw, "_connection", None), "sock", None),
        getattr(getattr(raw, "connection", None), "sock", None),
        getattr(
            getattr(
                getattr(getattr(raw, "_original_response", None), "fp", None),
                "raw",
                None,
            ),
            "_sock",
            None,
        ),
        getattr(
            getattr(
                getattr(getattr(raw, "_fp", None), "fp", None),
                "raw",
                None,
            ),
            "_sock",
            None,
        ),
    ]
    return next((candidate for candidate in candidates if candidate is not None), None)


def _validate_connected_peer(response: requests.Response) -> IPAddress:
    """Validate the actual connected peer, closing the DNS-rebinding TOCTOU gap."""

    sock = _socket_from_response(response)
    if sock is None:
        raise SecurityError("Could not verify the connected remote address.")
    try:
        peer_host = sock.getpeername()[0]
        address = ipaddress.ip_address(peer_host)
    except (OSError, ValueError, TypeError, IndexError) as exc:
        raise SecurityError("Could not verify the connected remote address.") from exc
    if not _is_public_address(address):
        raise SecurityError("The connected remote address is private, local, or reserved.")
    return address


def safe_download(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[Mapping[str, str]] = None,
    data: Optional[bytes | str] = None,
    json_body: Optional[object] = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    allowed_content_types: Optional[Iterable[str]] = None,
    session: Optional[requests.Session] = None,
) -> DownloadedResponse:
    """Download a bounded public resource while revalidating every redirect."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    if timeout <= 0:
        raise ValueError("timeout must be positive.")
    current_method = method.upper().strip()
    if current_method not in _ALLOWED_REMOTE_METHODS:
        raise SecurityError(f"Remote method '{current_method or 'empty'}' is not allowed.")
    current_url = validate_public_url(url)
    current_headers = _sanitize_request_headers(headers)
    current_data = data
    current_json = json_body
    owned_session = session is None
    http = session or requests.Session()
    previous_trust_env = getattr(http, "trust_env", False)
    http.trust_env = False
    try:
        for _ in range(MAX_REDIRECTS + 1):
            response = http.request(
                method=current_method,
                url=current_url,
                headers=current_headers,
                data=current_data,
                json=current_json,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
            try:
                _validate_connected_peer(response)
                if response.status_code in _REDIRECT_CODES:
                    location = response.headers.get("Location")
                    if not location:
                        raise SecurityError(
                            "Redirect response did not include a Location header."
                        )
                    next_url = validate_public_url(urljoin(current_url, location))
                    cross_origin = _origin(next_url) != _origin(current_url)
                    if cross_origin:
                        current_headers = _strip_cross_origin_secrets(current_headers)
                        if response.status_code in {307, 308} and current_method not in {
                            "GET",
                            "HEAD",
                        }:
                            raise SecurityError(
                                "Cross-origin redirects may not replay a request body."
                            )
                    current_url = next_url
                    if response.status_code in {301, 302, 303} and current_method not in {
                        "GET",
                        "HEAD",
                    }:
                        current_method = "GET"
                        current_data = None
                        current_json = None
                    continue
                response.raise_for_status()
                content_type = (
                    response.headers.get("Content-Type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                if allowed_content_types:
                    allowed = {value.lower() for value in allowed_content_types}
                    if content_type and content_type not in allowed:
                        raise SecurityError(
                            f"Unsupported remote content type '{content_type}'."
                        )
                raw_length = response.headers.get("Content-Length")
                if raw_length:
                    try:
                        declared_length = int(raw_length)
                    except ValueError:
                        declared_length = 0
                    if declared_length > max_bytes:
                        raise SecurityError(
                            f"Remote response exceeds the {max_bytes}-byte limit."
                        )
                body = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise SecurityError(
                            f"Remote response exceeds the {max_bytes}-byte limit."
                        )
                final_url = validate_public_url(response.url or current_url)
                return DownloadedResponse(
                    final_url=final_url,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=bytes(body),
                )
            finally:
                response.close()
        raise SecurityError(f"Remote URL exceeded the {MAX_REDIRECTS}-redirect limit.")
    finally:
        http.trust_env = previous_trust_env
        if owned_session:
            http.close()
