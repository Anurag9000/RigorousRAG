"""Security primitives shared by the HTTP, crawler, and tool layers."""

from __future__ import annotations

import hashlib
import ipaddress
import itertools
import json
import math
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, TypeAlias
from urllib.parse import urljoin, urlparse, urlunparse

import requests

from tools.config import bounded_float_env, bounded_int_env

IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address

DEFAULT_MAX_DOWNLOAD_BYTES = bounded_int_env(
    "MAX_REMOTE_DOWNLOAD_BYTES",
    5_000_000,
    minimum=1,
    maximum=1_000_000_000,
)
DEFAULT_MAX_UPLOAD_BYTES = bounded_int_env(
    "MAX_UPLOAD_BYTES",
    50_000_000,
    minimum=1,
    maximum=1_000_000_000,
)
DEFAULT_REQUEST_TIMEOUT = bounded_float_env(
    "REMOTE_REQUEST_TIMEOUT_SECONDS",
    15.0,
    minimum=0.1,
    maximum=300.0,
)
MAX_REDIRECTS = bounded_int_env(
    "MAX_REMOTE_REDIRECTS",
    4,
    minimum=0,
    maximum=20,
)
MAX_REMOTE_REQUEST_BODY_BYTES = bounded_int_env(
    "MAX_REMOTE_REQUEST_BODY_BYTES",
    1_000_000,
    minimum=1,
    maximum=20_000_000,
)

_OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HOST_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,200}$")
_CONTENT_TYPE_RE = re.compile(
    r"^[A-Za-z0-9!#$&^_.+-]{1,127}/[A-Za-z0-9!#$&^_.+-]{1,127}$"
)
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
_SENSITIVE_RESPONSE_HEADERS = {
    "authorization",
    "proxy-authenticate",
    "set-cookie",
    "set-cookie2",
    "www-authenticate",
}
_MAX_API_KEY_CONFIG_BYTES = 1_000_000
_MAX_API_KEYS = 10_000
_MAX_API_KEY_CHARS = 4096
_MAX_URL_CHARS = 8192
_MAX_ALLOWED_DOMAINS = 1000
_MAX_REQUEST_HEADERS = 100
_MAX_HEADER_VALUE_CHARS = 8192
_MAX_RESPONSE_HEADERS = 200
_MAX_DNS_ADDRESSES = 64
_INJECTED_SESSION_ENV_LOCK = threading.RLock()
_MISSING = object()


class SecurityError(ValueError):
    """Raised when an input violates an explicit security boundary."""


@dataclass(frozen=True)
class Principal:
    owner_id: str
    authenticated: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        if not isinstance(self.authenticated, bool):
            raise SecurityError("authenticated must be a boolean.")


@dataclass(frozen=True)
class DownloadedResponse:
    final_url: str
    status_code: int
    headers: Mapping[str, str]
    content: bytes


def normalize_owner_id(owner_id: str) -> str:
    if not isinstance(owner_id, str):
        raise SecurityError("Owner identifiers must be strings.")
    owner_id = owner_id.strip()
    if not _OWNER_RE.fullmatch(owner_id):
        raise SecurityError(
            "Owner identifiers must be 1-128 characters containing only letters, "
            "numbers, '.', '_' or '-'."
        )
    return owner_id


def _api_key(value: Any) -> str:
    if not isinstance(value, str):
        raise RuntimeError("Every configured API key must be a string.")
    if (
        not value
        or len(value) > _MAX_API_KEY_CHARS
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise RuntimeError(
            f"Every configured API key must contain 1-{_MAX_API_KEY_CHARS} valid characters."
        )
    return value


def parse_api_key_owners() -> Dict[str, str]:
    """Load a bounded API-key-to-owner mapping without caller-derived identity."""

    raw_mapping = os.getenv("API_KEY_OWNERS_JSON", "")
    if len(raw_mapping.encode("utf-8", errors="ignore")) > _MAX_API_KEY_CONFIG_BYTES:
        raise RuntimeError("API_KEY_OWNERS_JSON exceeds the configuration byte limit.")
    if raw_mapping.strip():
        try:
            parsed = json.loads(
                raw_mapping,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"Non-standard JSON constant {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise RuntimeError("API_KEY_OWNERS_JSON must contain valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("API_KEY_OWNERS_JSON must be a JSON object.")
        if len(parsed) > _MAX_API_KEYS:
            raise RuntimeError(
                f"API_KEY_OWNERS_JSON may contain at most {_MAX_API_KEYS} keys."
            )
        result: Dict[str, str] = {}
        for api_key, owner_id in parsed.items():
            key = _api_key(api_key)
            if not isinstance(owner_id, str):
                raise RuntimeError("Every configured owner ID must be a string.")
            result[key] = normalize_owner_id(owner_id)
        return result

    result: Dict[str, str] = {}
    raw_legacy = os.getenv("ALLOWED_API_KEYS", "")
    if len(raw_legacy.encode("utf-8", errors="ignore")) > _MAX_API_KEY_CONFIG_BYTES:
        raise RuntimeError("ALLOWED_API_KEYS exceeds the configuration byte limit.")
    for raw_key in raw_legacy.split(","):
        if not raw_key.strip():
            continue
        key = _api_key(raw_key.strip())
        if len(result) >= _MAX_API_KEYS:
            raise RuntimeError(
                f"ALLOWED_API_KEYS may contain at most {_MAX_API_KEYS} keys."
            )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        result[key] = f"api-{digest}"
    return result


def safe_upload_suffix(filename: Optional[str]) -> str:
    if not isinstance(filename, str) or len(filename) > 500 or "\x00" in filename:
        raise SecurityError("Upload filenames must contain at most 500 valid characters.")
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        raise SecurityError(
            f"Unsupported upload type '{suffix or 'none'}'. "
            f"Allowed: {', '.join(sorted(_ALLOWED_UPLOAD_SUFFIXES))}."
        )
    return suffix


def generated_upload_name(filename: Optional[str]) -> str:
    return safe_upload_suffix(filename)


def _canonical_hostname(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 253:
        return ""
    if any(character.isspace() or ord(character) < 33 for character in value):
        return ""
    candidate = value.rstrip(".").lower()
    try:
        return ipaddress.ip_address(candidate).compressed.lower()
    except ValueError:
        pass
    try:
        ascii_host = candidate.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return ""
    if len(ascii_host) > 253:
        return ""
    labels = ascii_host.split(".")
    if not labels or any(not _HOST_LABEL_RE.fullmatch(label) for label in labels):
        return ""
    return ascii_host


def _resolved_addresses(hostname: str, port: int) -> set[IPAddress]:
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SecurityError(f"Could not resolve remote hostname '{hostname}'.") from exc
    addresses: set[IPAddress] = set()
    for info in infos[:_MAX_DNS_ADDRESSES]:
        try:
            addresses.add(ipaddress.ip_address(info[4][0].split("%", 1)[0]))
        except (ValueError, TypeError, IndexError):
            continue
    if not addresses:
        raise SecurityError(f"Remote hostname '{hostname}' resolved to no usable address.")
    return addresses


def _is_public_address(address: IPAddress) -> bool:
    return bool(
        address.is_global
        and not address.is_multicast
        and not address.is_unspecified
    )


def validate_public_url(url: str) -> str:
    if not isinstance(url, str):
        raise SecurityError("URLs must be strings.")
    value = url.strip()
    if not value or len(value) > _MAX_URL_CHARS:
        raise SecurityError(f"URLs must contain 1-{_MAX_URL_CHARS} characters.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SecurityError("URLs may not contain control characters.")
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as exc:
        raise SecurityError("The URL contains an invalid port.") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise SecurityError("Only http:// and https:// URLs are allowed.")
    hostname = _canonical_hostname(parsed.hostname or "")
    if not hostname:
        raise SecurityError("The URL must contain a valid hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise SecurityError("Credentials embedded in URLs are not allowed.")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(
        ".localhost"
    ):
        raise SecurityError("Localhost destinations are not allowed.")
    selected_port = port or (443 if scheme == "https" else 80)
    try:
        addresses: set[IPAddress] = {ipaddress.ip_address(hostname)}
    except ValueError:
        addresses = _resolved_addresses(hostname, selected_port)
    if any(not _is_public_address(address) for address in addresses):
        raise SecurityError(
            "Private, local, reserved, and link-local destinations are blocked."
        )
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = rendered_host
    if port is not None:
        netloc = f"{rendered_host}:{port}"
    return urlunparse(
        (scheme, netloc, parsed.path, parsed.params, parsed.query, "")
    )


def hostname_matches(hostname: str, allowed_domains: Iterable[str]) -> bool:
    host = _canonical_hostname(hostname)
    if not host or isinstance(allowed_domains, (str, bytes, bytearray)):
        return False
    try:
        candidates = itertools.islice(iter(allowed_domains), _MAX_ALLOWED_DOMAINS)
    except Exception:
        return False
    for raw_domain in candidates:
        if not isinstance(raw_domain, str):
            continue
        try:
            parsed = urlparse(
                raw_domain if "://" in raw_domain else f"https://{raw_domain}"
            )
        except ValueError:
            continue
        domain = _canonical_hostname(parsed.hostname or "")
        if domain and (host == domain or host.endswith(f".{domain}")):
            return True
    return False


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    return scheme, _canonical_hostname(parsed.hostname or ""), port


def _sanitize_request_headers(
    headers: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        raise SecurityError("Remote request headers must be a mapping.")
    if len(headers) > _MAX_REQUEST_HEADERS:
        raise SecurityError(
            f"At most {_MAX_REQUEST_HEADERS} request headers are allowed."
        )
    sanitized: Dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            raise SecurityError(
                "Remote request header names and values must be strings."
            )
        name = raw_name.strip()
        value = raw_value.strip()
        lowered = name.lower()
        if not _HEADER_NAME_RE.fullmatch(name):
            raise SecurityError(
                "Remote request header names contain invalid characters."
            )
        if lowered in _FORBIDDEN_CALLER_HEADERS:
            raise SecurityError(f"Caller-controlled header '{name}' is not allowed.")
        if len(value) > _MAX_HEADER_VALUE_CHARS:
            raise SecurityError(
                "Remote request header values exceed the size limit."
            )
        if "\r" in value or "\n" in value or "\x00" in value:
            raise SecurityError(
                "Remote request headers may not contain control characters."
            )
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
            or lowered.endswith("-secret")
        ):
            continue
        result[name] = value
    return result


def _sanitize_content_types(
    values: Optional[Iterable[str]],
) -> Optional[set[str]]:
    if values is None:
        return None
    if isinstance(values, (str, bytes, bytearray)):
        raise SecurityError(
            "allowed_content_types must be an iterable of MIME types."
        )
    try:
        raw_values = list(itertools.islice(iter(values), 101))
    except Exception as exc:
        raise SecurityError("allowed_content_types must be iterable.") from exc
    if len(raw_values) > 100:
        raise SecurityError("At most 100 allowed content types may be supplied.")
    allowed: set[str] = set()
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            raise SecurityError("Allowed content types must be strings.")
        value = raw_value.split(";", 1)[0].strip().lower()
        if not _CONTENT_TYPE_RE.fullmatch(value):
            raise SecurityError(
                "Allowed content types must be valid MIME types."
            )
        allowed.add(value)
    if not allowed:
        raise SecurityError("At least one allowed content type is required.")
    return allowed


def _request_body(
    data: Optional[bytes | str],
    json_body: Optional[object],
    headers: Dict[str, str],
) -> tuple[Optional[bytes], None]:
    if data is not None and json_body is not None:
        raise SecurityError("Supply either data or json_body, not both.")
    if json_body is not None:
        try:
            encoded = json.dumps(
                json_body,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise SecurityError(
                "json_body is not a supported JSON value."
            ) from exc
        headers.setdefault("Content-Type", "application/json")
    elif data is None:
        encoded = None
    elif isinstance(data, str):
        encoded = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        encoded = bytes(data)
    else:
        raise SecurityError("Remote request data must be bytes or text.")
    if encoded is not None and len(encoded) > MAX_REMOTE_REQUEST_BODY_BYTES:
        raise SecurityError(
            "Remote request body exceeds the configured byte limit."
        )
    return encoded, None


def _socket_from_response(response: requests.Response) -> Any:
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
    return next(
        (candidate for candidate in candidates if candidate is not None),
        None,
    )


def _validate_connected_peer(response: requests.Response) -> IPAddress:
    """Validate the actual connected peer, closing the DNS-rebinding TOCTOU gap."""

    sock = _socket_from_response(response)
    if sock is None:
        raise SecurityError("Could not verify the connected remote address.")
    try:
        peer_host = sock.getpeername()[0]
        address = ipaddress.ip_address(str(peer_host).split("%", 1)[0])
    except (OSError, ValueError, TypeError, IndexError) as exc:
        raise SecurityError(
            "Could not verify the connected remote address."
        ) from exc
    if not _is_public_address(address):
        raise SecurityError(
            "The connected remote address is private, local, or reserved."
        )
    return address


def _bounded_response_headers(headers: Any) -> Dict[str, str]:
    if not isinstance(headers, Mapping):
        return {}
    bounded: Dict[str, str] = {}
    for index, (raw_name, raw_value) in enumerate(headers.items()):
        if index >= _MAX_RESPONSE_HEADERS:
            break
        try:
            name = str(raw_name)[:200]
            value = str(raw_value)[:_MAX_HEADER_VALUE_CHARS]
        except Exception:
            continue
        if name.lower() in _SENSITIVE_RESPONSE_HEADERS:
            continue
        bounded[name] = value
    return bounded


def _positive_integer(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer.")
    if not 1 <= numeric <= maximum:
        raise ValueError(f"{label} must be between 1 and {maximum}.")
    return numeric


def _positive_timeout(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timeout must be numeric.") from exc
    if not math.isfinite(numeric) or not 0.1 <= numeric <= 300.0:
        raise ValueError(
            "timeout must be finite and between 0.1 and 300 seconds."
        )
    return numeric


def _empty_session_value(name: str) -> Any:
    if name == "trust_env":
        return False
    if name == "headers":
        return requests.structures.CaseInsensitiveDict()
    if name == "cookies":
        return requests.cookies.RequestsCookieJar()
    if name in {"proxies", "params"}:
        return {}
    if name == "hooks":
        return requests.hooks.default_hooks()
    if name == "verify":
        return True
    if name in {"auth", "cert"}:
        return None
    raise KeyError(name)


def _neutralize_injected_session(http: Any) -> Dict[str, Any]:
    """Temporarily remove ambient authority while retaining transport adapters."""

    state: Dict[str, Any] = {}
    changed: list[str] = []
    for name in (
        "trust_env",
        "proxies",
        "auth",
        "headers",
        "cookies",
        "params",
        "hooks",
        "verify",
        "cert",
    ):
        try:
            previous = getattr(http, name, _MISSING)
        except Exception as exc:
            _restore_injected_session(http, state, changed)
            raise SecurityError(
                "Injected HTTP session state could not be inspected safely."
            ) from exc
        if previous is _MISSING:
            continue
        state[name] = previous
        try:
            setattr(http, name, _empty_session_value(name))
        except Exception as exc:
            _restore_injected_session(http, state, changed)
            raise SecurityError(
                "Injected HTTP session state could not be isolated safely."
            ) from exc
        changed.append(name)
    return state


def _restore_injected_session(
    http: Any,
    state: Mapping[str, Any],
    names: Optional[Iterable[str]] = None,
) -> None:
    restore_names = list(names if names is not None else state.keys())
    failed = False
    for name in reversed(restore_names):
        if name not in state:
            continue
        try:
            setattr(http, name, state[name])
        except Exception:
            failed = True
    if failed:
        raise SecurityError(
            "Injected HTTP session state could not be restored safely."
        )


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
    """Download one bounded public resource with an end-to-end time budget."""

    response_limit = _positive_integer(
        max_bytes,
        "max_bytes",
        1_000_000_000,
    )
    timeout_value = _positive_timeout(timeout)
    if not isinstance(method, str):
        raise SecurityError("Remote request methods must be strings.")
    current_method = method.upper().strip()
    if current_method not in _ALLOWED_REMOTE_METHODS:
        raise SecurityError(
            f"Remote method '{current_method or 'empty'}' is not allowed."
        )
    current_url = validate_public_url(url)
    current_headers = _sanitize_request_headers(headers)
    current_data, current_json = _request_body(
        data,
        json_body,
        current_headers,
    )
    allowed = _sanitize_content_types(allowed_content_types)
    deadline = time.monotonic() + timeout_value
    owned_session = session is None
    http = session or requests.Session()
    injected_lock_acquired = False
    injected_state: Dict[str, Any] = {}
    restore_error: Optional[BaseException] = None
    if not owned_session:
        _INJECTED_SESSION_ENV_LOCK.acquire()
        injected_lock_acquired = True
    try:
        if owned_session:
            http.trust_env = False
        else:
            injected_state = _neutralize_injected_session(http)
        for _ in range(MAX_REDIRECTS + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SecurityError(
                    "Remote request exceeded the configured time limit."
                )
            response = http.request(
                method=current_method,
                url=current_url,
                headers=current_headers,
                data=current_data,
                json=current_json,
                timeout=max(0.1, min(timeout_value, remaining)),
                allow_redirects=False,
                stream=True,
            )
            try:
                _validate_connected_peer(response)
                status_code = int(response.status_code)
                if status_code in _REDIRECT_CODES:
                    location = response.headers.get("Location")
                    if not isinstance(location, str) or not location.strip():
                        raise SecurityError(
                            "Redirect response did not include a valid Location header."
                        )
                    next_url = validate_public_url(
                        urljoin(current_url, location)
                    )
                    cross_origin = _origin(next_url) != _origin(current_url)
                    if cross_origin:
                        current_headers = _strip_cross_origin_secrets(
                            current_headers
                        )
                        if (
                            status_code in {307, 308}
                            and current_method not in {"GET", "HEAD"}
                        ):
                            raise SecurityError(
                                "Cross-origin redirects may not replay a request body."
                            )
                    current_url = next_url
                    if (
                        status_code in {301, 302, 303}
                        and current_method not in {"GET", "HEAD"}
                    ):
                        current_method = "GET"
                        current_data = None
                        current_json = None
                        current_headers = {
                            name: value
                            for name, value in current_headers.items()
                            if name.lower()
                            not in {"content-type", "content-encoding"}
                        }
                    continue
                response.raise_for_status()
                content_type = (
                    str(response.headers.get("Content-Type", ""))
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                if allowed is not None and content_type not in allowed:
                    raise SecurityError(
                        "Remote response omitted or returned an unsupported content type."
                    )
                raw_length = response.headers.get("Content-Length")
                if raw_length not in (None, ""):
                    try:
                        declared_length = int(raw_length)
                    except (TypeError, ValueError, OverflowError):
                        declared_length = -1
                    if declared_length > response_limit:
                        raise SecurityError(
                            "Remote response exceeds the byte limit."
                        )
                chunks: list[bytes] = []
                total = 0
                for raw_chunk in response.iter_content(
                    chunk_size=64 * 1024
                ):
                    if time.monotonic() > deadline:
                        raise SecurityError(
                            "Remote request exceeded the configured time limit."
                        )
                    if not raw_chunk:
                        continue
                    if not isinstance(
                        raw_chunk,
                        (bytes, bytearray, memoryview),
                    ):
                        raise SecurityError(
                            "Remote response chunks must contain bytes."
                        )
                    chunk = bytes(raw_chunk)
                    total += len(chunk)
                    if total > response_limit:
                        raise SecurityError(
                            "Remote response exceeds the byte limit."
                        )
                    chunks.append(chunk)
                return DownloadedResponse(
                    final_url=current_url,
                    status_code=status_code,
                    headers=_bounded_response_headers(response.headers),
                    content=b"".join(chunks),
                )
            finally:
                response.close()
        raise SecurityError("Remote request exceeded the redirect limit.")
    finally:
        if not owned_session and injected_state:
            try:
                _restore_injected_session(http, injected_state)
            except BaseException as exc:
                restore_error = exc
        if owned_session:
            try:
                http.close()
            except Exception:
                pass
        if injected_lock_acquired:
            _INJECTED_SESSION_ENV_LOCK.release()
        if restore_error is not None:
            raise restore_error
