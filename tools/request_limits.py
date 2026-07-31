"""ASGI request-size enforcement and narrow public tool error translation."""

from __future__ import annotations

import itertools
import json
import operator
from typing import Any, Awaitable, Callable, Dict

ASGIReceive = Callable[..., Awaitable[Dict[str, Any]]]
ASGISend = Callable[..., Awaitable[None]]
ASGIApp = Callable[[Dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]

_MISSING_VISUAL_DOCUMENT_ERROR = "The requested document was not found for this owner."
_MAX_REQUEST_BODY_BYTES = 1_000_000_000
_MAX_HEADER_FIELDS = 1000
_MAX_HEADER_NAME_BYTES = 256
_MAX_HEADER_VALUE_BYTES = 8192
_MAX_CONTENT_LENGTH_DIGITS = 20


class RequestBodyTooLarge(Exception):
    """Raised internally when a streamed request crosses the configured ceiling."""


class InvalidRequestBody(Exception):
    """Raised internally for malformed ASGI request-body messages or framing."""


def _positive_integer(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("max_bytes must be an integer.")
    try:
        parsed = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_bytes must be an integer.") from exc
    result = int(parsed)
    if not 1 <= result <= _MAX_REQUEST_BODY_BYTES:
        raise ValueError(
            f"max_bytes must be between 1 and {_MAX_REQUEST_BODY_BYTES}."
        )
    return result


class RequestBodyLimitMiddleware:
    """Enforce body ceilings and one owner-safe visual lookup failure boundary."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        if not callable(app):
            raise TypeError("app must be callable.")
        self.app = app
        self.max_bytes = _positive_integer(max_bytes)

    @staticmethod
    def _content_length(scope: Dict[str, Any]) -> int | None:
        """Return one unambiguous length, or raise for malformed HTTP framing."""

        if not isinstance(scope, dict):
            raise InvalidRequestBody
        try:
            headers = scope.get("headers", [])
        except Exception as exc:
            raise InvalidRequestBody from exc
        if isinstance(headers, (str, bytes, bytearray)):
            raise InvalidRequestBody
        try:
            fields = list(itertools.islice(iter(headers), _MAX_HEADER_FIELDS + 1))
        except Exception as exc:
            raise InvalidRequestBody from exc
        if len(fields) > _MAX_HEADER_FIELDS:
            raise InvalidRequestBody

        values: list[int] = []
        for field in fields:
            if not isinstance(field, (tuple, list)) or len(field) != 2:
                raise InvalidRequestBody
            name, value = field
            if not isinstance(name, bytes) or not isinstance(value, bytes):
                raise InvalidRequestBody
            if (
                not name
                or len(name) > _MAX_HEADER_NAME_BYTES
                or len(value) > _MAX_HEADER_VALUE_BYTES
            ):
                raise InvalidRequestBody
            if name.lower() != b"content-length":
                continue
            if not value or len(value) > _MAX_CONTENT_LENGTH_DIGITS:
                raise InvalidRequestBody
            try:
                decoded = value.decode("ascii")
            except UnicodeDecodeError as exc:
                raise InvalidRequestBody from exc
            if not decoded.isdigit():
                raise InvalidRequestBody
            try:
                parsed = int(decoded, 10)
            except (ValueError, OverflowError) as exc:
                raise InvalidRequestBody from exc
            if parsed < 0:
                raise InvalidRequestBody
            values.append(parsed)

        if not values:
            return None
        if len(set(values)) != 1:
            raise InvalidRequestBody
        return values[0]

    @staticmethod
    async def _json_error(
        send: ASGISend,
        *,
        status: int,
        detail: str,
        close: bool = False,
    ) -> None:
        body = json.dumps(
            {"detail": detail},
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
        ]
        if close:
            headers.append((b"connection", b"close"))
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )

    async def _reject(self, send: ASGISend) -> None:
        await self._json_error(
            send,
            status=413,
            detail=f"Request body exceeds the {self.max_bytes}-byte limit.",
            close=True,
        )

    @staticmethod
    async def _finish_started_response(send: ASGISend) -> None:
        await send(
            {
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            }
        )

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if not isinstance(scope, dict):
            raise TypeError("ASGI scope must be a dictionary.")
        if not callable(receive) or not callable(send):
            raise TypeError("ASGI receive and send must be callable.")
        try:
            scope_type = scope.get("type")
        except Exception as exc:
            raise TypeError("ASGI scope could not be inspected.") from exc
        if scope_type != "http":
            await self.app(scope, receive, send)
            return
        try:
            declared = self._content_length(scope)
        except InvalidRequestBody:
            await self._json_error(
                send,
                status=400,
                detail="Malformed request headers.",
                close=True,
            )
            return
        if declared is not None and declared > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        response_started = False
        response_complete = False

        async def limited_receive() -> Dict[str, Any]:
            nonlocal received
            try:
                message = await receive()
            except RequestBodyTooLarge:
                raise
            if not isinstance(message, dict):
                raise InvalidRequestBody
            try:
                message_type = message.get("type")
            except Exception as exc:
                raise InvalidRequestBody from exc
            if message_type == "http.request":
                try:
                    body = message.get("body", b"")
                    more_body = message.get("more_body", False)
                except Exception as exc:
                    raise InvalidRequestBody from exc
                if not isinstance(body, bytes) or not isinstance(more_body, bool):
                    raise InvalidRequestBody
                received += len(body)
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: Dict[str, Any]) -> None:
            nonlocal response_started, response_complete
            if not isinstance(message, dict):
                raise TypeError("ASGI response messages must be dictionaries.")
            try:
                message_type = message.get("type")
            except Exception as exc:
                raise TypeError("ASGI response message could not be inspected.") from exc
            if message_type == "http.response.start":
                response_started = True
            elif message_type == "http.response.body":
                try:
                    more_body = message.get("more_body", False)
                except Exception as exc:
                    raise TypeError(
                        "ASGI response body message could not be inspected."
                    ) from exc
                if not isinstance(more_body, bool):
                    raise TypeError("ASGI response more_body must be a boolean.")
                if not more_body:
                    response_complete = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if not response_started:
                await self._reject(send)
            elif not response_complete:
                await self._finish_started_response(send)
        except InvalidRequestBody:
            if not response_started:
                await self._json_error(
                    send,
                    status=400,
                    detail="Malformed request body.",
                    close=True,
                )
            elif not response_complete:
                await self._finish_started_response(send)
        except ValueError as exc:
            # Translate only the exact owner-scoped metadata absence sentinel. Other
            # ValueErrors—including invalid JSON or programming defects—remain visible
            # to the normal server error boundary.
            arguments = exc.args
            path = scope.get("path")
            exact_missing = (
                len(arguments) == 1
                and isinstance(arguments[0], str)
                and arguments[0] == _MISSING_VISUAL_DOCUMENT_ERROR
            )
            if (
                not response_started
                and isinstance(path, str)
                and path == "/tool/visual-entailment"
                and exact_missing
            ):
                await self._json_error(
                    send,
                    status=404,
                    detail="Document not found.",
                )
                return
            raise
