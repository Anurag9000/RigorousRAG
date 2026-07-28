"""ASGI request-size enforcement before multipart or JSON body parsing."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict

ASGIReceive = Callable[..., Awaitable[Dict[str, Any]]]
ASGISend = Callable[..., Awaitable[None]]
ASGIApp = Callable[[Dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]


class RequestBodyTooLarge(Exception):
    """Raised internally when a streamed request crosses the configured ceiling."""


class RequestBodyLimitMiddleware:
    """Reject oversized HTTP bodies before framework parsers allocate the full input."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")

    @staticmethod
    def _content_length(scope: Dict[str, Any]) -> int | None:
        values = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"content-length"
        ]
        if not values:
            return None
        parsed_values = []
        for value in values:
            try:
                parsed = int(value.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                return None
            if parsed < 0:
                return None
            parsed_values.append(parsed)
        if len(set(parsed_values)) != 1:
            return None
        return parsed_values[0]

    async def _reject(self, send: ASGISend) -> None:
        body = json.dumps(
            {"detail": f"Request body exceeds the {self.max_bytes}-byte limit."},
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        declared = self._content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        response_started = False
        response_complete = False

        async def limited_receive() -> Dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: Dict[str, Any]) -> None:
            nonlocal response_started, response_complete
            if message.get("type") == "http.response.start":
                response_started = True
            elif (
                message.get("type") == "http.response.body"
                and not message.get("more_body", False)
            ):
                response_complete = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if not response_started:
                await self._reject(send)
            elif not response_complete:
                # ASGI status/headers cannot be replaced after response start. Finish the
                # response explicitly instead of leaving the connection hanging.
                await send({
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                })
