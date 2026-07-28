"""ASGI request-size enforcement before multipart or JSON body parsing."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict

ASGIApp = Callable[[Dict[str, Any], Callable[..., Awaitable[Dict[str, Any]]], Callable[..., Awaitable[None]]], Awaitable[None]]


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
        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                parsed = int(value.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                return None
            return parsed if parsed >= 0 else None
        return None

    async def _reject(self, send: Callable[..., Awaitable[None]]) -> None:
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
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[..., Awaitable[Dict[str, Any]]],
        send: Callable[..., Awaitable[None]],
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

        async def limited_receive() -> Dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: Dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if not response_started:
                await self._reject(send)
