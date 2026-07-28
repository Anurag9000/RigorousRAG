import asyncio
import json

from tools.request_limits import RequestBodyLimitMiddleware


def run(coroutine):
    return asyncio.run(coroutine)


def test_declared_oversized_body_is_rejected_before_app_runs():
    called = []
    sent = []

    async def app(_scope, _receive, _send):
        called.append(True)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
    run(
        middleware(
            {
                "type": "http",
                "headers": [(b"content-length", b"11")],
            },
            receive,
            send,
        )
    )

    assert called == []
    assert sent[0]["status"] == 413
    payload = json.loads(sent[1]["body"])
    assert "10-byte limit" in payload["detail"]


def test_chunked_body_is_rejected_as_soon_as_stream_crosses_limit():
    sent = []
    messages = iter(
        [
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"78901", "more_body": False},
        ]
    )

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    async def consuming_app(_scope, wrapped_receive, _send):
        while True:
            message = await wrapped_receive()
            if not message.get("more_body"):
                break

    middleware = RequestBodyLimitMiddleware(consuming_app, max_bytes=10)
    run(middleware({"type": "http", "headers": []}, receive, send))

    assert sent[0]["status"] == 413
    assert sent[1]["more_body"] is False


def test_body_within_limit_reaches_application():
    sent = []
    called = []

    async def receive():
        return {"type": "http.request", "body": b"12345", "more_body": False}

    async def send(message):
        sent.append(message)

    async def app(_scope, wrapped_receive, wrapped_send):
        message = await wrapped_receive()
        called.append(message["body"])
        await wrapped_send({"type": "http.response.start", "status": 204, "headers": []})
        await wrapped_send({"type": "http.response.body", "body": b"", "more_body": False})

    middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
    run(
        middleware(
            {"type": "http", "headers": [(b"content-length", b"5")]},
            receive,
            send,
        )
    )

    assert called == [b"12345"]
    assert sent[0]["status"] == 204
