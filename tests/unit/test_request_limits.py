import asyncio
import json

import pytest

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
    assert (b"connection", b"close") in sent[0]["headers"]
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


def test_response_started_before_limit_violation_is_completed_not_left_hanging():
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

    async def early_response_app(_scope, wrapped_receive, wrapped_send):
        await wrapped_send({"type": "http.response.start", "status": 202, "headers": []})
        await wrapped_send({
            "type": "http.response.body",
            "body": b"partial",
            "more_body": True,
        })
        await wrapped_receive()
        await wrapped_receive()

    middleware = RequestBodyLimitMiddleware(early_response_app, max_bytes=10)
    run(middleware({"type": "http", "headers": []}, receive, send))

    assert sent[0]["status"] == 202
    assert sent[1]["body"] == b"partial"
    assert sent[-1] == {
        "type": "http.response.body",
        "body": b"",
        "more_body": False,
    }


def test_conflicting_content_lengths_fall_back_to_stream_counting():
    sent = []
    called = []

    async def receive():
        return {"type": "http.request", "body": b"12345678901", "more_body": False}

    async def send(message):
        sent.append(message)

    async def app(_scope, wrapped_receive, _wrapped_send):
        called.append(True)
        await wrapped_receive()

    middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
    run(
        middleware(
            {
                "type": "http",
                "headers": [
                    (b"content-length", b"5"),
                    (b"content-length", b"11"),
                ],
            },
            receive,
            send,
        )
    )

    assert called == [True]
    assert sent[0]["status"] == 413


def test_visual_missing_document_value_error_becomes_generic_404():
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    async def app(_scope, _receive, _send):
        raise ValueError("The requested document was not found for this owner.")

    middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
    run(
        middleware(
            {
                "type": "http",
                "path": "/tool/visual-entailment",
                "headers": [],
            },
            receive,
            send,
        )
    )

    assert sent[0]["status"] == 404
    payload = json.loads(sent[1]["body"])
    assert payload == {"detail": "Document not found."}


def test_unrelated_visual_value_error_is_not_hidden():
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    async def app(_scope, _receive, _send):
        raise ValueError("invalid scientific JSON")

    middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
    with pytest.raises(ValueError, match="invalid scientific JSON"):
        run(
            middleware(
                {
                    "type": "http",
                    "path": "/tool/visual-entailment",
                    "headers": [],
                },
                receive,
                send,
            )
        )


def test_non_visual_value_error_is_not_hidden():
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    async def app(_scope, _receive, _send):
        raise ValueError("programming bug")

    middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
    with pytest.raises(ValueError, match="programming bug"):
        run(
            middleware(
                {"type": "http", "path": "/query", "headers": []},
                receive,
                send,
            )
        )


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
