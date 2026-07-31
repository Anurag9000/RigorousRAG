import asyncio
import json
from decimal import Decimal
from fractions import Fraction

import pytest

from tools.request_limits import RequestBodyLimitMiddleware


def run(coroutine):
    return asyncio.run(coroutine)


def test_constructor_strictly_validates_app_and_byte_limit():
    async def app(_scope, _receive, _send):
        return None

    with pytest.raises(TypeError, match="app"):
        RequestBodyLimitMiddleware(object(), max_bytes=10)
    for invalid in (
        True,
        1.5,
        Decimal("1.5"),
        Fraction(3, 2),
        "bad",
        0,
        -1,
        1_000_000_001,
    ):
        with pytest.raises(ValueError, match="max_bytes"):
            RequestBodyLimitMiddleware(app, max_bytes=invalid)


def test_constructor_accepts_exact_index_protocol_limit():
    class ExactInteger:
        def __index__(self):
            return 10

    async def app(_scope, _receive, _send):
        return None

    middleware = RequestBodyLimitMiddleware(app, max_bytes=ExactInteger())

    assert middleware.max_bytes == 10


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


@pytest.mark.parametrize(
    "headers",
    [
        [(b"content-length", b"5"), (b"content-length", b"11")],
        [(b"content-length", b" bad")],
        [(b"content-length", b"+11")],
        [(b"content-length", b"-1")],
        [(b"content-length", b"1" * 21)],
        [("content-length", "11")],
        [object()],
        b"not-a-header-list",
    ],
)
def test_conflicting_or_malformed_content_lengths_fail_before_app(headers):
    sent = []
    called = []

    async def receive():
        return {"type": "http.request", "body": b"123", "more_body": False}

    async def send(message):
        sent.append(message)

    async def app(_scope, _wrapped_receive, _wrapped_send):
        called.append(True)

    middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
    run(
        middleware(
            {"type": "http", "headers": headers},
            receive,
            send,
        )
    )

    assert called == []
    assert sent[0]["status"] == 400
    assert (b"connection", b"close") in sent[0]["headers"]
    assert json.loads(sent[1]["body"]) == {"detail": "Malformed request headers."}


def test_too_many_or_oversized_header_fields_fail_before_app():
    headers_sets = [
        [(b"x", b"y")] * 1001,
        [(b"x" * 257, b"y")],
        [(b"x", b"y" * 8193)],
    ]
    for headers in headers_sets:
        sent = []
        called = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        async def app(_scope, _receive, _send):
            called.append(True)

        middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
        run(middleware({"type": "http", "headers": headers}, receive, send))

        assert called == []
        assert sent[0]["status"] == 400


def test_duplicate_identical_content_lengths_are_accepted():
    sent = []
    called = []

    async def receive():
        return {"type": "http.request", "body": b"12345", "more_body": False}

    async def send(message):
        sent.append(message)

    async def app(_scope, wrapped_receive, wrapped_send):
        called.append((await wrapped_receive())["body"])
        await wrapped_send({"type": "http.response.start", "status": 204, "headers": []})
        await wrapped_send({"type": "http.response.body", "body": b"", "more_body": False})

    middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
    run(
        middleware(
            {
                "type": "http",
                "headers": [
                    (b"content-length", b"5"),
                    (b"Content-Length", b"5"),
                ],
            },
            receive,
            send,
        )
    )

    assert called == [b"12345"]
    assert sent[0]["status"] == 204


def test_malformed_request_message_becomes_generic_400_before_response_start():
    for message in (
        "not-a-dictionary",
        {"type": "http.request", "body": "not-bytes", "more_body": False},
        {"type": "http.request", "body": bytearray(b"bytes"), "more_body": False},
        {"type": "http.request", "body": b"bytes", "more_body": "yes"},
    ):
        sent = []

        async def receive(message=message):
            return message

        async def send(outgoing):
            sent.append(outgoing)

        async def app(_scope, wrapped_receive, _wrapped_send):
            await wrapped_receive()

        middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
        run(middleware({"type": "http", "headers": []}, receive, send))

        assert sent[0]["status"] == 400
        assert (b"connection", b"close") in sent[0]["headers"]
        assert json.loads(sent[1]["body"]) == {"detail": "Malformed request body."}


def test_malformed_request_after_response_start_is_completed():
    sent = []

    async def receive():
        return {"type": "http.request", "body": "not-bytes", "more_body": False}

    async def send(message):
        sent.append(message)

    async def app(_scope, wrapped_receive, wrapped_send):
        await wrapped_send({"type": "http.response.start", "status": 202, "headers": []})
        await wrapped_send({
            "type": "http.response.body",
            "body": b"partial",
            "more_body": True,
        })
        await wrapped_receive()

    middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
    run(middleware({"type": "http", "headers": []}, receive, send))

    assert sent[-1] == {
        "type": "http.response.body",
        "body": b"",
        "more_body": False,
    }


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


def test_unrelated_or_hostile_visual_value_error_is_not_hidden_or_stringified():
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    async def ordinary_app(_scope, _receive, _send):
        raise ValueError("invalid scientific JSON")

    middleware = RequestBodyLimitMiddleware(ordinary_app, max_bytes=10)
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

    class Hostile:
        def __str__(self):
            raise RuntimeError("must not stringify")

    marker = Hostile()

    async def hostile_app(_scope, _receive, _send):
        raise ValueError(marker)

    middleware = RequestBodyLimitMiddleware(hostile_app, max_bytes=10)
    with pytest.raises(ValueError) as captured:
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
    assert captured.value.args == (marker,)


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


def test_non_http_scope_is_passed_through_unchanged():
    called = []

    async def app(scope, receive, send):
        called.append((scope, receive, send))

    async def receive():
        return {"type": "websocket.receive"}

    async def send(_message):
        return None

    scope = {"type": "websocket"}
    middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
    run(middleware(scope, receive, send))

    assert called == [(scope, receive, send)]


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
