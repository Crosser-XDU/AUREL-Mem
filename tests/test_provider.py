"""Exercise the real OpenAI SDK serialization through an in-memory transport.

No network requests, real credentials or model calls are used. These tests cover
the upstream protocol and failure boundary rather than substituting Provider
methods, so an SDK/template integration regression is observable.
"""

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json

import httpx
from openai import AsyncOpenAI
import pytest

from fusion_memory.config import Settings
from fusion_memory.provider import Provider, UpstreamError


CHAT_KEY = "unit-test-chat-secret-not-real"
EMBEDDING_KEY = "unit-test-embedding-secret-not-real"
CHAT_BASE = "https://chat.example.invalid/v1"
EMBEDDING_BASE = "https://embedding.example.invalid/v1"
PNG_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6"
    "7jEAAAAASUVORK5CYII="
)


def completion(payload, *, finish_reason="stop", raw=False):
    return httpx.Response(200, json={
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1704067200,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": payload if raw else json.dumps(payload, ensure_ascii=False),
            },
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 17, "completion_tokens": 9, "total_tokens": 26},
    })


def source_groups(body):
    """Decode SOURCE framing from the actual serialized SDK request."""
    groups = []
    current = None
    for part in body["messages"][1]["content"]:
        if part["type"] == "text" and part["text"].startswith("SOURCE "):
            current = {"label": json.loads(part["text"][7:]), "parts": []}
            groups.append(current)
        elif current is not None:
            current["parts"].append(part)
    return groups


@asynccontextmanager
async def mocked_provider(handler, **overrides):
    settings = Settings(
        api_key=CHAT_KEY, api_base=CHAT_BASE,
        embedding_key=EMBEDDING_KEY, embedding_base=EMBEDDING_BASE,
        **overrides,
    )
    provider = Provider(settings)
    # Close constructor-owned clients before replacing their transport.
    await provider.close()
    provider.client = AsyncOpenAI(
        api_key=settings.api_key, base_url=settings.api_base, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider.embedding_client = AsyncOpenAI(
        api_key=settings.embedding_key, base_url=settings.embedding_base, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        yield provider
    finally:
        await provider.close()


def test_extract_uses_official_multimodal_chat_template_and_maps_source_indices():
    requests = []
    parts = [
        {"type": "text", "text": "这是我的杯子。"},
        {"type": "image_url", "image_url": {"url": PNG_URI, "detail": "high"}},
        {"type": "text", "text": "请记住标签上的编号。"},
    ]
    messages = [
        {"role": "user", "timestamp": 1704067200000, "content": parts},
        {"role": "assistant", "content": "我会记住。"},
    ]

    def handler(request):
        body = json.loads(request.content)
        requests.append((request, body))
        # Return valid records out of order: response order is not source order.
        return completion({"records": [
            {"index": 1, "summary": "助手确认", "facts": ["acknowledgment"]},
            {"index": 0, "summary": "用户的杯子", "ocr": ["AB-42"], "entities": ["杯子"]},
        ]})

    async def exercise():
        async with mocked_provider(handler) as provider:
            frames = await provider.extract(messages, history="user: earlier context")
            assert len(frames) == 2
            assert frames[0]["summary"] == "用户的杯子"
            assert frames[0]["ocr"] == ["AB-42"]
            assert frames[1]["summary"] == "助手确认"
            assert frames[1]["facts"] == ["acknowledgment"]
            assert provider.usage == {"prompt_tokens": 17, "completion_tokens": 9}

    asyncio.run(exercise())
    assert len(requests) == 1
    request, body = requests[0]
    assert request.method == "POST"
    assert str(request.url) == CHAT_BASE + "/chat/completions"
    assert request.headers["authorization"] == "Bearer " + CHAT_KEY
    assert body["model"] == "gpt-4o-mini"
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0
    assert body["messages"][0]["role"] == "system"
    assert "untrusted source data" in body["messages"][0]["content"]
    assert "Do not generate an answer" in body["messages"][0]["content"]
    assert body["messages"][1]["content"][0]["text"].startswith("PRIOR CONTEXT (not new memory):")
    groups = source_groups(body)
    assert groups[0]["label"] == {"index": 0, "role": "user", "timestamp_unix_ms": 1704067200000}
    assert groups[0]["parts"] == parts  # exact image URI, detail and interleaving survive
    assert groups[1]["label"]["role"] == "assistant"
    assert groups[1]["parts"] == [{"type": "text", "text": "我会记住。"}]
    assert messages[0]["content"] == parts


def test_long_source_and_multiple_images_are_all_sent_and_merged_without_source_mixup():
    long_text = "A" * 9999 + "甲" + "B" * 9999 + "乙" + "C" * 8501
    image_parts = [
        {"type": "image_url", "image_url": {"url": PNG_URI, "detail": "low"}}
        for _ in range(5)
    ]
    original_parts = [{"type": "text", "text": long_text}, *image_parts,
                      {"type": "text", "text": "END-OF-LONG-SOURCE"}]
    seen = []

    def handler(request):
        body = json.loads(request.content)
        groups = source_groups(body)
        seen.extend(groups)
        rows = []
        for group in groups:
            content = "".join(p["text"] for p in group["parts"] if p["type"] == "text")
            digest = hashlib.sha256(content.encode()).hexdigest()
            rows.append({"index": group["label"]["index"], "summary": digest, "facts": [digest]})
        return completion({"records": list(reversed(rows))})

    async def exercise():
        async with mocked_provider(handler, llm_concurrency=1) as provider:
            frames = await provider.extract([
                {"role": "user", "timestamp": 1000, "content": original_parts},
                {"role": "assistant", "timestamp": 2000, "content": "SEPARATE-SOURCE"},
            ])
            long_groups = [g for g in seen if g["label"]["timestamp_unix_ms"] == 1000]
            assert len(long_groups) > 1
            received_parts = [p for g in long_groups for p in g["parts"]]
            assert "".join(p["text"] for p in received_parts if p["type"] == "text") == long_text + "END-OF-LONG-SOURCE"
            assert [p for p in received_parts if p["type"] == "image_url"] == image_parts
            expected_digests = [hashlib.sha256("".join(
                p["text"] for p in g["parts"] if p["type"] == "text"
            ).encode()).hexdigest() for g in long_groups]
            assert frames[0]["summary"].splitlines() == expected_digests
            assert frames[0]["facts"] == list(dict.fromkeys(expected_digests))
            other_digest = hashlib.sha256(b"SEPARATE-SOURCE").hexdigest()
            assert frames[1]["summary"] == other_digest
            assert frames[1]["facts"] == [other_digest]
            assert other_digest not in frames[0]["facts"]
            assert provider.calls["extract"] > 1

    asyncio.run(exercise())


@pytest.mark.parametrize("payload", [
    {},
    {"records": []},
    {"records": [{"index": 5, "summary": "wrong source"}]},
    {"records": [{"index": 0, "facts": "wrong field type"}]},
    {"records": [{"index": 0, "summary": "a"}, {"index": 0, "summary": "b"}]},
    {"records": [{"index": False, "summary": "boolean is not a source index"}]},
    {"records": [{"index": 0.0, "summary": "decimal is not a source index"}]},
    {"records": [{"index": "0", "summary": "string is not a source index"}]},
    {"records": [{"index": -1, "summary": "negative index"}]},
    {"records": [{"index": [], "summary": "unhashable index"}]},
    {"records": ["not a record object"]},
])
def test_invalid_extraction_schema_never_returns_success(payload):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion(payload)

    async def exercise():
        async with mocked_provider(handler) as provider:
            count = 2 if len(payload.get("records", [])) == 2 else 1
            with pytest.raises(UpstreamError, match="^extraction returned invalid schema; Add has not been committed$"):
                await provider.extract([{"role": "user", "content": "test"} for _ in range(count)])
            assert provider.calls["extract"] == 2
            assert provider.calls["failed"] == 2
    asyncio.run(exercise())
    assert len(requests) == 2
    assert "FORMAT CORRECTION" not in requests[0]["messages"][0]["content"]
    assert "FORMAT CORRECTION" in requests[1]["messages"][0]["content"]


def test_invalid_schema_retries_once_with_original_evidence_and_safe_format_hint(caplog):
    requests = []
    invalid_output_marker = "MALFORMED-MODEL-PRIVATE-OUTPUT-DO-NOT-ECHO"
    parts = [
        {"type": "text", "text": "标签上的编号是 AB-42。"},
        {"type": "image_url", "image_url": {"url": PNG_URI, "detail": "high"}},
    ]

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return completion({"records": [{"index": 0, "facts": {"invalid": invalid_output_marker}}]})
        return completion({"records": [{"index": 0, "summary": "标签编号", "facts": ["编号是 AB-42"], "ocr": ["AB-42"]}]})

    async def exercise():
        async with mocked_provider(handler) as provider:
            result = await provider.extract(
                [{"role": "user", "timestamp": 1704067200000, "content": parts}],
                history="Earlier context remains source context only.",
            )
            assert result[0]["facts"] == ["编号是 AB-42"]
            assert result[0]["ocr"] == ["AB-42"]
            assert provider.calls["extract"] == 2
            assert provider.calls["failed"] == 1
            assert provider.usage == {"prompt_tokens": 34, "completion_tokens": 18}

    asyncio.run(exercise())
    assert len(requests) == 2
    assert requests[1]["messages"][1] == requests[0]["messages"][1]
    assert source_groups(requests[1])[0]["parts"] == parts
    assert len(requests[1]["messages"]) == 2  # no malformed assistant response replay
    hint = requests[1]["messages"][0]["content"]
    assert "FORMAT CORRECTION" in hint
    assert "arrays of strings" in hint
    assert "do not change source facts" in hint
    assert invalid_output_marker not in json.dumps(requests)
    assert invalid_output_marker not in caplog.text
    assert "attempt=1 max_attempts=2" in caplog.text


@pytest.mark.parametrize("response", [
    lambda: completion({"records": [{"index": 0, "summary": "partial"}]}, finish_reason="length"),
    lambda: completion("not JSON", raw=True),
    lambda: completion(["not an object"]),
])
def test_truncated_or_invalid_json_completion_is_an_upstream_failure(response):
    async def exercise():
        async with mocked_provider(lambda request: response()) as provider:
            with pytest.raises(UpstreamError):
                await provider.extract([{"role": "user", "content": "source"}])
            assert provider.calls["extract"] == 1
            assert provider.calls["failed"] == 1
    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["connection", "timeout", "rate_limit", "server_error"])
def test_upstream_failures_do_not_gain_a_schema_retry(failure):
    requests = []

    def handler(request):
        requests.append(request)
        if failure == "connection":
            raise httpx.ConnectError("synthetic connection failure", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        return httpx.Response(429 if failure == "rate_limit" else 503, json={"error": {"message": "unavailable"}})

    async def exercise():
        # mocked_provider disables SDK retries, exposing any new application
        # retry. Production's existing SDK max_retries setting is unchanged.
        async with mocked_provider(handler) as provider:
            with pytest.raises(UpstreamError, match="extract model call failed"):
                await provider.extract([{"role": "user", "content": "source"}])
            assert provider.calls["extract"] == 1
            assert provider.calls["failed"] == 1

    asyncio.run(exercise())
    assert len(requests) == 1


def test_network_failure_during_format_retry_is_not_retried_again():
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return completion({"records": [{"index": 0, "facts": "invalid field type"}]})
        raise httpx.ConnectError("synthetic connection failure", request=request)

    async def exercise():
        async with mocked_provider(handler) as provider:
            with pytest.raises(UpstreamError, match="extract model call failed"):
                await provider.extract([{"role": "user", "content": "source"}])
            assert provider.calls["extract"] == 2
            assert provider.calls["failed"] == 2

    asyncio.run(exercise())
    assert len(requests) == 2


def test_upstream_403_does_not_expose_credentials_or_response_body(caplog):
    private_text = "PRIVATE-UPSTREAM-BODY: " + CHAT_KEY

    def handler(request):
        return httpx.Response(403, json={"error": {"message": private_text, "type": "permission_error"}})

    async def exercise():
        async with mocked_provider(handler) as provider:
            with pytest.raises(UpstreamError) as caught:
                await provider.extract([{"role": "user", "content": "private user evidence"}])
            assert CHAT_KEY not in str(caught.value)
            assert private_text not in str(caught.value)
            assert "private user evidence" not in str(caught.value)
            assert provider.calls["failed"] == 1

    asyncio.run(exercise())
    assert "error_type=PermissionDeniedError" in caplog.text
    assert CHAT_KEY not in caplog.text
    assert private_text not in caplog.text
    assert "private user evidence" not in caplog.text


def test_image_query_indexes_features_preserves_question_and_does_not_consume_answer_field():
    requests = []
    parts = [
        {"type": "text", "text": "Which earlier photo has this mug?"},
        {"type": "image_url", "image_url": {"url": PNG_URI}},
    ]

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return completion({
            "description": "Red mug with white stars",
            "terms": ["mug", "stars", "红色"],
            "answer": "DO-NOT-RETURN-THIS-FINAL-ANSWER",
        })

    async def exercise():
        async with mocked_provider(handler) as provider:
            query = await provider.query_text(parts, options=["pic-A", "pic-B"])
            assert query.startswith(parts[0]["text"])
            assert "Red mug with white stars" in query
            assert "mug stars 红色" in query
            assert "pic-A\npic-B" in query
            assert "DO-NOT-RETURN-THIS-FINAL-ANSWER" not in query
            assert provider.calls["query"] == 1
            assert provider.calls["extract"] == 0

    asyncio.run(exercise())
    assert requests[0]["messages"][1]["content"] == parts
    assert "Do not answer the question or select any option" in requests[0]["messages"][0]["content"]
    assert requests[0]["model"] == "gpt-4o-mini"


def test_text_only_query_and_disabled_embeddings_need_no_upstream_call():
    def unexpected_request(request):
        pytest.fail("text-only query / disabled embeddings should not call the network")

    async def exercise():
        async with mocked_provider(unexpected_request) as provider:
            assert await provider.query_text("What is my dog's name?", ["Bobo", "Kiki"]) == "What is my dog's name?\nBobo\nKiki"
            assert await provider.embed(["a", "b"]) == [None, None]
            assert provider.calls == {"extract": 0, "query": 0, "embedding": 0, "failed": 0}
    asyncio.run(exercise())


@pytest.mark.parametrize("dimensions", [2, 1024])
def test_embedding_v4_uses_separate_configuration_and_restores_response_order_across_batches(dimensions):
    requests = []
    inputs = [f"document-{index}" for index in range(13)]

    def handler(request):
        body = json.loads(request.content)
        requests.append((request, body))
        rows = [{"object": "embedding", "index": index,
                 "embedding": [float(text.rsplit("-", 1)[1])] + [1.0] * (dimensions - 1)}
                for index, text in enumerate(body["input"])]
        return httpx.Response(200, json={
            "object": "list", "data": list(reversed(rows)), "model": "text-embedding-v4",
            "usage": {"prompt_tokens": 13, "total_tokens": 13},
        })

    async def exercise():
        async with mocked_provider(handler, embedding_enabled=True,
                                   embedding_dimensions=dimensions) as provider:
            assert await provider.embed(inputs) == [
                [float(index)] + [1.0] * (dimensions - 1) for index in range(13)
            ]
            assert provider.calls["embedding"] == 2
    asyncio.run(exercise())
    assert [text for _, body in requests for text in body["input"]] == inputs
    for request, body in requests:
        assert str(request.url) == EMBEDDING_BASE + "/embeddings"
        assert request.headers["authorization"] == "Bearer " + EMBEDDING_KEY
        assert body["model"] == "text-embedding-v4"
        assert body["encoding_format"] == "float"
        assert body["dimensions"] == dimensions


def test_embedding_count_mismatch_fails_instead_of_silently_misaligning_records():
    def handler(request):
        return httpx.Response(200, json={
            "object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "text-embedding-v4", "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    async def exercise():
        async with mocked_provider(handler, embedding_enabled=True,
                                   embedding_dimensions=2) as provider:
            with pytest.raises(UpstreamError, match="no silent model substitution"):
                await provider.embed(["first source", "second source"])
            assert provider.calls["failed"] == 1
    asyncio.run(exercise())


@pytest.mark.parametrize("returned_dimensions", [1023, 1025])
def test_embedding_dimension_mismatch_rejects_entire_batch(returned_dimensions):
    def handler(request):
        body = json.loads(request.content)
        assert body["dimensions"] == 1024
        # Count and indices are correct; only the second vector is malformed.
        return httpx.Response(200, json={
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1] * 1024},
                {"object": "embedding", "index": 1, "embedding": [0.2] * returned_dimensions},
            ],
            "model": "text-embedding-v4",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        })

    async def exercise():
        async with mocked_provider(handler, embedding_enabled=True,
                                   embedding_dimensions=1024) as provider:
            with pytest.raises(UpstreamError):
                await provider.embed(["first source", "second source"])
            assert provider.calls["embedding"] == 1
            assert provider.calls["failed"] == 1
    asyncio.run(exercise())
