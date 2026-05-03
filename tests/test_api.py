"""
Integration tests — exercise the full SROP HTTP surface.
LLM mocked at the ADK boundary via the mock_adk fixture (see conftest).
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_create_session(client) -> None:
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001"})
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert body["user_id"] == "u_test_001"


@pytest.mark.asyncio
async def test_healthz(client) -> None:
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client, mock_adk) -> None:
    resp = await client.post("/v1/chat/nonexistent-id", json={"content": "hello"})
    assert resp.status_code == 404
    assert resp.json()["title"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk) -> None:
    """Core integration test: routing + state persistence across two turns."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_t2", "plan_tier": "pro"})
    assert sess.status_code == 200
    session_id = sess.json()["session_id"]

    # Turn 1 — knowledge question.
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["routed_to"] == "knowledge"
    assert "reply" in body1
    trace_id = body1["trace_id"]

    # Trace must include retrieved chunk IDs.
    trace = await client.get(f"/v1/traces/{trace_id}")
    assert trace.status_code == 200
    assert len(trace.json()["retrieved_chunk_ids"]) > 0

    # Turn 2 — same session; agent must answer plan_tier from state.
    r2 = await client.post(
        f"/v1/chat/{session_id}", json={"content": "What is my plan tier?"}
    )
    assert r2.status_code == 200
    assert "pro" in r2.json()["reply"].lower()


@pytest.mark.asyncio
async def test_idempotency_replays_response(client, mock_adk) -> None:
    """Same Idempotency-Key returns the cached response without re-running pipeline."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_idem", "plan_tier": "free"})
    session_id = sess.json()["session_id"]

    headers = {"Idempotency-Key": "key-abc-123"}
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
        headers=headers,
    )
    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
        headers=headers,
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Same trace_id ⇒ second call did not re-invoke the pipeline.
    assert r1.json()["trace_id"] == r2.json()["trace_id"]


@pytest.mark.asyncio
async def test_trace_not_found_returns_404(client) -> None:
    resp = await client.get("/v1/traces/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["title"] == "TRACE_NOT_FOUND"


@pytest.mark.asyncio
async def test_create_session_validates_plan_tier(client) -> None:
    resp = await client.post(
        "/v1/sessions", json={"user_id": "u_x", "plan_tier": "platinum"}
    )
    assert resp.status_code == 422
