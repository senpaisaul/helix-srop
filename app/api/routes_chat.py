"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.

Includes E1 idempotency: when the client supplies an Idempotency-Key
header, the (session_id, key) pair is used as a replay cache. A second
request with the same key returns the cached response without re-running
the pipeline.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IdempotencyKey
from app.db.session import get_db
from app.srop import pipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10_000)


class ChatResponse(BaseModel):
    reply: str
    routed_to: str
    trace_id: str


async def _replay_if_cached(
    session_id: str, key: str | None, db: AsyncSession
) -> ChatResponse | None:
    if not key:
        return None
    result = await db.execute(
        select(IdempotencyKey).where(
            IdempotencyKey.session_id == session_id,
            IdempotencyKey.key == key,
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return ChatResponse(**row.response_json)


async def _store_response(
    session_id: str, key: str, response: ChatResponse, db: AsyncSession
) -> None:
    db.add(
        IdempotencyKey(
            session_id=session_id,
            key=key,
            response_json=json.loads(response.model_dump_json()),
        )
    )
    try:
        await db.commit()
    except IntegrityError:
        # Concurrent insert with the same (session_id, key) — fine,
        # the other writer won the race. Discard our row.
        await db.rollback()


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ChatResponse:
    """
    Run one SROP turn.

    Errors:
      - SESSION_NOT_FOUND  → 404 (raised inside pipeline.run)
      - UPSTREAM_TIMEOUT   → 504 (raised inside pipeline.run)
    """
    cached = await _replay_if_cached(session_id, idempotency_key, db)
    if cached is not None:
        return cached

    result = await pipeline.run(session_id, body.content, db)
    response = ChatResponse(
        reply=result.content,
        routed_to=result.routed_to,
        trace_id=result.trace_id,
    )

    if idempotency_key:
        await _store_response(session_id, idempotency_key, response, db)

    return response
