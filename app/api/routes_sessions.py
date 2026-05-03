"""
POST /v1/sessions — create a session.
"""
from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session as SessionModel
from app.db.models import User
from app.db.session import get_db
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])

PlanTier = Literal["free", "pro", "enterprise"]


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    plan_tier: PlanTier = "free"


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """
    Create a new session. Upsert the user, then create the session row
    with an initial SessionState that reflects the requested plan_tier.
    """
    result = await db.execute(select(User).where(User.user_id == body.user_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(user_id=body.user_id, plan_tier=body.plan_tier)
        db.add(user)
    else:
        # Update plan_tier if the caller supplied a non-default one.
        user.plan_tier = body.plan_tier

    state = SessionState(user_id=body.user_id, plan_tier=body.plan_tier)
    session = SessionModel(
        session_id=str(uuid.uuid4()),
        user_id=body.user_id,
        state=state.to_db_dict(),
    )
    db.add(session)
    await db.commit()

    return CreateSessionResponse(session_id=session.session_id, user_id=body.user_id)
