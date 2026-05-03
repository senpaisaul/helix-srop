"""
Account tools — used by AccountAgent.

Mock data is deterministic per user_id so demos are reproducible across
process restarts. The integration evaluation cares about wiring (the
agent must call these tools when the user asks about builds/account),
not about real backend integration.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

PIPELINES = ["api-build", "frontend-build", "worker-build", "docs-build", "release"]
BRANCHES = ["main", "develop", "feature/auth-rotation", "hotfix/cache", "release/v1.2"]
STATUSES = ["passed", "passed", "passed", "failed", "cancelled"]  # weighted toward pass


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str
    branch: str
    started_at: str  # ISO8601 UTC
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


def _user_seed(user_id: str) -> int:
    return int(hashlib.sha256(user_id.encode()).hexdigest()[:8], 16)


def _mock_builds(user_id: str, limit: int) -> list[BuildSummary]:
    seed = _user_seed(user_id)
    now = datetime.now(UTC)
    builds: list[BuildSummary] = []
    for i in range(limit):
        idx = (seed + i) % 1000
        builds.append(
            BuildSummary(
                build_id=f"bld_{user_id}_{idx:04d}",
                pipeline=PIPELINES[(seed + i) % len(PIPELINES)],
                status=STATUSES[(seed + i * 7) % len(STATUSES)],
                branch=BRANCHES[(seed + i * 3) % len(BRANCHES)],
                started_at=(now - timedelta(hours=i * 4 + 1)).isoformat(),
                duration_seconds=120 + ((seed + i * 11) % 600),
            )
        )
    return builds


# ---------------------------------------------------------------------------
# Public ADK tool functions. These are what the AccountAgent calls.
# ---------------------------------------------------------------------------


async def get_recent_builds(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Return the user's most recent CI builds, newest first.

    Args:
        user_id: the Helix user identifier from session context.
        limit: how many builds to return (1-20).

    Returns:
        List of dicts with keys: build_id, pipeline, status, branch,
        started_at (ISO8601), duration_seconds.
    """
    from app.srop import trace_context

    limit = max(1, min(20, limit))
    builds = _mock_builds(user_id, limit)
    out = [asdict(b) for b in builds]
    trace_context.record_tool_call(
        "get_recent_builds",
        {"user_id": user_id, "limit": limit},
        {"count": len(out), "build_ids": [b["build_id"] for b in out]},
    )
    return out


async def get_account_status(user_id: str) -> dict[str, Any]:
    """
    Return the user's current account status: plan tier and usage limits.

    Args:
        user_id: the Helix user identifier from session context.

    Returns:
        dict with plan_tier, concurrent_builds_used / _limit,
        storage_used_gb / _limit.
    """
    from sqlalchemy import select

    from app.db.models import User
    from app.db.session import AsyncSessionLocal
    from app.srop import trace_context

    # Read the real plan_tier from the users table so the agent's answer
    # is consistent with whatever was set at session-create time.
    plan = "free"
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.user_id == user_id))
        user = result.scalar_one_or_none()
        if user is not None:
            plan = user.plan_tier
    if plan not in {"free", "pro", "enterprise"}:
        plan = "free"

    seed = _user_seed(user_id)
    limits = {"free": (2, 5.0), "pro": (10, 50.0), "enterprise": (50, 500.0)}[plan]
    status = asdict(
        AccountStatus(
            user_id=user_id,
            plan_tier=plan,
            concurrent_builds_used=(seed % limits[0]) // 2,
            concurrent_builds_limit=limits[0],
            storage_used_gb=round((seed % 1000) / 100, 2),
            storage_limit_gb=limits[1],
        )
    )
    trace_context.record_tool_call(
        "get_account_status", {"user_id": user_id}, status
    )
    return status
