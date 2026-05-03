"""
Escalation tool — used by EscalationAgent (extension E2).

create_ticket writes a row to the tickets table and returns its id.
Each tool call opens its own AsyncSession because ADK tools are
plain async functions without access to the FastAPI request scope.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from app.db.models import Ticket
from app.db.session import AsyncSessionLocal


async def create_ticket(
    user_id: str,
    summary: str,
    priority: Literal["low", "normal", "high"] = "normal",
) -> dict[str, Any]:
    """
    Create a Helix support ticket on behalf of the user.

    Use this when the user explicitly asks to escalate, raise a ticket,
    or reach a human, or when the issue is severe (production outage,
    data loss, security concern).

    Args:
        user_id: the Helix user identifier from session context.
        summary: a short, clear description of the problem.
        priority: 'low' | 'normal' | 'high'.

    Returns:
        dict with ticket_id, status, priority, summary.
    """
    from app.srop import trace_context

    ticket_id = f"tkt_{uuid.uuid4().hex[:12]}"
    async with AsyncSessionLocal() as db:
        db.add(
            Ticket(
                ticket_id=ticket_id,
                user_id=user_id,
                summary=summary[:1000],
                priority=priority,
                status="open",
            )
        )
        await db.commit()
    result: dict[str, Any] = {
        "ticket_id": ticket_id,
        "status": "open",
        "priority": priority,
        "summary": summary,
    }
    trace_context.record_tool_call(
        "create_ticket",
        {"user_id": user_id, "summary": summary, "priority": priority},
        result,
    )
    return result
