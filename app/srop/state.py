"""
Session state schema — persisted in sessions.state (JSON column).

This is the restart-survival mechanism: every turn loads this from DB,
the agent reads it via dynamic instruction injection, and we save it
back at the end of the turn. Killing uvicorn between turns cannot
lose context because nothing context-relevant lives in process memory.

Keep this small — only fields the agent CANNOT re-derive from message history.
"""
from typing import Literal

from pydantic import BaseModel, Field

AgentName = Literal["knowledge", "account", "escalation", "smalltalk"]


class SessionState(BaseModel):
    user_id: str
    plan_tier: Literal["free", "pro", "enterprise"] = "free"
    last_agent: AgentName | None = None
    turn_count: int = 0
    # IDs of tickets opened this session (E2). Lets the agent reference them later.
    open_ticket_ids: list[str] = Field(default_factory=list)
    # One-line summary of the most recent assistant turn — gives the LLM
    # quick context on what just happened without re-loading full history.
    last_summary: str = ""

    def to_db_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_db_dict(cls, data: dict) -> "SessionState":
        return cls.model_validate(data)
