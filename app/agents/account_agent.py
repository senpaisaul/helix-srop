"""
AccountAgent — answers questions about the user's Helix account.

Wired as an AgentTool by the root orchestrator. The user_id is read
from the session context preamble that the root injects into every
turn; the agent passes it through when calling tools.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent

from app.agents._model import get_model
from app.agents.tools.account_tools import get_account_status, get_recent_builds

ACCOUNT_INSTRUCTION = """
You are the Helix Account specialist. Answer questions about the
authenticated user's account: builds, plan, usage, limits.

Process:
1. Read the user_id from the [SESSION CONTEXT: ...] preamble in the user
   message. NEVER ask the user for their user_id.
2. Pick the right tool:
   - "show me builds", "recent builds", "last failed build" → get_recent_builds
   - "plan tier", "usage", "storage", "limits", "account status" → get_account_status
3. Call the tool with the user_id from context.
4. Summarize the result clearly. For builds, list newest first with status,
   pipeline, branch, and a short relative time.

Do not invent build IDs, statuses, or plan tiers. If a tool returns
empty data, say so.
"""


account_agent = LlmAgent(
    name="account_agent",
    model=get_model(),
    description="Specialist for the user's account: builds, plan tier, usage, limits.",
    instruction=ACCOUNT_INSTRUCTION,
    tools=[get_recent_builds, get_account_status],
)
