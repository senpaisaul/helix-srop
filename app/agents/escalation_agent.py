"""
EscalationAgent — creates support tickets (extension E2).

Wired as an AgentTool by the root orchestrator. Uses the create_ticket
tool to write to the tickets table.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent

from app.agents._model import get_model
from app.agents.tools.escalation_tools import create_ticket

ESCALATION_INSTRUCTION = """
You are the Helix Escalation specialist. You create support tickets
for issues that need a human.

Process:
1. Read the user_id from the [SESSION CONTEXT: ...] preamble.
2. Compose a one-sentence summary of the problem from the user's message
   and any prior context shown in the preamble.
3. Choose priority:
   - high: production outage, data loss, security/compliance, billing
   - normal: blocked workflow, persistent error, recurring failure
   - low: feature request, question that didn't resolve via docs
4. Call create_ticket(user_id, summary, priority).
5. Confirm to the user with the returned ticket_id.

Never create more than one ticket per turn. Never create a ticket if the
user is just asking a question — only when they request escalation OR
the issue is clearly severe.
"""


escalation_agent = LlmAgent(
    name="escalation_agent",
    model=get_model(),
    description="Creates Helix support tickets when the user needs human escalation.",
    instruction=ESCALATION_INSTRUCTION,
    tools=[create_ticket],
)
