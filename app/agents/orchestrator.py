"""
SROP Root Orchestrator — Google ADK agent.

Routes each user turn to the right specialist via ADK's AgentTool. The
LLM picks the tool — there is no string-parsing branch in this codebase.

Each AgentTool wraps a sub-agent. ADK presents it to the LLM as a
function whose name matches the sub-agent's `name` field, so the LLM
"calls" knowledge_agent / account_agent / escalation_agent the same way
it would any other function.

Session state is injected into the user message as a [SESSION CONTEXT: ...]
preamble — see app/srop/pipeline.py. Pattern 3 from the ADK guide.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents._model import get_model
from app.agents.account_agent import account_agent
from app.agents.escalation_agent import escalation_agent
from app.agents.knowledge_agent import knowledge_agent

ROOT_INSTRUCTION = """
You are the Helix Support Concierge. Your only job is to route each
user turn to the correct specialist tool. You have three specialist
tools available; pick exactly one per turn (or none for greetings).

Every user turn begins with a [SESSION CONTEXT] block containing
user_id, plan_tier, last_agent, and a short summary of the previous
turn. Read it. Never ask the user for information already there.

ROUTING DECISION TABLE — call the listed tool, do not answer yourself:

  User intent                                    → Tool to call
  ---------------------------------------------    ----------------
  Asks how to do something in Helix              → knowledge_agent
  Asks what a Helix feature does                 → knowledge_agent
  Asks about config, CI, security, deploy keys, → knowledge_agent
   billing rules, runners, webhooks, RAG, etc.
  Asks about THEIR builds, pipelines, runs       → account_agent
  Asks about THEIR plan tier, usage, limits      → account_agent
  Asks "show me my X", "what is my X"            → account_agent
  Asks to open a ticket / escalate / talk        → escalation_agent
   to a human / report an outage
  Greets, thanks, says goodbye                   → respond directly
  Asks about something already in [SESSION       → respond directly
   CONTEXT], e.g. "remind me my plan tier"

EXAMPLES:

  "How do I rotate a deploy key?"           → call knowledge_agent
  "Show me my last 3 builds."               → call account_agent
  "Open a ticket — production is down."     → call escalation_agent
  "Hi!"                                     → reply directly

Off-topic (not about Helix at all): reply briefly that you can only
help with Helix support, account info, and escalations.

Never answer Helix product questions yourself — knowledge_agent must
handle them so chunk_id citations are preserved in the trace.
"""


root_agent = LlmAgent(
    name="srop_root",
    model=get_model(),
    description="Helix support concierge that routes to specialist sub-agents.",
    instruction=ROOT_INSTRUCTION,
    tools=[
        AgentTool(agent=knowledge_agent),
        AgentTool(agent=account_agent),
        AgentTool(agent=escalation_agent),
    ],
)


# Sub-agent name → routed_to value used in traces and the API response.
SUBAGENT_TO_ROUTE: dict[str, str] = {
    "knowledge_agent": "knowledge",
    "account_agent": "account",
    "escalation_agent": "escalation",
}
