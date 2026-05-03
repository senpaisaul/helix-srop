"""
SROP pipeline — the heart of the assignment.

For one turn this module:
  1. Loads the Session row + SessionState from DB (rebuild from JSON column).
  2. Loads the last few messages for short-term context.
  3. Composes a [SESSION CONTEXT: ...] preamble and prepends it to the user
     message. This is Pattern 3 from the ADK guide: state lives in the DB,
     gets injected per turn, agent sees no internal session API.
  4. Runs the root orchestrator with asyncio.wait_for so a stuck LLM
     becomes a 504 instead of hanging the request.
  5. Walks the ADK event stream to extract:
        routed_to       (which AgentTool the LLM picked)
        tool_calls      (every function call the agents made)
        retrieved_chunk_ids  (pulled from search_docs_tool results)
        final_text      (the assistant's reply)
  6. Persists the user message, assistant message, AgentTrace row, and the
     updated SessionState — all in one transaction for restart safety.

Restart-survival proof: nothing turn-relevant lives in process memory.
Kill uvicorn between turns; on restart the next request reads state
from sessions.state and message history from messages — both are the
ground truth.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message
from app.db.models import Session as SessionModel
from app.settings import settings
from app.srop import trace_context
from app.srop.state import SessionState

log = structlog.get_logger()

RECENT_MESSAGE_WINDOW = 6  # last N messages included verbatim in preamble


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


@dataclass
class _TurnCapture:
    """Mutable accumulator for what we extract from the ADK event stream."""

    routed_to: str = "smalltalk"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    final_text: str = ""


# ---------------------------------------------------------------------------
# State + history loading
# ---------------------------------------------------------------------------


async def _load_session(session_id: str, db: AsyncSession) -> SessionModel:
    result = await db.execute(
        select(SessionModel).where(SessionModel.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise SessionNotFoundError(f"Session {session_id} does not exist")
    return session


async def _load_recent_messages(
    session_id: str, db: AsyncSession, limit: int = RECENT_MESSAGE_WINDOW
) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    rows = list(result.scalars().all())
    rows.reverse()  # oldest first for the preamble
    return rows


def _build_context_preamble(state: SessionState, recent: list[Message]) -> str:
    history_lines: list[str] = []
    for m in recent[-RECENT_MESSAGE_WINDOW:]:
        snippet = (m.content or "").strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        history_lines.append(f"  {m.role}: {snippet}")
    history_block = "\n".join(history_lines) if history_lines else "  (none)"
    return (
        "[SESSION CONTEXT]\n"
        f"  user_id: {state.user_id}\n"
        f"  plan_tier: {state.plan_tier}\n"
        f"  last_agent: {state.last_agent or 'none'}\n"
        f"  turn: {state.turn_count + 1}\n"
        f"  open_tickets: {state.open_ticket_ids or '[]'}\n"
        f"  last_summary: {state.last_summary or '(first turn)'}\n"
        "[RECENT HISTORY]\n"
        f"{history_block}\n"
    )


# ---------------------------------------------------------------------------
# ADK event-stream extraction
# ---------------------------------------------------------------------------

_AGENT_TOOL_NAMES = {"knowledge_agent", "account_agent", "escalation_agent"}
_SUBAGENT_TO_ROUTE = {
    "knowledge_agent": "knowledge",
    "account_agent": "account",
    "escalation_agent": "escalation",
}


def _extract_function_calls(event: Any) -> list[Any]:
    """Pull function_call parts from a Gemini-style event content."""
    out: list[Any] = []
    content = getattr(event, "content", None)
    if not content:
        return out
    parts = getattr(content, "parts", None) or []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc is not None and getattr(fc, "name", None):
            out.append(fc)
    return out


def _extract_function_responses(event: Any) -> list[Any]:
    out: list[Any] = []
    content = getattr(event, "content", None)
    if not content:
        return out
    parts = getattr(content, "parts", None) or []
    for part in parts:
        fr = getattr(part, "function_response", None)
        if fr is not None and getattr(fr, "name", None):
            out.append(fr)
    return out


def _extract_text(event: Any) -> str:
    content = getattr(event, "content", None)
    if not content:
        return ""
    parts = getattr(content, "parts", None) or []
    texts = [getattr(p, "text", None) or "" for p in parts]
    return "".join(t for t in texts if t)


async def _consume_events(events_iter: Any) -> _TurnCapture:
    """
    Walk ADK's async event iterator and capture everything we need for
    the trace + final response. Defensive against minor API drift.
    """
    cap = _TurnCapture()
    # Track last text + last sub-agent text as fallbacks: the root's
    # is_final_response() event sometimes carries only a function_call and
    # no text (e.g. when the sub-agent already produced the user-visible
    # answer and the root has nothing to add). In that case, the visible
    # answer is whatever the sub-agent's final text was.
    last_any_text = ""
    last_subagent_text = ""

    async for event in events_iter:
        author = getattr(event, "author", None)

        # Function calls (tool invocations).
        for fc in _extract_function_calls(event):
            name = fc.name
            args = dict(fc.args) if getattr(fc, "args", None) else {}
            if name in _AGENT_TOOL_NAMES:
                cap.routed_to = _SUBAGENT_TO_ROUTE[name]
                # Don't add the AgentTool call itself to tool_calls — it's routing,
                # not a real tool. Sub-tool calls are recorded individually.
                continue
            cap.tool_calls.append({"tool_name": name, "args": args, "result": None})

        # Function results (tool returns).
        for fr in _extract_function_responses(event):
            name = fr.name
            response = getattr(fr, "response", None)
            response_dict: Any
            if response is None:
                response_dict = None
            elif isinstance(response, dict):
                response_dict = response
            else:
                try:
                    response_dict = dict(response)
                except (TypeError, ValueError):
                    response_dict = str(response)
            # Match to most recent uncompleted call by name.
            for tc in reversed(cap.tool_calls):
                if tc["tool_name"] == name and tc["result"] is None:
                    tc["result"] = response_dict
                    break
            # Mine chunk_ids out of search_docs results.
            if name == "search_docs_tool" and isinstance(response_dict, dict):
                for chunk in response_dict.get("chunks", []) or []:
                    cid = chunk.get("chunk_id") if isinstance(chunk, dict) else None
                    if cid:
                        cap.retrieved_chunk_ids.append(cid)

        # Always remember the latest text we've seen, segregating root vs
        # sub-agent so we can fall back if the root emits no final text.
        text = _extract_text(event)
        if text:
            last_any_text = text
            if author and author != "srop_root":
                last_subagent_text = text

        # Final assistant message — last wins.
        if hasattr(event, "is_final_response") and event.is_final_response():
            if text:
                cap.final_text = text

    if not cap.final_text:
        cap.final_text = last_subagent_text or last_any_text
    return cap


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def _persist_turn(
    db: AsyncSession,
    session: SessionModel,
    state: SessionState,
    user_message: str,
    cap: _TurnCapture,
    trace_id: str,
    latency_ms: int,
) -> None:
    user_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session.session_id,
        role="user",
        content=user_message,
        trace_id=trace_id,
    )
    assistant_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session.session_id,
        role="assistant",
        content=cap.final_text,
        trace_id=trace_id,
    )
    trace = AgentTrace(
        trace_id=trace_id,
        session_id=session.session_id,
        routed_to=cap.routed_to,
        tool_calls=cap.tool_calls,
        retrieved_chunk_ids=cap.retrieved_chunk_ids,
        latency_ms=latency_ms,
    )

    # Update state. Fold ticket IDs from any escalation calls into open_ticket_ids.
    state.turn_count += 1
    valid_routes = {"knowledge", "account", "escalation", "smalltalk"}
    state.last_agent = cap.routed_to if cap.routed_to in valid_routes else "smalltalk"  # type: ignore[assignment]
    summary = cap.final_text.strip().replace("\n", " ")
    state.last_summary = summary[:200] + ("..." if len(summary) > 200 else "")
    for tc in cap.tool_calls:
        if tc["tool_name"] == "create_ticket" and isinstance(tc.get("result"), dict):
            tid = tc["result"].get("ticket_id")
            if tid and tid not in state.open_ticket_ids:
                state.open_ticket_ids.append(tid)
    session.state = state.to_db_dict()

    db.add_all([user_msg, assistant_msg, trace])
    await db.commit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _run_orchestrator(session_id: str, user_id: str, prompt: str) -> _TurnCapture:
    """
    Spin up an InMemoryRunner per turn and walk events. The agent
    objects themselves are module-level singletons (built once at
    import time) so we don't pay LlmAgent init cost per request.

    Wrapped in tenacity retry for transient 429/503 errors; the Gemini
    free tier RPM budget is tight (5 req/min on flash) and bursty test
    runs trip it. We honour the API's suggested retry_delay via
    exponential backoff.
    """
    import google.api_core.exceptions as gax
    from google.adk.runners import InMemoryRunner
    from google.genai.types import Content, Part
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    from app.agents.orchestrator import root_agent

    runner = InMemoryRunner(agent=root_agent, app_name="helix_srop")
    # InMemoryRunner.session_service exists on >=0.5; create a fresh ADK
    # session (we keep our own state in DB so we don't reuse ADK sessions).
    adk_session = await runner.session_service.create_session(
        app_name="helix_srop", user_id=user_id, session_id=session_id
    )
    new_message = Content(role="user", parts=[Part.from_text(text=prompt)])

    @retry(
        retry=retry_if_exception_type(
            (gax.ResourceExhausted, gax.ServiceUnavailable, gax.DeadlineExceeded)
        ),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _consume() -> _TurnCapture:
        # Reset trace_context inside the retry so a partial run on attempt N
        # doesn't bleed into attempt N+1.
        trace_context.reset()
        events = runner.run_async(
            user_id=user_id,
            session_id=adk_session.id,
            new_message=new_message,
        )
        return await _consume_events(events)

    return await _consume()


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    """Run one SROP turn end-to-end. Raises SessionNotFoundError / UpstreamTimeoutError."""
    trace_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)

    session = await _load_session(session_id, db)
    state = SessionState.from_db_dict(session.state or {"user_id": session.user_id})
    recent = await _load_recent_messages(session_id, db)

    preamble = _build_context_preamble(state, recent)
    prompt = f"{preamble}\n\n[USER MESSAGE]\n{user_message}"

    # Reset the per-turn trace collector. Tools (search_docs_tool,
    # get_recent_builds, etc.) write into it via app.srop.trace_context;
    # this is the only way to see sub-agent tool calls because ADK's event
    # stream only surfaces the root agent's perspective.
    trace_context.reset()

    started = time.perf_counter()
    try:
        cap = await asyncio.wait_for(
            _run_orchestrator(session_id, state.user_id, prompt),
            timeout=settings.llm_timeout_seconds,
        )
    except TimeoutError as exc:
        # Python 3.11+: asyncio.TimeoutError is an alias for builtin TimeoutError.
        log.warning("llm_timeout", timeout_s=settings.llm_timeout_seconds)
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        ) from exc
    latency_ms = int((time.perf_counter() - started) * 1000)

    # Merge: event-stream gave us routed_to + final_text; contextvar gave us
    # all sub-agent tool calls and the chunk_ids cited.
    cap.tool_calls = trace_context.collected_tool_calls() + cap.tool_calls
    cap.retrieved_chunk_ids = (
        trace_context.collected_chunk_ids() + cap.retrieved_chunk_ids
    )

    if not cap.final_text:
        cap.final_text = "(no response)"

    await _persist_turn(db, session, state, user_message, cap, trace_id, latency_ms)
    log.info(
        "turn_complete",
        routed_to=cap.routed_to,
        latency_ms=latency_ms,
        n_tool_calls=len(cap.tool_calls),
        n_chunks=len(cap.retrieved_chunk_ids),
    )
    return PipelineResult(
        content=cap.final_text, routed_to=cap.routed_to, trace_id=trace_id
    )
