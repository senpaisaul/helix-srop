"""
Per-turn trace collector via contextvars.

ADK's event stream surfaces only the root agent's view, so any tool
call inside a sub-agent (e.g. KnowledgeAgent → search_docs_tool) is
invisible to our pipeline event walker. We close that gap with a
contextvar: each tool appends itself to a list, and the pipeline
resets+reads the list around the agent run.

Each entry is a dict matching the AgentTrace.tool_calls schema:
    {"tool_name": str, "args": dict, "result": dict | str | None}
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_tool_calls: ContextVar[list[dict[str, Any]]] = ContextVar("_helix_tool_calls", default=[])
_chunk_ids: ContextVar[list[str]] = ContextVar("_helix_chunk_ids", default=[])


def reset() -> None:
    """Call once at the start of pipeline.run()."""
    _tool_calls.set([])
    _chunk_ids.set([])


def record_tool_call(tool_name: str, args: dict[str, Any], result: Any) -> None:
    """Tools call this after they finish. Result is best-effort serializable."""
    calls = _tool_calls.get()
    calls.append({"tool_name": tool_name, "args": args, "result": result})
    _tool_calls.set(calls)


def record_chunk_ids(chunk_ids: list[str]) -> None:
    """search_docs_tool calls this with the list of chunk IDs it returned."""
    if not chunk_ids:
        return
    existing = _chunk_ids.get()
    existing.extend(chunk_ids)
    _chunk_ids.set(existing)


def collected_tool_calls() -> list[dict[str, Any]]:
    return list(_tool_calls.get())


def collected_chunk_ids() -> list[str]:
    return list(_chunk_ids.get())
