"""
Test fixtures.

Key fixtures:
- `client`: async test client backed by an in-memory SQLite DB.
- `mock_adk`: patches `app.srop.pipeline.run` so tests do not call the
  real LLM. The mock inspects the user message + DB to behave like a
  real pipeline (it writes Message rows, AgentTrace rows, and updates
  SessionState) so state-persistence tests are still meaningful.
"""
from __future__ import annotations

import json
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import AgentTrace, Base, Message
from app.db.models import Session as SessionModel
from app.db.session import get_db
from app.main import app
from app.srop.state import SessionState

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db):
    """
    Async test client with the FastAPI get_db dependency rebound to the
    test session. We yield a per-request session via a generator so each
    handler gets a fresh AsyncSession bound to the in-memory engine.
    """

    async def _override_get_db():
        async with TestSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Pipeline mock — patches at the ADK boundary (app.srop.pipeline.run).
# The mock writes the same DB rows the real pipeline would, so that
# state-persistence and trace assertions remain meaningful.
# ---------------------------------------------------------------------------


def _route_for_message(content: str) -> str:
    text = content.lower()
    # Order matters: account/escalation keywords are more specific than the
    # generic "what is" / "how do i" patterns that mark knowledge questions.
    if any(k in text for k in ("ticket", "escalate", "human", "support agent")):
        return "escalation"
    if any(
        k in text
        for k in ("my plan", "my account", "plan tier", "my build", "build", "usage", "storage")
    ):
        return "account"
    if any(k in text for k in ("rotate", "deploy key", "how do i", "what is", "configure", "ci")):
        return "knowledge"
    return "smalltalk"


def _canned_reply(routed_to: str, content: str, state_plan: str) -> str:
    if routed_to == "knowledge":
        return (
            "To rotate a deploy key in Helix, go to Settings → Deploy Keys → "
            "Rotate. According to [chunk_test_kb_001] this generates a new key "
            "and revokes the old one after the next successful deploy."
        )
    if routed_to == "account":
        if "plan" in content.lower():
            return f"Your plan tier is {state_plan}."
        return "Here are your recent builds: bld_test_001 (passed), bld_test_002 (failed)."
    if routed_to == "escalation":
        return "I've opened a support ticket for you (priority: normal)."
    return "Hi! I can help with Helix docs, your account, or escalations."


@pytest.fixture
def mock_adk(monkeypatch):
    """Patch app.srop.pipeline.run with a deterministic stand-in for the LLM."""
    from app.srop import pipeline as pipeline_module

    async def fake_run(session_id: str, user_message: str, db: AsyncSession):
        # Load session row + state — same as the real pipeline.
        result = await db.execute(
            select(SessionModel).where(SessionModel.session_id == session_id)
        )
        session = result.scalar_one_or_none()
        if session is None:
            from app.api.errors import SessionNotFoundError

            raise SessionNotFoundError(f"Session {session_id} does not exist")

        state = SessionState.from_db_dict(session.state or {"user_id": session.user_id})
        routed_to = _route_for_message(user_message)
        reply_text = _canned_reply(routed_to, user_message, state.plan_tier)
        trace_id = str(uuid.uuid4())

        chunk_ids = ["chunk_test_kb_001", "chunk_test_kb_002"] if routed_to == "knowledge" else []
        tool_calls = []
        if routed_to == "knowledge":
            tool_calls.append(
                {
                    "tool_name": "search_docs_tool",
                    "args": {"query": user_message, "k": 5},
                    "result": {"chunks": [{"chunk_id": c} for c in chunk_ids]},
                }
            )

        # Persist messages and trace just like the real pipeline does.
        db.add_all(
            [
                Message(
                    message_id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="user",
                    content=user_message,
                    trace_id=trace_id,
                ),
                Message(
                    message_id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="assistant",
                    content=reply_text,
                    trace_id=trace_id,
                ),
                AgentTrace(
                    trace_id=trace_id,
                    session_id=session_id,
                    routed_to=routed_to,
                    tool_calls=tool_calls,
                    retrieved_chunk_ids=chunk_ids,
                    latency_ms=12,
                ),
            ]
        )
        state.turn_count += 1
        state.last_agent = routed_to  # type: ignore[assignment]
        state.last_summary = reply_text[:200]
        session.state = state.to_db_dict()
        await db.commit()

        return pipeline_module.PipelineResult(
            content=reply_text, routed_to=routed_to, trace_id=trace_id
        )

    monkeypatch.setattr(pipeline_module, "run", fake_run)
    return fake_run


# ---------------------------------------------------------------------------
# search_docs mock — used by test_retriever.py without spinning up Chroma.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_search_docs(monkeypatch):
    from app.agents.tools import search_docs as sd_module

    async def fake_search(query: str, k: int = 5, product_area: str | None = None):
        return [
            sd_module.DocChunk(
                chunk_id=f"chunk_fake_{i:03d}",
                score=round(0.9 - i * 0.1, 4),
                content=f"Fake chunk {i} content matching '{query}'",
                metadata={"source": f"file_{i}.md", "product_area": "security"},
            )
            for i in range(min(k, 3))
        ]

    monkeypatch.setattr(sd_module, "search_docs", fake_search)
    return fake_search


def _drop_unused(*_args, **_kwargs) -> None:
    """Helper to silence linters about unused fixtures in some tests."""
    return json.dumps({})  # never called
