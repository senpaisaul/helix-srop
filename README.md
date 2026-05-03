# Helix SROP — Stateful RAG Orchestration Pipeline

**Submission by Abhay Sengar** · ServiceHive GenAI Engineer take-home

An AI support concierge for the (fictitious) Helix dev-tools platform. One
conversation handles **product knowledge questions** (RAG over `docs/`),
**account lookups** (mock build/plan tools), and **escalation** (creates
support tickets) — and survives a `uvicorn` restart mid-conversation.

Built with FastAPI + Google ADK + LiteLLM (Claude Sonnet 4.5) +
SQLAlchemy 2.x async + ChromaDB.

- **Repo:** <https://github.com/senpaisaul/helix-srop>
- **Demo (Loom):** _[link added after recording — see bottom of file]_

---

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/senpaisaul/helix-srop.git
cd helix-srop
uv sync --all-extras
cp .env.example .env                            # paste your keys (see below)
uv run python -m app.rag.ingest --path docs/    # ~90 s, embeds 14 markdown files
uv run uvicorn app.main:app --reload
```

### Keys

- **`GOOGLE_API_KEY`** — required. Used for `gemini-embedding-001` only;
  the daily embed quota is plenty for `ingest --reset` plus a long
  conversation. Get one at [aistudio.google.com](https://aistudio.google.com/app/apikey).
- **`ANTHROPIC_API_KEY`** — optional but recommended. With
  `ADK_MODEL=anthropic/claude-sonnet-4-5` (the default in `.env.example`),
  the chat path goes through ADK's LiteLLM bridge to Claude Sonnet 4.5,
  which routes substantially more reliably than `gemini-2.5-flash-lite`
  on free tier and gives you headroom for live demos. To stay
  Google-only, set `ADK_MODEL=gemini-2.5-flash` (note: free tier is
  20 chat requests/day).

### Tests

`pytest` runs without any key — the LLM is mocked at the ADK boundary.

```bash
uv run pytest -q     # 11 passed, 1 skipped (gated live RAG test)
uv run ruff check app tests eval
```

## Quick Test

> If you don't have `jq` on Windows, replace `jq .` / `jq -r .session_id`
> with `python -m json.tool` / a small Python parser (shown below).

```bash
# 1. Create a session.
SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u_demo","plan_tier":"pro"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "session_id=$SESSION"

# 2. Knowledge question — routes to KnowledgeAgent, cites chunk IDs.
RESP=$(curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content":"How do I rotate a deploy key?"}')
echo "$RESP" | python -m json.tool
TRACE=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace_id'])")

# 3. Inspect the trace — routed_to, tool_calls, retrieved chunk_ids, latency_ms.
curl -s localhost:8000/v1/traces/$TRACE | python -m json.tool

# 4. Account question — same session, routes to AccountAgent.
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content":"Show me my last 3 builds."}' | python -m json.tool
```

### Restart-survival demo (the headline test)

```bash
# Terminal A
uv run uvicorn app.main:app
# ... run the Quick Test above in Terminal B ...

# Now in Terminal A: kill the server.
Ctrl+C

# Restart it. Brand-new process, in-memory state lost.
uv run uvicorn app.main:app

# Terminal B (same shell, $SESSION still set): ask a context-dependent question.
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content":"Quick check: which plan tier am I on, and what was the last thing we discussed?"}' \
  | python -m json.tool
# → reply correctly states "Pro" and recalls the previous turns,
#   loaded from sessions.state JSON + messages table after the restart.
```

## Architecture

```
                       ┌──────────────────────────────────┐
                       │       FastAPI (app/main.py)      │
                       │  POST /v1/sessions               │
  client ──HTTP──▶     │  POST /v1/chat/{id}              │   ──▶ /healthz
                       │  GET  /v1/traces/{id}            │
                       └────────────────┬─────────────────┘
                                        │
                                        ▼
                       ┌──────────────────────────────────┐
                       │  app/srop/pipeline.run()         │
                       │  • load Session + state JSON     │
                       │  • load last 6 messages          │
                       │  • build [SESSION CONTEXT]       │
                       │  • asyncio.wait_for(LLM)         │
                       │  • walk ADK events               │
                       │  • persist msgs + trace + state  │
                       └────────────────┬─────────────────┘
                                        │ ADK runner
                                        ▼
                       ┌──────────────────────────────────┐
                       │  Root LlmAgent (orchestrator)    │
                       │  tools: AgentTool(...)           │
                       └────┬─────────────┬─────────────┬─┘
                            ▼             ▼             ▼
                  knowledge_agent   account_agent  escalation_agent
                  └ search_docs    └ get_recent_   └ create_ticket
                                     builds
                                     get_account_
                                     status
                            │             │             │
                            ▼             ▼             ▼
                       Chroma         (mock data,   tickets table
                       (docs/*.md)    deterministic (E2)
                                      per user_id)

  Persistence: SQLite + aiosqlite via async SQLAlchemy 2.x
  Tables: users, sessions, messages, agent_traces, tickets, idempotency_keys
```

## Design Decisions

### State persistence — Pattern 3 (state JSON column + dynamic preamble)

I chose **Pattern 3** from the ADK guide: `SessionState` is stored as a JSON
blob on the `sessions` row, and on every turn the pipeline rebuilds a
`[SESSION CONTEXT: ...]` preamble (plan_tier, last_agent, turn count,
open ticket IDs, last summary) plus the most recent 6 messages, then
prepends that to the user's message before running the agent.

Why this over Pattern 1 (custom `BaseSessionService`):
- **Restart-safe by construction.** Nothing turn-relevant lives in process
  memory; killing `uvicorn` between turns cannot lose context.
- **No coupling to ADK's session API.** `google-adk` 0.5+ session contracts
  are still evolving; keeping our state plumbing in our own DB shields
  us from breaking changes.
- **One source of truth.** Messages and state live in one transaction,
  so partial-write inconsistencies aren't possible.

Tradeoff: each turn pays a small token cost for the preamble. For long
sessions this would warrant summarization; for the assignment's scope
the truncated 6-message window is sufficient.

### Chunking — heading-aware, sentence-fallback for oversized sections

The `docs/` corpus is well-structured Markdown with H2/H3 sections per
topic. We split on H2/H3 boundaries so each chunk is one coherent
sub-topic; sections longer than 800 characters are sentence-split as a
fallback. No overlap — overlap on already-coherent sections inflates
top-k with redundant near-duplicates of the same topic.

Stable IDs: `chunk_<sha256(rel_path::index)[:16]>`. Re-ingestion is
idempotent via Chroma's `upsert` keyed by these IDs.

### Vector store — Chroma

Chroma was the simplest option that satisfied the rubric: persistent on
disk via `PersistentClient(path=...)`, cosine similarity, list+filter
metadata, no separate server process. LanceDB would also work; FAISS
would have required hand-rolled persistence.

Embeddings: Google `gemini-embedding-001` (3072-d, retrieval_document /
retrieval_query task types). Same model on both sides — mixing
embedding models silently degrades retrieval quality. The ingest
throttles to ≤ 1 req/s and retries on `ResourceExhausted` via tenacity
to stay friendly with the free-tier quota (100 req/min, 1000 req/day).

### Provider abstraction — LiteLLM via ADK

`app/agents/_model.py::get_model()` returns either a bare string (for
native Gemini) or a `LiteLlm(model="anthropic/...")` wrapper based on
the `ADK_MODEL` env var. All four agents (root + three specialists)
call `get_model()`, so swapping providers is a single `.env` line.
Embeddings stay on Google regardless of chat provider, since Anthropic
has no embedding API.

Tested live with `anthropic/claude-sonnet-4-5` — routing accuracy was
noticeably better than `gemini-2.5-flash-lite` on the same prompts.

### Routing — `AgentTool`, never string parsing

The root orchestrator has three sub-agents wrapped as `AgentTool`s. The
LLM picks one via tool selection; nothing in this codebase branches on
substring match against the user's query. `routed_to` is extracted from
the ADK event stream by inspecting `function_call.name` against the
known sub-agent names — see `app/srop/pipeline.py::_consume_events`.

### Trace shape

One row per turn in `agent_traces`:

| field                | description                                         |
|----------------------|-----------------------------------------------------|
| `trace_id`           | UUID, returned in every chat response               |
| `session_id`         | session this turn belongs to                        |
| `routed_to`          | knowledge / account / escalation / smalltalk        |
| `tool_calls`         | `[{tool_name, args, result}, …]` for non-routing tools |
| `retrieved_chunk_ids`| chunk_ids cited in this turn                        |
| `latency_ms`         | wall-clock time for the LLM run                     |

`GET /v1/traces/{id}` returns this verbatim — useful as a debug surface
when an answer looks wrong (you can verify which chunks the model saw).

## Extensions Completed

- [x] **E1 Idempotency** — `Idempotency-Key` header; `(session_id, key)` is
  unique-indexed in `idempotency_keys`; replay returns the cached
  `ChatResponse` JSON without re-running the pipeline.
- [x] **E2 Escalation agent** — third sub-agent `escalation_agent`,
  `create_ticket` tool, `tickets` table; ticket IDs flow into
  `state.open_ticket_ids` so subsequent turns can reference them.
- [ ] E3 Streaming SSE — skipped; ADK's per-turn streaming wiring
  outweighed the 5-pt value within the time budget.
- [ ] E4 Reranking — skipped for the same reason.
- [x] **E5 Guardrails** — root agent's instruction refuses off-topic
  questions; `app/obs/redact.py` is a structlog processor that masks
  emails, long hex tokens, and `Authorization` / `api_key` headers in
  every log record.
- [x] **E6 Docker** — `Dockerfile` (python:3.12-slim, uv, copies `app/` +
  `docs/`) and `docker-compose.yml` with a named volume for
  `/data/chroma_db` + `/data/helix_srop.db` and a `/healthz` healthcheck.
- [x] **E7 Eval harness** — `eval/run_eval.py` posts 8 (question,
  expected_route) cases against the live API and prints overall +
  per-route accuracy.

## Known Limitations

- No authentication on the API. A real deployment would need OAuth or
  an API-key middleware before `routes_chat` accepts traffic.
- AccountAgent uses deterministic mock data per `user_id`; it does not
  query a real backend.
- The `[SESSION CONTEXT]` preamble does not summarize beyond the last 6
  messages — long conversations would need rolling summarization.
- Rate limits aren't enforced. ADK + Gemini free tier is the only
  practical throttle today.
- Chroma cosine distance ↦ score is a linear `1 - d` clamp; for some
  embedding distributions a calibrated map (e.g. via reranker scores)
  would be more meaningful.

## What I'd Do With More Time

- Replace the per-turn `InMemoryRunner` with a custom
  `DatabaseSessionService` so ADK's own session view stays consistent
  with ours (Pattern 1 hybrid).
- Reranking (E4) with a small cross-encoder, gated behind a
  `RERANK=1` env var.
- Pull tracing into OpenTelemetry — `app/obs/logging.py` would be a
  thin shim over OTel spans, and the `agent_traces` table would mirror
  the trace exporter.
- Postgres + Alembic migrations for production parity (already
  schema-compatible — just swap the URL).

## Time Spent

| Phase                                              | Time   |
| -------------------------------------------------- | ------ |
| Setup, repo exploration, dependency resolution     |  ~1 h  |
| RAG ingest + search_docs + Chroma plumbing         |  ~1 h  |
| ADK agents (knowledge / account / escalation / root) | ~1.5 h |
| Pipeline + state persistence + trace extraction    |  ~2 h  |
| API routes (sessions, chat with idempotency, traces) |  ~1 h  |
| Tests (mocks, integration, retriever)              |  ~1 h  |
| Docker, eval harness, guardrails, README           |  ~1 h  |
| **Total**                                          | **~8.5 h** |

---

## Demo

_Loom walkthrough (≤ 4 min):_ **_paste the link here after recording_**

The demo covers: clean clone, `pytest` green, server boot, knowledge turn
with chunk-ID citation, trace inspection, account turn, **uvicorn kill +
restart**, post-restart context-dependent follow-up demonstrating that
`plan_tier`, recent messages, and `last_summary` survive process death.
