"""
search_docs tool — used by KnowledgeAgent.

Performs cosine-similarity search over the Chroma collection and returns
chunk IDs, normalized scores in [0, 1], and content. The agent uses
chunk_id values to cite sources in its answer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.rag.store import embed_query, get_collection

# Chunks below this score are dropped — they're typically off-topic noise.
SCORE_THRESHOLD = 0.3


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any]


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """
    Search the Helix docs vector store and return the top-k matching chunks.

    Args:
        query: natural-language question from the user.
        k: maximum number of chunks to return.
        product_area: optional filter (e.g. 'security', 'ci-cd').

    Returns:
        List of DocChunk ordered by descending score. Each chunk_id is
        stable across ingestions and can be cited in the agent's answer.
    """
    if not query or not query.strip():
        return []

    query_vec = await embed_query(query)
    coll = get_collection()
    where = {"product_area": product_area} if product_area else None

    results = coll.query(
        query_embeddings=[query_vec],
        n_results=max(1, k),
        where=where,
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    chunks: list[DocChunk] = []
    for chunk_id, distance, doc, meta in zip(ids, distances, documents, metadatas, strict=False):
        # Cosine distance in [0, 2]. Map to similarity in [0, 1] and clamp.
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        if score < SCORE_THRESHOLD:
            continue
        chunks.append(
            DocChunk(
                chunk_id=chunk_id,
                score=round(score, 4),
                content=doc,
                metadata=dict(meta) if meta else {},
            )
        )

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks


def format_chunks_for_agent(chunks: list[DocChunk]) -> str:
    """Render retrieval results so the LLM can cite chunk IDs verbatim."""
    if not chunks:
        return "No relevant documentation was found for this query."
    parts: list[str] = []
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        parts.append(
            f"[{c.chunk_id}] (score: {c.score:.2f}, source: {source})\n{c.content}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# ADK tool wrapper.
#
# ADK inspects the function signature + docstring to build the tool schema.
# We expose a thin async function so the LLM sees a single named tool and
# the orchestrator+sub-agent contract stays stable.
# ---------------------------------------------------------------------------


async def search_docs_tool(query: str, k: int = 5) -> dict[str, Any]:
    """
    Search the Helix product documentation for content relevant to the query.

    Use this whenever the user asks how to do something, what something
    means, or about a Helix feature. The result includes citation IDs that
    you must reference in your answer (e.g., "[chunk_abc123]").

    Args:
        query: the user's natural-language question.
        k: how many chunks to retrieve (default 5).

    Returns:
        dict with keys:
        - chunks: list of {chunk_id, score, content, metadata}
        - formatted: human-readable rendering for inclusion in the prompt
    """
    from app.srop import trace_context

    chunks = await search_docs(query, k=k)
    chunk_dicts = [asdict(c) for c in chunks]
    formatted = format_chunks_for_agent(chunks)

    # Side-channel for the pipeline trace collector — sub-agent tool calls
    # do not surface as events at the root runner.
    trace_context.record_chunk_ids([c.chunk_id for c in chunks])
    trace_context.record_tool_call(
        "search_docs_tool",
        {"query": query, "k": k},
        {"chunk_count": len(chunks), "chunk_ids": [c.chunk_id for c in chunks]},
    )

    return {"chunks": chunk_dicts, "formatted": formatted}
