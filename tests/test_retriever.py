"""
Unit tests for RAG retrieval.

The real retrieval test (against Chroma + a live embedding model) is
gated behind RUN_LIVE_RAG=1; CI runs against the mocked variant so it
does not need a Google API key.
"""
from __future__ import annotations

import os

import pytest

from app.agents.tools.search_docs import search_docs
from app.rag.ingest import chunk_markdown, extract_frontmatter, make_chunk_id


def test_chunker_produces_non_empty_chunks() -> None:
    text = (
        "# Header\n\nIntro paragraph.\n\n"
        "## Section A\n\nSection A body. Sentence two.\n\n"
        "## Section B\n\nSection B body."
    )
    chunks = chunk_markdown(text, chunk_size=200)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)


def test_frontmatter_extracted() -> None:
    text = "---\ntitle: Demo\nproduct_area: security\n---\nbody text here"
    metadata, body = extract_frontmatter(text)
    assert metadata["title"] == "Demo"
    assert metadata["product_area"] == "security"
    assert body.strip() == "body text here"


def test_chunk_ids_are_stable_and_deterministic() -> None:
    a = make_chunk_id("docs/foo.md", 3)
    b = make_chunk_id("docs/foo.md", 3)
    c = make_chunk_id("docs/foo.md", 4)
    assert a == b
    assert a != c
    assert a.startswith("chunk_")


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids(mock_search_docs) -> None:
    """Mocked variant — exercises the search_docs contract without Chroma."""
    results = await mock_search_docs("how to rotate a deploy key", k=3)
    assert len(results) > 0
    assert all(r.chunk_id for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_RAG") != "1",
    reason="Live RAG test — set RUN_LIVE_RAG=1 after running ingest",
)
@pytest.mark.asyncio
async def test_search_docs_live() -> None:
    results = await search_docs("how to rotate a deploy key", k=3)
    assert len(results) > 0
    assert all(r.chunk_id for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)
