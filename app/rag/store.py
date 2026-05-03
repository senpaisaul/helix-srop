"""
Vector store + embedding helpers shared by ingest and search_docs.

We use Chroma (PersistentClient) at settings.chroma_persist_dir and
Google's text-embedding-004 model. Singletons created lazily so that
unit tests can run without hitting the network.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.settings import settings

COLLECTION_NAME = "helix_docs"
# gemini-embedding-001 is the current stable embedding model.
# Default output dim is 3072; we don't pin it because Chroma allocates
# the index lazily from the first vector's dimension.
EMBED_MODEL = "models/gemini-embedding-001"


@lru_cache(maxsize=1)
def get_chroma_client() -> ClientAPI:
    """One Chroma client per process. Persistence dir comes from settings."""
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection() -> Collection:
    """Idempotent — safe to call from ingest or query paths."""
    return get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_sync(texts: list[str], task_type: str) -> list[list[float]]:
    """
    Synchronous embedding call. Wrap in asyncio.to_thread() from async paths.

    task_type must be 'retrieval_document' at ingest, 'retrieval_query' at search.
    Mixing them silently degrades retrieval quality.

    Wrapped in tenacity retry — Gemini free tier is 100 req/min, and a
    bursty ingest will trip the quota; we honour the suggested retry_delay
    via exponential backoff.
    """
    import google.api_core.exceptions as gax
    import google.generativeai as genai
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env (see .env.example)."
        )
    genai.configure(api_key=settings.google_api_key)

    @retry(
        retry=retry_if_exception_type(
            (gax.ResourceExhausted, gax.ServiceUnavailable, gax.DeadlineExceeded)
        ),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call() -> dict:
        return genai.embed_content(
            model=EMBED_MODEL,
            content=texts,
            task_type=task_type,
        )

    result = _call()
    embedding = result["embedding"]
    # google-generativeai returns a flat list for a single string and
    # a list-of-lists for a list. Normalize to list-of-lists.
    if texts and isinstance(texts, list) and embedding and not isinstance(embedding[0], list):
        embedding = [embedding]
    return embedding


async def embed_documents(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(_embed_sync, texts, "retrieval_document")


async def embed_query(text: str) -> list[float]:
    vectors = await asyncio.to_thread(_embed_sync, [text], "retrieval_query")
    return vectors[0]


def reset_collection() -> None:
    """Drop and recreate the collection. Used by tests and re-ingest --reset."""
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet — that's fine.
        pass
    get_collection()


def collection_stats() -> dict[str, Any]:
    coll = get_collection()
    return {"count": coll.count(), "name": COLLECTION_NAME}
