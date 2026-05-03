"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 800
    python -m app.rag.ingest --path docs/ --reset

Reads markdown files, extracts YAML frontmatter, chunks heading-aware,
embeds with Google text-embedding-004, and upserts into Chroma.

Re-ingestion is idempotent: chunk IDs are derived from
sha256(file_path::chunk_index)[:16], so running twice does not duplicate.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

from app.rag.store import embed_documents, get_collection, reset_collection

DEFAULT_CHUNK_SIZE = 800
# Gemini free tier is 100 embed_content requests/min. We embed one chunk
# per call (gemini-embedding-001 returns one vector per call regardless
# of list input), so we throttle by inserting a small inter-call sleep.
EMBED_BATCH = 1
INTER_CALL_DELAY_SECONDS = 0.7


def extract_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """
    Parse a YAML-style frontmatter block at the top of a markdown file.

    Returns (metadata_dict, body_without_frontmatter). If the file has no
    frontmatter, returns ({}, original_text).

    We avoid importing pyyaml to keep the dep tree small — only flat
    key:value and simple list values are supported (which is what the
    corpus uses).
    """
    match = re.match(r"^---\n(.*?)\n---\n?", text, re.DOTALL)
    if not match:
        return {}, text
    body = text[match.end():]
    block = match.group(1)
    metadata: dict[str, Any] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            metadata[key] = [v.strip() for v in inner.split(",") if v.strip()]
        else:
            metadata[key] = value
    return metadata, body


def _split_long_section(section: str, max_chars: int) -> list[str]:
    """Sentence-aware fallback for sections that exceed max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", section)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current).strip())
            current = []
            current_len = 0
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


def chunk_markdown(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = 0) -> list[str]:
    """
    Heading-aware chunker for markdown.

    Splits on H2/H3 headings so each chunk corresponds to a coherent
    sub-section. Sections longer than chunk_size are sentence-split.

    The `overlap` arg is accepted for API symmetry but ignored — heading
    boundaries already give us coherent units, and overlap on coherent
    sections degrades retrieval precision (multiple chunks for the same
    topic dominate top-k).
    """
    del overlap  # see docstring
    # Split on lines that begin a new ## or ### heading. Keep the heading
    # attached to its body via lookahead.
    sections = re.split(r"\n(?=#{2,3} )", text)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(_split_long_section(section, chunk_size))
    return chunks


def make_chunk_id(file_path: str, chunk_index: int) -> str:
    """Stable ID — re-ingestion of the same file produces the same IDs."""
    raw = f"{file_path}::{chunk_index}".encode()
    return "chunk_" + hashlib.sha256(raw).hexdigest()[:16]


def extract_metadata(file_path: Path, text: str) -> tuple[dict[str, Any], str]:
    metadata, body = extract_frontmatter(text)
    metadata.setdefault("source", file_path.name)
    metadata.setdefault("title", file_path.stem)
    metadata.setdefault("product_area", "general")
    # Chroma rejects list values in metadata; flatten tags to a comma-separated string.
    if isinstance(metadata.get("tags"), list):
        metadata["tags"] = ", ".join(metadata["tags"])
    return metadata, body


async def ingest_directory(
    docs_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = 0,
    reset: bool = False,
) -> dict[str, int]:
    """
    Walk docs_path, chunk + embed every .md file, upsert into Chroma.
    Returns {files: N, chunks: M}.
    """
    if reset:
        print("Resetting collection...")
        reset_collection()

    md_files = sorted(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")
    if not md_files:
        return {"files": 0, "chunks": 0}

    coll = get_collection()
    total_chunks = 0

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata, body = extract_metadata(file_path, text)
        chunks = chunk_markdown(body, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            print(f"  {file_path.name}: 0 chunks (skipped)")
            continue

        ids = [make_chunk_id(str(file_path.relative_to(docs_path)), i) for i in range(len(chunks))]
        metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]

        # Embed in small batches to stay under rate limits.
        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            embeddings.extend(await embed_documents(batch))
            await asyncio.sleep(INTER_CALL_DELAY_SECONDS)

        coll.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
        total_chunks += len(chunks)
        print(f"  {file_path.name}: {len(chunks)} chunks")

    print(f"Ingest complete. Files: {len(md_files)}, chunks: {total_chunks}")
    return {"files": len(md_files), "chunks": total_chunks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--reset", action="store_true", help="Drop the collection first")
    args = parser.parse_args()
    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap, args.reset))


if __name__ == "__main__":
    main()
