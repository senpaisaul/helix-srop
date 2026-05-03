"""
KnowledgeAgent — answers product questions using RAG.

Wired as an AgentTool by the root orchestrator. Has exactly one tool:
search_docs_tool. The instruction enforces citation of chunk IDs so the
trace can be audited and the user can verify the source.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent

from app.agents._model import get_model
from app.agents.tools.search_docs import search_docs_tool

KNOWLEDGE_INSTRUCTION = """
You are the Helix Knowledge specialist. Answer Helix product/docs
questions using ONLY the results returned by the search_docs_tool.

Process for every turn:
1. Call search_docs_tool with the user's question (verbatim is fine).
2. Read the returned chunks. Each has a chunk_id like 'chunk_abc123def'.
3. Compose your answer using ONLY those chunks. Do NOT add knowledge
   from training data.
4. Cite chunk IDs inline, e.g. "According to [chunk_abc123def] you can ...".
5. If retrieval returned no relevant chunks, say so plainly. Do not guess.

Never claim a feature, command, or fact that is not in the chunks.
"""


knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=get_model(),
    description=(
        "Specialist that answers Helix product/docs questions using RAG "
        "and cites chunk IDs."
    ),
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[search_docs_tool],
)
