import os
from typing import Optional

from code_chunker.ast_chunker import CodeChunk
from search.utils.llm import _llm_complete


SYSTEM_PROMPT = (
    "You are a code understanding assistant. Given a user's question and "
    "relevant code chunks, produce a clear answer that references the code. "
    "Be concise and specific."
)


def _build_context(chunks: list[CodeChunk]) -> str:
    parts = []
    for chunk in chunks:
        header = f"# {chunk.chunk_type}: {chunk.name}  ({chunk.file_path}:{chunk.start_line})"
        if chunk.docstring:
            header += f"\n# docstring: {chunk.docstring}"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n".join(parts)


def generate(
    query: str,
    chunks: list[CodeChunk],
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    timeout: int = 30,
) -> str:
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return "[generation failed: no API key]"

    context = _build_context(chunks)
    user_msg = f"Question: {query}\n\nRelevant code:\n{context}"

    try:
        return _llm_complete(SYSTEM_PROMPT, user_msg, api_key, model, base_url, timeout)
    except Exception as exc:
        return f"[generation failed: {exc}]"
