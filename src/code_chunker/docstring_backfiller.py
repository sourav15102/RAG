"""
LLM-based docstring backfiller for CodeChunk objects.

Uses the DeepSeek API (OpenAI-compatible) to generate concise
docstrings for chunks that lack one.
"""

import json
import os
import urllib.request
from typing import Optional

from .ast_chunker import CodeChunk


DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_CHARS = 200
DEFAULT_TIMEOUT = 30


SYSTEM_PROMPT = (
    "You are a code summarizer. Given a piece of Python source code, "
    "produce a concise plain-text docstring that describes what it does. "
    "Respond with ONLY the docstring text — no markdown, no surrounding quotes, no explanation."
)


def _build_prompt(chunk: CodeChunk) -> str:
    ctx = ""
    if chunk.parent_class:
        ctx += f"Part of class: {chunk.parent_class}\n"
    ctx += f"Type: {chunk.chunk_type}\n"
    if chunk.decorators:
        ctx += f"Decorators: {', '.join(chunk.decorators)}\n"
    ctx += f"\nCode:\n{chunk.content}"
    return ctx


class DocstringBackfiller:
    """Fills missing docstrings on CodeChunk objects via an LLM API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_summary_chars: int = DEFAULT_MAX_CHARS,
        base_url: str = "https://api.deepseek.com",
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "api_key is required — pass it directly or set the DEEPSEEK_API_KEY env var"
            )
        self.model = model
        self.max_summary_chars = max_summary_chars
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backfill(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Return a new list of chunks with missing docstrings filled by the LLM.

        Chunks that already have a docstring are returned unchanged.
        """
        filled = []
        for chunk in chunks:
            if chunk.docstring:
                filled.append(chunk)
                continue

            summary = self._summarize(chunk)
            filled.append(CodeChunk(
                content=chunk.content,
                chunk_type=chunk.chunk_type,
                name=chunk.name,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                parent_class=chunk.parent_class,
                decorators=chunk.decorators,
                docstring=summary,
                calls=chunk.calls,
                is_fallback_split=chunk.is_fallback_split,
            ))
        return filled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summarize(self, chunk: CodeChunk) -> str:
        prompt = _build_prompt(chunk)
        user_msg = (
            f"{prompt}\n\n"
            f"Summarize in at most {self.max_summary_chars} characters."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": self.max_summary_chars * 2,
            "temperature": 0.3,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            text = result["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            text = f"[summary failed: {exc}]"

        if len(text) > self.max_summary_chars:
            text = text[: self.max_summary_chars].rsplit(" ", 1)[0] + "…"
        return text
