from __future__ import annotations

import anthropic

SYSTEM_PROMPT = (
    "You are a helpful assistant that writes factual document excerpts. "
    "When given a question, write a short passage (2-4 sentences) that directly "
    "answers it, as if it were an excerpt from a relevant article or document. "
    "Do not reference the question. Do not say 'the answer is'. Just write the passage."
)


class HyDEGenerator:
    """Generates a hypothetical document for a query using Claude."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self._client = anthropic.Anthropic()
        self._model = model

    def generate(self, query: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        return response.content[0].text.strip()
