from __future__ import annotations

import anthropic

from src.retrieval.retriever import RetrievalResult

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions strictly based on the provided context.
If the context does not contain enough information to answer, say so clearly.
Do not make up facts or draw on knowledge outside the context.\
"""

RAG_TEMPLATE = """\
Context:
{context}

Question: {query}\
"""


def _build_context(results: list[RetrievalResult]) -> str:
    seen: set[str] = set()
    chunks: list[str] = []
    for r in results:
        if r.parent.id not in seen:
            seen.add(r.parent.id)
            source = r.parent.metadata.get("filename", r.parent.id)
            chunks.append(f"[Source: {source}]\n{r.parent.content}")
    return "\n\n---\n\n".join(chunks)


class RAGGenerator:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self._client = anthropic.Anthropic()
        self._model = model

    def build_prompt(self, query: str, results: list[RetrievalResult]) -> tuple[str, str]:
        """Returns (system_prompt, user_message) without calling the API."""
        context = _build_context(results)
        return SYSTEM_PROMPT, RAG_TEMPLATE.format(context=context, query=query)

    def generate(self, query: str, results: list[RetrievalResult]) -> str:
        system, prompt = self.build_prompt(query, results)

        print("\nAnswer:\n")
        full_response = []
        with self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response.append(text)

        print("\n")
        return "".join(full_response)
