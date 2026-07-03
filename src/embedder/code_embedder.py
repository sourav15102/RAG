from dataclasses import dataclass
from typing import Optional

from code_chunker.ast_chunker import CodeChunk

from .voyage_client import VoyageClient


@dataclass
class CodeEmbedding:
    chunk: CodeChunk
    embedding: list[float]


def _format_code_input(chunk: CodeChunk) -> str:
    parts = [f"# {chunk.chunk_type}: {chunk.name}"]
    if chunk.parent_class:
        parts.append(f"# class: {chunk.parent_class}")
    if chunk.docstring:
        parts.append(f"# docstring: {chunk.docstring}")
    parts.append(chunk.content)
    return "\n".join(parts)


class CodeEmbedder:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-4",
        batch_size: int = 128,
    ):
        self._client = VoyageClient(api_key=api_key, model=model)
        self.batch_size = batch_size

    def embed(self, chunks: list[CodeChunk]) -> list[CodeEmbedding]:
        texts = [_format_code_input(c) for c in chunks]
        embeddings = self._client.embed(texts, input_type="document")
        return [
            CodeEmbedding(chunk=chunk, embedding=emb)
            for chunk, emb in zip(chunks, embeddings)
        ]

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed([text], input_type="query")[0]
