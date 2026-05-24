from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from src.models import ChildChunk, ParentChunk, RawDocument


@dataclass
class ChunkingConfig:
    parent_tokens: int = 1500
    child_tokens: int = 300
    child_overlap_tokens: int = 50


class ParentChildChunker:
    """
    Splits a RawDocument into large parent chunks and small overlapping child chunks.

    Children are what get embedded and searched. When a child is retrieved, its
    parent (with full surrounding context) is what gets passed to the LLM.
    """

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self._enc = tiktoken.get_encoding("cl100k_base")

    def chunk(self, doc: RawDocument) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parents = self._make_parents(doc)
        children = []
        for parent in parents:
            children.extend(self._make_children(parent))
        return parents, children

    # ------------------------------------------------------------------ #

    def _make_parents(self, doc: RawDocument) -> list[ParentChunk]:
        tokens = self._enc.encode(doc.content)
        size = self.config.parent_tokens
        parents = []
        for i, start in enumerate(range(0, len(tokens), size)):
            text = self._enc.decode(tokens[start : start + size])
            parents.append(
                ParentChunk(
                    id=f"{doc.id}_p{i}",
                    doc_id=doc.id,
                    content=text,
                    metadata={**doc.metadata, "parent_index": i},
                )
            )
        return parents

    def _make_children(self, parent: ParentChunk) -> list[ChildChunk]:
        tokens = self._enc.encode(parent.content)
        size = self.config.child_tokens
        step = size - self.config.child_overlap_tokens
        children = []
        i = 0
        start = 0
        while start < len(tokens):
            end = min(start + size, len(tokens))
            text = self._enc.decode(tokens[start:end])
            children.append(
                ChildChunk(
                    id=f"{parent.id}_c{i}",
                    parent_id=parent.id,
                    doc_id=parent.doc_id,
                    content=text,
                    metadata={**parent.metadata, "child_index": i},
                )
            )
            if end == len(tokens):
                break
            start += step
            i += 1
        return children
