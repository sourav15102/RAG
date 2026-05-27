from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import RawDocument
from src.sources.base import DocumentSource


class BaseIndexer(ABC):
    """
    Contract every indexer implementation must fulfill.

    Implementations decide how documents are stored and made searchable —
    e.g. chunk + embed into Qdrant, or build a PageIndex tree via LLM reasoning.
    Swap implementations in ingest.py without touching the rest of the pipeline.
    """

    @abstractmethod
    def add_document(self, doc: RawDocument) -> None:
        """Accept a single raw document. May queue it internally."""
        ...

    @abstractmethod
    def finalize(self) -> None:
        """Flush all queued documents and persist the index."""
        ...

    def index_source(self, source: DocumentSource) -> None:
        """Convenience: iterate a source and index every document."""
        total = 0
        for doc in source.load():
            print(f"Indexing: {doc.metadata.get('filename', doc.id)}")
            self.add_document(doc)
            total += 1
        print(f"\nFinalizing index for {total} document(s)...")
        self.finalize()
