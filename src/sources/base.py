from abc import ABC, abstractmethod
from typing import Iterator

from src.models import RawDocument


class DocumentSource(ABC):
    """
    Contract every document source must implement.
    Yield RawDocument objects — the chunker and indexer don't care where they came from.
    """

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source (used in document IDs and metadata)."""
        ...

    @abstractmethod
    def load(self) -> Iterator[RawDocument]:
        """Yield all documents from this source."""
        ...
