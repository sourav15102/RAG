from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[Any]) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        vector_name: str = "code",
    ) -> list[Any]:
        ...
