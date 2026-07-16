from sentence_transformers import CrossEncoder
from code_chunker.ast_chunker import CodeChunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._model: CrossEncoder | None = None

    def _load(self) -> None:
        if self._model is None:
            self._model = CrossEncoder(self._model_name, device=self._device)

    def rerank(
        self,
        query: str,
        chunks: list[CodeChunk],
        top_n: int = 5,
    ) -> list[CodeChunk]:
        if not chunks:
            return []
        self._load()
        pairs = [(query, c.content) for c in chunks]
        scores = self._model.predict(pairs)
        scored = list(zip(scores, chunks))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_n]]
