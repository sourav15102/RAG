import json
import os
import urllib.request
from typing import Optional


VOYAGE_API_BASE = "https://api.voyageai.com/v1"
DEFAULT_MODEL = "voyage-4"
DEFAULT_TIMEOUT = 30
MAX_BATCH_SIZE = 128


class VoyageClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = VOYAGE_API_BASE,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "api_key is required — pass it directly or set the VOYAGE_API_KEY env var"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            results.extend(self._embed_batch(batch, input_type))
        return results

    def _embed_batch(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        payload = {
            "input": texts,
            "model": self.model,
            "input_type": input_type,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
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
        except Exception as exc:
            raise RuntimeError(f"Voyage API call failed: {exc}") from exc

        result["data"].sort(key=lambda d: d["index"])
        return [d["embedding"] for d in result["data"]]
