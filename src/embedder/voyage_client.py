import json
import os
import time
import urllib.request
from typing import Optional


VOYAGE_API_BASE = "https://api.voyageai.com/v1"
DEFAULT_MODEL = "voyage-4"
DEFAULT_TIMEOUT = 30
MAX_BATCH_SIZE = 30


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
        batch_count = 0
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            if batch_count > 0:
                time.sleep(25)
            batch = texts[i : i + MAX_BATCH_SIZE]
            results.extend(self._embed_batch(batch, input_type))
            batch_count += 1
        return results

    def _embed_batch(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        last_exc = None
        for attempt in range(2):
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
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    last_exc = exc
                    time.sleep(15 * (2 ** attempt))
                else:
                    raise RuntimeError(f"Voyage API call failed: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"Voyage API call failed: {exc}") from exc
        else:
            raise RuntimeError(f"Voyage API rate limit exceeded after retries") from last_exc

        result["data"].sort(key=lambda d: d["index"])
        return [d["embedding"] for d in result["data"]]
