import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from code_chunker.code_chunker import CodeChunker
from embedder.code_embedder import CodeEmbedder
from ingester.steps.search_step import SearchStep
from ingester.storage.bm25_store import BM25Store
from ingester.storage.qdrant_store import QdrantStore
from ingester.step import PipelineContext

load_dotenv()

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

SAMPLE_DIR = "sample_docs"
GOLDEN_PATH = "sample_docs/golden_set.json"
OUTPUT_PATH = "eval_results.json"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
ES_HOST = "http://localhost:9200"
ES_INDEX = "code_chunks"
QDRANT_COLLECTION = "code_chunks"


class RAGEvaluator:
    def __init__(
        self,
        sample_dir: str = SAMPLE_DIR,
        golden_path: str = GOLDEN_PATH,
        output_path: str = OUTPUT_PATH,
        top_k: int = 20,
        rerank_top_n: int = 5,
        rewrite: bool = False,
        use_hyde: bool = False,
    ):
        self.sample_dir = sample_dir
        self.golden_path = golden_path
        self.output_path = output_path
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.rewrite = rewrite
        self.use_hyde = use_hyde

    def run(self) -> dict[str, Any]:
        es = Elasticsearch(ES_HOST)
        if es.indices.exists(index=ES_INDEX):
            es.indices.delete(index=ES_INDEX)

        qdrant = QdrantStore(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection=QDRANT_COLLECTION,
        )
        try:
            qdrant._client.delete_collection(collection_name=QDRANT_COLLECTION)
        except Exception:
            pass
        qdrant._ensure_collection()
        bm25 = BM25Store(es=es, index=ES_INDEX)
        bm25.ensure_index()

        embedder = CodeEmbedder(api_key=VOYAGE_API_KEY)
        chunker = CodeChunker(api_key=DEEPSEEK_API_KEY)

        all_chunks: list = []
        for py_file in sorted(Path(self.sample_dir).glob("*.py")):
            chunks = chunker.process(py_file)
            all_chunks.extend(chunks)
            print(f"Chunked {py_file}: {len(chunks)} chunks")
        print(f"\nEmbedding {len(all_chunks)} chunks ...")
        embeddings = embedder.embed(all_chunks)
        print("Upserting to Qdrant ...")
        qdrant.upsert(embeddings)
        print("Indexing to BM25 ...")
        bm25.index(all_chunks)

        with open(self.golden_path) as f:
            golden = json.load(f)
        print(f"\nLoaded {len(golden)} queries from golden set\n")

        search = SearchStep(
            bm25_store=bm25,
            qdrant_store=qdrant,
            embedder=embedder,
            top_k=self.top_k,
            rerank_top_n=self.rerank_top_n,
            llm_api_key=DEEPSEEK_API_KEY,
            rewrite=self.rewrite,
            use_hyde=self.use_hyde,
        )

        per_query: list[dict[str, Any]] = []
        for idx, entry in enumerate(golden):
            qid = entry["query_id"]
            query = entry["query"]
            golden_chunks = set(entry["chunks"])

            ctx = PipelineContext()
            output = search.execute(ctx, query)

            retrieved = output["chunk_ids"]

            if idx < len(golden) - 1:
                time.sleep(22)
            retrieved_set = set(retrieved)
            tp = retrieved_set & golden_chunks

            precision = len(tp) / len(retrieved) if retrieved else 0.0
            recall = len(tp) / len(golden_chunks) if golden_chunks else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            mrr = 0.0
            for rank, cid in enumerate(retrieved, start=1):
                if cid in golden_chunks:
                    mrr = 1.0 / rank
                    break

            per_query.append({
                "query_id": qid,
                "query": query,
                "golden_chunks": list(golden_chunks),
                "retrieved_chunks": retrieved,
                "true_positives": list(tp),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mrr": round(mrr, 4),
            })

            print(f"  {qid}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} MRR={mrr:.3f}")

        n = len(per_query)
        summary = {
            "num_queries": n,
            "eval_k": self.rerank_top_n,
            "avg_precision": round(sum(r["precision"] for r in per_query) / n, 4),
            "avg_recall": round(sum(r["recall"] for r in per_query) / n, 4),
            "avg_f1": round(sum(r["f1"] for r in per_query) / n, 4),
            "avg_mrr": round(sum(r["mrr"] for r in per_query) / n, 4),
            "hit_rate": round(
                sum(1 for r in per_query if r["true_positives"]) / n, 4
            ),
        }

        output = {"summary": summary, "per_query": per_query}
        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n===== Results (k={self.rerank_top_n}) =====")
        print(json.dumps(summary, indent=2))
        print(f"\nFull results saved to {self.output_path}")

        return output


if __name__ == "__main__":
    RAGEvaluator().run()
