from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from code_chunker.ast_chunker import CodeChunk
from embedder.code_embedder import CodeEmbedding
from ingester.ingester import Ingester
from ingester.pipeline import Pipeline, PipelineConfig
from ingester.step import PipelineContext, Step
from ingester.steps.chunk_step import ChunkStep
from ingester.steps.embed_step import EmbedStep
from ingester.steps.store_step import StoreStep
from ingester.storage.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_code():
    return "def foo():\n    return 42\n"


@pytest.fixture
def sample_chunks():
    return [
        CodeChunk(
            content="def foo():\n    return 42\n",
            chunk_type="function",
            name="foo",
            file_path="test.py",
            start_line=1,
            end_line=2,
        )
    ]


@pytest.fixture
def sample_embeddings(sample_chunks):
    return [
        CodeEmbedding(chunk=sample_chunks[0], embedding=[0.1, 0.2, 0.3])
    ]


# ---------------------------------------------------------------------------
# Step + PipelineContext
# ---------------------------------------------------------------------------

class TestPipelineContext:
    def test_defaults(self):
        ctx = PipelineContext()
        assert ctx.document_path == ""
        assert ctx.document_source == ""
        assert ctx.config == {}
        assert ctx.state == {}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_runs_steps_in_order(self):
        calls = []

        class LogStep(Step):
            name = "log"
            def __init__(self, label):
                self.label = label
            def execute(self, ctx, data):
                calls.append(self.label)
                return data

        pipeline = Pipeline(PipelineConfig(
            name="test",
            steps=[LogStep("a"), LogStep("b"), LogStep("c")],
        ))
        result = pipeline.run("hello")
        assert calls == ["a", "b", "c"]
        assert result == "hello"

    def test_passes_data_between_steps(self):
        class DoubleStep(Step):
            name = "double"
            def execute(self, ctx, data):
                return data * 2

        pipeline = Pipeline(PipelineConfig(
            name="double",
            steps=[DoubleStep(), DoubleStep()],
        ))
        assert pipeline.run(3) == 12

    def test_creates_default_context(self):
        class CaptureCtx(Step):
            name = "capture"
            def execute(self, ctx, data):
                self._ctx = ctx
                return data

        step = CaptureCtx()
        pipeline = Pipeline(PipelineConfig(name="t", steps=[step]))
        pipeline.run("x")
        assert isinstance(step._ctx, PipelineContext)


# ---------------------------------------------------------------------------
# Ingester
# ---------------------------------------------------------------------------

class TestIngester:
    def test_runs_single_pipeline(self):
        class ConstStep(Step):
            name = "c"
            def execute(self, ctx, data):
                return "result"

        ingester = Ingester(pipelines=[
            Pipeline(PipelineConfig(name="pipe", steps=[ConstStep()])),
        ])
        results = ingester.ingest("input")
        assert results == {"pipe": "result"}

    def test_runs_multiple_pipelines_in_parallel(self):
        import time

        class SlowStep(Step):
            name = "slow"
            def __init__(self, label, delay=0.2):
                self.label = label
                self.delay = delay
            def execute(self, ctx, data):
                time.sleep(self.delay)
                return self.label

        ingester = Ingester(pipelines=[
            Pipeline(PipelineConfig(name="a", steps=[SlowStep("a", 0.3)])),
            Pipeline(PipelineConfig(name="b", steps=[SlowStep("b", 0.1)])),
        ])
        start = time.time()
        results = ingester.ingest("x")
        elapsed = time.time() - start
        assert results == {"a": "a", "b": "b"}
        assert elapsed < 0.45  # parallel, not sequential


# ---------------------------------------------------------------------------
# VectorStore (mock implementation)
# ---------------------------------------------------------------------------

class MockVectorStore(VectorStore):
    def __init__(self):
        self.records = []

    def upsert(self, records):
        self.records.extend(records)

    def search(self, query_vector, top_k=10, vector_name="code"):
        return self.records[:top_k]


# ---------------------------------------------------------------------------
# ChunkStep
# ---------------------------------------------------------------------------

class TestChunkStep:
    def test_uses_path_from_ctx(self, sample_code, sample_chunks):
        step = ChunkStep(api_key="sk-test")
        step._chunker = MagicMock(return_value=sample_chunks)
        step._chunker.process = MagicMock(return_value=sample_chunks)

        ctx = PipelineContext(document_path="mod.py")
        result = step.execute(ctx, sample_code)

        step._chunker.process.assert_called_once_with(sample_code, file_path="mod.py")
        assert ctx.state["file_path"] == "mod.py"
        assert result == sample_chunks

    def test_uses_data_as_path_when_file_exists(self, tmp_path, sample_chunks):
        f = tmp_path / "mod.py"
        f.write_text("def foo(): pass")

        step = ChunkStep(api_key="sk-test")
        step._chunker = MagicMock()
        step._chunker.process = MagicMock(return_value=sample_chunks)

        ctx = PipelineContext()
        result = step.execute(ctx, str(f))

        step._chunker.process.assert_called_once()
        assert ctx.state["file_path"] == str(f)
        assert result == sample_chunks


# ---------------------------------------------------------------------------
# EmbedStep
# ---------------------------------------------------------------------------

class TestEmbedStep:
    def test_embeds_chunks(self, sample_chunks, sample_embeddings):
        step = EmbedStep(api_key="vo-test")
        step._embedder = MagicMock()
        step._embedder.embed = MagicMock(return_value=sample_embeddings)

        ctx = PipelineContext()
        result = step.execute(ctx, sample_chunks)

        step._embedder.embed.assert_called_once_with(sample_chunks)
        assert result == sample_embeddings


# ---------------------------------------------------------------------------
# StoreStep
# ---------------------------------------------------------------------------

class TestStoreStep:
    def test_stores_embeddings(self, sample_embeddings):
        store = MockVectorStore()
        step = StoreStep(store)

        ctx = PipelineContext()
        result = step.execute(ctx, sample_embeddings)

        assert store.records == sample_embeddings
        assert result == sample_embeddings  # passes through

    def test_upsert_called_with_correct_data(self, sample_embeddings):
        mock_store = MagicMock(spec=VectorStore)
        step = StoreStep(mock_store)

        ctx = PipelineContext()
        step.execute(ctx, sample_embeddings)

        mock_store.upsert.assert_called_once_with(sample_embeddings)


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestCodeEmbeddingPipeline:
    def test_full_flow_with_mocked_steps(self, sample_code, sample_chunks, sample_embeddings):
        mock_store = MagicMock(spec=VectorStore)

        pipeline = Pipeline(PipelineConfig(
            name="code_embedding",
            steps=[
                ChunkStep(api_key="sk-test"),
                EmbedStep(api_key="vo-test"),
                StoreStep(mock_store),
            ],
        ))

        with (
            patch.object(pipeline.steps[0], "_chunker") as mock_chunker,
            patch.object(pipeline.steps[1], "_embedder") as mock_embedder,
        ):
            mock_chunker.process = MagicMock(return_value=sample_chunks)
            mock_embedder.embed = MagicMock(return_value=sample_embeddings)

            result = pipeline.run(sample_code, PipelineContext(document_path="test.py"))

        mock_chunker.process.assert_called_once_with(sample_code, file_path="test.py")
        mock_embedder.embed.assert_called_once_with(sample_chunks)
        mock_store.upsert.assert_called_once_with(sample_embeddings)
        assert result == sample_embeddings


class TestIngesterWithPipeline:
    def test_ingester_orchestrates_code_pipeline(self, sample_code, sample_chunks, sample_embeddings):
        mock_store = MagicMock(spec=VectorStore)

        pipeline = Pipeline(PipelineConfig(
            name="code_embedding",
            steps=[
                ChunkStep(api_key="sk-test"),
                EmbedStep(api_key="vo-test"),
                StoreStep(mock_store),
            ],
        ))

        ingester = Ingester(pipelines=[pipeline])

        with (
            patch.object(pipeline.steps[0], "_chunker") as mock_chunker,
            patch.object(pipeline.steps[1], "_embedder") as mock_embedder,
        ):
            mock_chunker.process = MagicMock(return_value=sample_chunks)
            mock_embedder.embed = MagicMock(return_value=sample_embeddings)

            results = ingester.ingest(sample_code, PipelineContext(document_path="test.py"))

        assert "code_embedding" in results
        assert results["code_embedding"] == sample_embeddings
        mock_store.upsert.assert_called_once_with(sample_embeddings)
