import json
from unittest.mock import patch

from code_chunker.ast_chunker import CodeChunk
from search.generator import GenerationResult, _build_context, _parse_claims, generate


def make_chunk(**overrides):
    defaults = dict(
        content="def foo():\n    return 42\n",
        chunk_type="function",
        name="foo",
        file_path="mod.py",
        start_line=1,
        end_line=2,
    )
    defaults.update(overrides)
    return CodeChunk(**defaults)


class TestBuildContext:
    def test_includes_header_and_line_range(self):
        ctx = _build_context([make_chunk()])
        assert "# function: foo  (mod.py:1-2)" in ctx
        assert "def foo():" in ctx

    def test_includes_docstring_when_present(self):
        ctx = _build_context([make_chunk(docstring="does a thing")])
        assert "# docstring: does a thing" in ctx

    def test_joins_multiple_chunks(self):
        ctx = _build_context([make_chunk(name="a"), make_chunk(name="b")])
        assert ctx.count("# function:") == 2


class TestParseClaims:
    def test_parses_valid_claim(self):
        claims = _parse_claims([{
            "claim": "validates input",
            "source_chunk": "mod.py",
            "source_function": "foo",
            "lines": "1-2",
            "confidence": "high",
        }])
        assert len(claims) == 1
        assert claims[0].confidence == "high"

    def test_defaults_missing_fields(self):
        claims = _parse_claims([{}])
        assert claims[0].claim == ""
        assert claims[0].confidence == "medium"

    def test_normalizes_invalid_confidence(self):
        claims = _parse_claims([{"confidence": "very sure"}])
        assert claims[0].confidence == "medium"


class TestGenerate:
    def test_no_api_key_returns_failure_message(self):
        result = generate("what does foo do?", [make_chunk()], api_key="")
        assert "no API key" in result.answer

    def test_parses_well_formed_llm_response(self):
        raw = json.dumps({
            "answer": "foo returns 42",
            "claims": [{"claim": "returns 42", "source_chunk": "mod.py",
                        "source_function": "foo", "lines": "2", "confidence": "high"}],
            "unanswered_parts": "",
        })
        with patch("search.generator._llm_complete", return_value=raw):
            result = generate("what does foo do?", [make_chunk()], api_key="key")
        assert isinstance(result, GenerationResult)
        assert result.answer == "foo returns 42"
        assert result.claims[0].source_function == "foo"

    def test_non_json_response_falls_back_to_raw_text(self):
        with patch("search.generator._llm_complete", return_value="not json"):
            result = generate("q", [make_chunk()], api_key="key")
        assert result.answer == "not json"
        assert result.claims == []

    def test_llm_exception_is_caught(self):
        with patch("search.generator._llm_complete", side_effect=RuntimeError("boom")):
            result = generate("q", [make_chunk()], api_key="key")
        assert "generation failed" in result.answer
