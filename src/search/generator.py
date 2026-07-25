import json
import os
from dataclasses import dataclass, field
from typing import Optional

from code_chunker.ast_chunker import CodeChunk
from search.utils.llm import _llm_complete


@dataclass
class Claim:
    claim: str
    source_chunk: str
    source_function: str
    lines: str
    confidence: str  # "high" | "medium" | "low"


@dataclass
class GenerationResult:
    answer: str
    claims: list[Claim] = field(default_factory=list)
    unanswered_parts: str = ""


SYSTEM_PROMPT = (
    "You are a code assistant answering questions about a specific codebase.\n\n"
    "STRICT RULES:\n"
    "1. Only use information explicitly present in the provided code chunks.\n"
    '2. If the answer is not in the chunks, say exactly: '
    '"I don\'t have enough information in the retrieved code to answer this."\n'
    "3. Never generalize beyond what a chunk explicitly states.\n"
    "4. Never infer connections between chunks — if two chunks are related, "
    "a chunk must explicitly show that relationship (a function call, "
    "an import, a reference).\n"
    "5. For every claim you make, you must cite the specific chunk "
    "it came from.\n\n"
    "RESPONSE FORMAT:\n"
    "Return a JSON object with this exact structure:\n"
    "{\n"
    '  "answer": "process_payment validates that amount > 0 before charging",\n'
    '  "claims": [\n'
    "    {\n"
    '      "claim": "validates amount > 0",\n'
    '      "source_chunk": "payments/service.py",\n'
    '      "source_function": "PaymentService.process",\n'
    '      "lines": "45-48",\n'
    '      "confidence": "high"\n'
    "    }\n"
    "  ],\n"
    '  "unanswered_parts": "How the charge gateway is selected"\n'
    "}\n\n"
    "Confidence values:\n"
    '  "high"   = directly stated in the chunk\n'
    '  "medium" = strongly implied by the chunk\n'
    '  "low"    = inferred or uncertain\n\n'
    "Respond ONLY with valid JSON."
)


def _build_context(chunks: list[CodeChunk]) -> str:
    parts = []
    for chunk in chunks:
        line_range = f"{chunk.start_line}-{chunk.end_line}"
        header = (
            f"# {chunk.chunk_type}: {chunk.name}"
            f"  ({chunk.file_path}:{line_range})"
        )
        if chunk.docstring:
            header += f"\n# docstring: {chunk.docstring}"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n".join(parts)


def _parse_claims(raw_claims: list[dict]) -> list[Claim]:
    parsed = []
    for c in raw_claims:
        conf = c.get("confidence", "medium")
        if conf not in ("high", "medium", "low"):
            conf = "medium"
        parsed.append(Claim(
            claim=c.get("claim", ""),
            source_chunk=c.get("source_chunk", ""),
            source_function=c.get("source_function", ""),
            lines=c.get("lines", ""),
            confidence=conf,
        ))
    return parsed


def generate(
    query: str,
    chunks: list[CodeChunk],
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    timeout: int = 30,
) -> GenerationResult:
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return GenerationResult(answer="[generation failed: no API key]")

    context = _build_context(chunks)
    user_msg = f"Question: {query}\n\nRelevant code:\n{context}"

    try:
        raw = _llm_complete(
            SYSTEM_PROMPT, user_msg, api_key, model, base_url, timeout,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        return GenerationResult(answer=f"[generation failed: {exc}]")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return GenerationResult(answer=raw)

    return GenerationResult(
        answer=parsed.get("answer", ""),
        claims=_parse_claims(parsed.get("claims", [])),
        unanswered_parts=parsed.get("unanswered_parts", ""),
    )
