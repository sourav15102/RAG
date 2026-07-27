"""
search_indexer.py

Synthetic search indexing & ranking module: tokenization, inverted index,
TF-IDF style scoring, and a small query DSL. Test fixture for AST-based
code chunking + RAG evaluation.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "in", "on", "at", "to", "of", "for", "with", "by", "this", "that",
})
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
MIN_TOKEN_LENGTH = 2


class IndexError_(Exception):
    """Base exception for indexing failures (named to avoid shadowing builtin)."""
    pass


class DocumentNotFoundError(IndexError_):
    """Raised when a document id is referenced but not present in the index."""
    pass


class QueryParseError(Exception):
    """Raised when a query string cannot be parsed by the query DSL."""
    pass


def tokenize(text: str, stopwords: frozenset = DEFAULT_STOPWORDS) -> list[str]:
    """Lowercase, tokenize, and strip stopwords/short tokens from text."""
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    return [t for t in raw_tokens if t not in stopwords and len(t) >= MIN_TOKEN_LENGTH]


def term_frequencies(tokens: list[str]) -> dict[str, int]:
    """Return a term -> raw count mapping for a list of tokens."""
    return dict(Counter(tokens))


def compute_tf(term_count: int, total_terms: int) -> float:
    """Compute normalized term frequency."""
    if total_terms == 0:
        return 0.0
    return term_count / total_terms


def compute_idf(doc_count_with_term: int, total_docs: int) -> float:
    """Compute inverse document frequency with smoothing."""
    return math.log((1 + total_docs) / (1 + doc_count_with_term)) + 1


def compute_tfidf(tf: float, idf: float) -> float:
    """Combine tf and idf into a single tf-idf weight."""
    return tf * idf


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse term-weight dicts."""
    shared_terms = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in shared_terms)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def ngrams(tokens: list[str], n: int = 2) -> list[str]:
    """Generate n-gram strings from a token list."""
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def highlight_snippet(text: str, query_terms: list[str], window: int = 40) -> str:
    """Return a short excerpt around the first matched query term, with markers."""
    lowered = text.lower()
    first_pos = None
    for term in query_terms:
        pos = lowered.find(term.lower())
        if pos != -1 and (first_pos is None or pos < first_pos):
            first_pos = pos
    if first_pos is None:
        return text[: window * 2] + ("..." if len(text) > window * 2 else "")
    start = max(0, first_pos - window)
    end = min(len(text), first_pos + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of doc ids using Reciprocal Rank Fusion."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def parse_simple_query(query: str) -> dict:
    """Parse a tiny query DSL supporting quoted phrases and -exclusions.

    Example: 'cache invalidation -legacy "exact phrase"' returns:
        {"terms": ["cache", "invalidation"], "excluded": ["legacy"], "phrases": ["exact phrase"]}
    """
    phrases = re.findall(r'"([^"]+)"', query)
    remainder = re.sub(r'"[^"]+"', "", query)
    excluded = []
    terms = []
    for token in remainder.split():
        if token.startswith("-") and len(token) > 1:
            excluded.append(token[1:].lower())
        elif token.strip():
            terms.append(token.lower())
    if not terms and not phrases:
        raise QueryParseError(f"Query has no searchable terms: {query!r}")
    return {"terms": terms, "excluded": excluded, "phrases": phrases}


def legacy_score_boost(base_score, doc_metadata, boost_recent=True, recency_half_life_days=30, category_boosts=None):
    # NOTE: legacy scoring tweak layered on top of tf-idf before the RRF rewrite.
    score = base_score
    if boost_recent and "age_days" in doc_metadata:
        age = doc_metadata["age_days"]
        decay = 0.5 ** (age / recency_half_life_days)
        score = score * (0.5 + 0.5 * decay)
    if category_boosts and "category" in doc_metadata:
        cat = doc_metadata["category"]
        if cat in category_boosts:
            score = score * category_boosts[cat]
    if doc_metadata.get("is_deprecated"):
        score = score * 0.1
    return score


@dataclass
class Document:
    """A single indexed document."""
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def token_count(self) -> int:
        """Return the number of tokens in this document's text."""
        return len(tokenize(self.text))


@dataclass
class SearchResult:
    """A single scored search result."""
    doc_id: str
    score: float
    snippet: str = ""


class InvertedIndex:
    """A simple in-memory inverted index with TF-IDF scoring."""

    def __init__(self):
        self._documents: dict[str, Document] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._term_freqs: dict[str, dict[str, int]] = {}

    def add_document(self, document: Document) -> None:
        """Tokenize and index a document, updating postings lists."""
        tokens = tokenize(document.text)
        self._documents[document.doc_id] = document
        self._term_freqs[document.doc_id] = term_frequencies(tokens)
        for term in set(tokens):
            self._postings[term].add(document.doc_id)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document and clean up its postings entries."""
        if doc_id not in self._documents:
            raise DocumentNotFoundError(f"No such document: {doc_id}")
        del self._documents[doc_id]
        del self._term_freqs[doc_id]
        for term, doc_ids in self._postings.items():
            doc_ids.discard(doc_id)

    def document_count(self) -> int:
        """Return the total number of indexed documents."""
        return len(self._documents)

    def docs_containing(self, term: str) -> set[str]:
        """Return the set of doc ids whose text contains a given term."""
        return self._postings.get(term.lower(), set())

    def _vector_for(self, doc_id: str) -> dict[str, float]:
        term_freqs = self._term_freqs[doc_id]
        total_terms = sum(term_freqs.values())
        total_docs = self.document_count()
        vector = {}
        for term, count in term_freqs.items():
            tf = compute_tf(count, total_terms)
            idf = compute_idf(len(self._postings[term]), total_docs)
            vector[term] = compute_tfidf(tf, idf)
        return vector

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search the index for a query string, returning ranked results."""
        parsed = parse_simple_query(query)
        candidate_ids: set[str] = set()
        for term in parsed["terms"]:
            candidate_ids |= self.docs_containing(term)
        for excluded_term in parsed["excluded"]:
            candidate_ids -= self.docs_containing(excluded_term)
        if not candidate_ids:
            return []
        query_tokens = parsed["terms"]
        query_tf = term_frequencies(query_tokens)
        total_docs = self.document_count()
        query_vector = {
            term: compute_tfidf(
                compute_tf(count, len(query_tokens)),
                compute_idf(len(self._postings.get(term, set())), total_docs),
            )
            for term, count in query_tf.items()
        }
        scored = []
        for doc_id in candidate_ids:
            doc_vector = self._vector_for(doc_id)
            score = cosine_similarity(query_vector, doc_vector)
            snippet = highlight_snippet(self._documents[doc_id].text, query_tokens)
            scored.append(SearchResult(doc_id=doc_id, score=score, snippet=snippet))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def multi_field_search(self, queries_by_ranking_name: dict[str, str], top_k: int = 10) -> list[tuple[str, float]]:
        """Run several named queries and fuse their rankings via RRF."""
        rankings = []
        for _, query in queries_by_ranking_name.items():
            results = self.search(query, top_k=top_k * 2)
            rankings.append([r.doc_id for r in results])
        return reciprocal_rank_fusion(rankings)[:top_k]