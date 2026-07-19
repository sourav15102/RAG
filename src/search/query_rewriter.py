from search.utils.llm import _llm_complete

SYSTEM_PROMPT = (
    "You are a query rewriting assistant for a code search engine. "
    "Rewrite the user's query to be more precise for finding relevant source code. "
    "Expand abbreviations, add technical keywords, and include synonyms. "
    "Respond with only the rewritten query — no explanation."
)


def rewrite_query(
    query: str,
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    timeout: int = 10,
) -> str:
    try:
        return _llm_complete(SYSTEM_PROMPT, query, api_key, model, base_url, timeout)
    except Exception:
        return query
