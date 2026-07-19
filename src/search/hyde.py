from search.utils.llm import _llm_complete

SYSTEM_PROMPT = (
    "You are a code documentation generator. Given a user query about code, "
    "write a concise docstring for a hypothetical code module that would answer "
    "the query. Describe what the module does, key functions, and how they relate. "
    "Respond with only the docstring — no code, no explanation."
)


def generate_hypothetical_docstring(
    query: str,
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    timeout: int = 10,
) -> str:
    try:
        return _llm_complete(SYSTEM_PROMPT, query, api_key, model, base_url, timeout)
    except Exception:
        return ""
