"""LLM client for OpenRouter."""

import os
import json
import httpx

from core import web_search

# Shared HTTP client - reused for all calls (connection pooling)
# Static singleton pattern per Dean/Ghemawat: avoids allocation + TLS handshake per call
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Get or create shared HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=120.0)
    return _http_client


async def generate_search_queries(user_query: str, num_queries: int = 6) -> list[str]:
    """Generate multiple arxiv search queries from a user's research topic.

    Searches web in parallel with prompt preparation (Dean/Ghemawat: defer/parallelize).
    """
    import asyncio

    # Run web search in thread pool (non-blocking)
    loop = asyncio.get_event_loop()
    web_task = loop.run_in_executor(None, web_search.search_summary, user_query, 5)

    # Await web search result (was running in background)
    web_context = await web_task

    prompt = f"""You are a research assistant helping find arxiv papers.

User's search: {user_query}

{f"Web search results for context:{chr(10)}{web_context}" if web_context else ""}

Generate {num_queries} arxiv search queries. Mix BOTH approaches:

SPECIFIC (from web results):
- Extract EXACT paper titles mentioned (e.g., "DeepSeek-V3.2")
- Extract EXACT technical terms (e.g., "DeepSeek Sparse Attention" not "Dense Stereo Attention")
- Include model names, version numbers, author names from web results

SEMANTIC (your knowledge):
- Add related concepts and techniques
- Include broader research areas
- Add synonyms and alternative phrasings

Return ONLY a JSON array of strings. Example:
["exact paper title from web", "exact technique from web", "related concept", "broader area"]"""

    response = await complete(
        [{"role": "user", "content": prompt}],
        temperature=0.5
    )

    # Parse JSON from response
    try:
        # Handle markdown code blocks
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        queries = json.loads(text)
        if isinstance(queries, list):
            return queries[:num_queries]
    except (json.JSONDecodeError, IndexError):
        pass

    # Fallback: just use original query
    return [user_query]


async def complete(messages: list[dict], temperature: float = 0.3) -> str:
    """Call OpenRouter chat completions API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")

    client = _get_client()
    response = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
