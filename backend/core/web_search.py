"""Web search using DuckDuckGo."""

from duckduckgo_search import DDGS


def search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return results.

    Returns list of dicts with 'title', 'href', 'body' keys.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        return results
    except Exception as e:
        print(f"Web search failed: {e}")
        return []


def search_summary(query: str, max_results: int = 5) -> str:
    """Search and return a text summary of results."""
    results = search(query, max_results)
    if not results:
        return ""

    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        lines.append(f"- {title}: {body}")

    return "\n".join(lines)
