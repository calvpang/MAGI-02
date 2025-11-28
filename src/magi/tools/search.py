"""Web search tool for MAGI-02.

This module provides web search capability using DuckDuckGo.
"""

from strands import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information using DuckDuckGo.

    This tool performs a web search and returns relevant results
    with titles, URLs, and snippets.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)

    Returns:
        A formatted string containing search results
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Error: duckduckgo-search is not installed. Install with: pip install duckduckgo-search"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No search results found for: {query}"

        # Format results
        output_parts = [f"## Web Search Results for: {query}\n"]

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("href", "No URL")
            snippet = result.get("body", "No description available")

            output_parts.append(f"### {i}. {title}")
            output_parts.append(f"**URL:** {url}")
            output_parts.append(f"{snippet}\n")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Error performing web search: {str(e)}"
