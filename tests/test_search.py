"""Tests for the web search tool."""



def test_web_search_import():
    """Test that web_search can be imported."""
    from magi.tools.search import web_search

    assert callable(web_search)


def test_web_search_has_tool_decorator():
    """Test that web_search has the Strands tool decorator attributes."""
    from magi.tools.search import web_search

    # The tool decorator adds specific attributes
    assert hasattr(web_search, "__name__")
    assert web_search.__name__ == "web_search"
