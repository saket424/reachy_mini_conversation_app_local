"""Web search tool using DuckDuckGo."""

import logging
from typing import Any, Dict

from ddgs import DDGS

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class WebSearch(Tool):
    """Search the web for current information using DuckDuckGo."""

    name = "web_search"
    description = (
        "Search the web for current, real-time information. Use this when the user asks about "
        "recent events, news, sports scores, weather, or anything that requires up-to-date information "
        "you don't already know."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web.",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run a DuckDuckGo web search and return top results."""
        query = kwargs.get("query", "")
        if not query:
            return {"error": "No search query provided"}

        logger.info("Tool call: web_search query=%s", query)

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))

            if not results:
                return {"status": "no_results", "query": query, "summary": "No results found."}

            snippets = []
            for r in results:
                snippets.append(f"- {r.get('title', '')}: {r.get('body', '')}")

            summary = "\n".join(snippets)
            logger.info("Web search returned %d results for: %s", len(results), query)
            return {"status": "ok", "query": query, "results": summary}

        except Exception as e:
            logger.error("Web search failed: %s", e)
            return {"error": f"Search failed: {e}"}
