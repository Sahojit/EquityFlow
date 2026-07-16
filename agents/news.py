"""News agent — fetches recent news articles and extracts sentiment and summaries.

Uses Tavily with a news topic filter. All articles are analysed in a single batched
LLM call instead of one call per article, cutting latency from O(n) to O(1).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

from pydantic import BaseModel, Field
from tavily import TavilyClient

from graph.state import NewsResult, ResearchState
from llm.client import call_structured, call_with_backoff
from llm.tracing import create_span

logger = logging.getLogger(__name__)

MAX_NEWS_ITEMS = 6


class ArticleAnalysis(BaseModel):
    """Sentiment and summary for a single article in the batch response."""

    sentiment: str = Field(description="One of: 'positive', 'neutral', 'negative'.")
    summary: str = Field(description="One-sentence summary of the article's key point.")


class NewsAnalysisBatch(BaseModel):
    """Wrapper returned by the single batched LLM call."""

    articles: list[ArticleAnalysis] = Field(
        description="One entry per input article, in the same order."
    )


_NEWS_SYSTEM_PROMPT = """\
You are a financial news analyst. You will receive a numbered list of news article
headlines and snippets. For each article, determine the sentiment toward the company
and write a one-sentence summary.

Respond with ONLY valid JSON matching this schema — one entry per article, in order:
{
  "articles": [
    {"sentiment": "positive" | "neutral" | "negative", "summary": "One sentence."},
    ...
  ]
}

Do not include any text outside the JSON object.
"""


def _extract_company_name(query: str) -> str:
    """Return the first four words of the query as a Tavily search term.

    Args:
        query: Original user research query.

    Returns:
        Short string suitable for Tavily search.
    """
    return " ".join(query.split()[:4])


def _analyse_articles_batch(raw_results: list[dict]) -> list[ArticleAnalysis]:
    """Analyse all articles in a single LLM call.

    Formats articles as a numbered list and asks the LLM to return one
    sentiment+summary per article in order. Falls back to an empty list on
    any LLM failure so the pipeline continues with partial data.

    Args:
        raw_results: List of Tavily result dicts with 'title' and 'content' keys.

    Returns:
        List of ArticleAnalysis objects, one per input article (may be shorter
        than raw_results if the LLM omits entries).
    """
    numbered = "\n\n".join(
        f"{i + 1}. Headline: {r.get('title', '')}\n   Snippet: {r.get('content', '')[:300]}"
        for i, r in enumerate(raw_results)
    )
    messages = [
        {"role": "system", "content": _NEWS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyse these {len(raw_results)} articles:\n\n{numbered}"},
    ]
    try:
        batch: NewsAnalysisBatch = call_with_backoff(
            call_structured,
            messages=messages,
            response_schema=NewsAnalysisBatch,
        )
        return batch.articles
    except Exception as exc:
        logger.warning("Batch news analysis failed: %s. Returning empty list.", exc)
        return []


def _parse_published_at(raw: dict) -> datetime:
    """Parse Tavily's published_date field, defaulting to now on failure.

    Args:
        raw: Single Tavily result dict.

    Returns:
        Timezone-aware datetime.
    """
    published_str: str | None = raw.get("published_date")
    if published_str:
        try:
            return datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(UTC)


def news_node(state: ResearchState) -> ResearchState:
    """Fetch recent news and analyse all articles in one batched LLM call.

    Searches Tavily for up to MAX_NEWS_ITEMS news articles, then sends them all
    to the LLM in a single request (vs. one call per article previously). Falls
    back to an empty list if Tavily or the LLM is unavailable.

    Args:
        state: Current pipeline state. Reads ``query`` and ``job_id``.

    Returns:
        Updated state with ``news_results`` populated (may be empty on failure).
    """
    query = state.get("query", "")
    job_id = state.get("job_id", "unknown")

    span = create_span(
        "news_node",
        trace_id=state.get("langfuse_trace_id"),
        input_data={"query": query, "job_id": job_id},
    )

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key:
        warning = "TAVILY_API_KEY not set — news_node returning empty results."
        logger.warning(warning)
        span.update(level="WARNING", status_message=warning, output={"news_results": []})
        span.end()
        return {"news_results": []}

    search_term = _extract_company_name(query)
    logger.info("Searching news for: %s (job=%s)", search_term, job_id)
    client = TavilyClient(api_key=tavily_key)

    try:
        response = client.search(
            query=f"{search_term} news",
            max_results=MAX_NEWS_ITEMS,
            include_answer=False,
            search_depth="basic",
            topic="news",
        )
        raw_results: list[dict] = response.get("results", [])[:MAX_NEWS_ITEMS]
    except Exception as exc:
        warning = f"Tavily news search failed: {exc}. Returning empty news results."
        logger.warning(warning)
        span.update(level="WARNING", status_message=warning, output={"news_results": []})
        span.end()
        return {"news_results": []}

    if not raw_results:
        span.update(output={"article_count": 0})
        span.end()
        return {"news_results": []}

    analyses = _analyse_articles_batch(raw_results)

    news_results: list[NewsResult] = []
    for i, raw in enumerate(raw_results):
        if i >= len(analyses):
            break
        analysis = analyses[i]
        sentiment_raw = analysis.sentiment.lower()
        if sentiment_raw not in ("positive", "neutral", "negative"):
            sentiment_raw = "neutral"

        news_results.append(
            NewsResult(
                headline=raw.get("title", ""),
                source_name=raw.get("source", "Unknown"),
                url=raw.get("url", ""),
                published_at=_parse_published_at(raw),
                sentiment=sentiment_raw,
                summary=analysis.summary,
            )
        )

    span.update(output={"article_count": len(news_results)})
    span.end()
    logger.info("news_node collected %d articles for job %s.", len(news_results), job_id)
    return {"news_results": news_results}
