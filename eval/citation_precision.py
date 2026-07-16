"""Citation precision metric for AlphaAgents research notes.

Measures what fraction of sentences in the full_text contain at least one
URL from the note's citations list.
"""

from __future__ import annotations

import re

from graph.state import ResearchNote


def citation_precision(note: ResearchNote) -> float:
    """Calculate the fraction of sentences in full_text that contain a citation URL.

    A sentence is considered cited if it contains at least one URL that appears
    in note.citations. Blank lines and markdown headings (lines starting with #)
    are excluded from the sentence count.

    Args:
        note: A ResearchNote with full_text and citations fields populated.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if there are no scoreable sentences
        or no citations.
    """
    if not note.citations or not note.full_text:
        return 0.0

    lines = note.full_text.splitlines()
    scoreable = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    if not scoreable:
        return 0.0

    cited_count = 0
    for line in scoreable:
        for url in note.citations:
            if url in line:
                cited_count += 1
                break

    return cited_count / len(scoreable)


def extract_urls_from_text(text: str) -> list[str]:
    """Extract all URLs from a markdown text string.

    Matches both bare URLs and markdown link syntax [text](url).

    Args:
        text: Markdown text to search.

    Returns:
        List of unique URL strings found in the text.
    """
    pattern = re.compile(
        r"\[.*?\]\((https?://[^\s)]+)\)|"
        r"(?<!\()(https?://[^\s\)]+)"
    )
    matches = pattern.findall(text)
    urls: list[str] = []
    seen: set[str] = set()
    for match in matches:
        url = match[0] if match[0] else match[1]
        if url and url not in seen:
            urls.append(url)
            seen.add(url)
    return urls
