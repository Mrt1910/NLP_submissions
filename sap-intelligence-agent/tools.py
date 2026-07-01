"""
tools.py
--------
Tool Usage (Agent Requirement)

What this module does:
Defines the "tools" available to the AI CEO Agent - functions that
produce factual, deterministic signals without calling the LLM. This is
what makes the system an agent rather than a single LLM call: the agent
decides which tools to use, calls them, and then reasons over their
output alongside retrieved evidence.

Two tools are provided:
1. sentiment_tool   - wraps sentiment.py's lexicon-based scoring
2. trend_tool       - counts how often key business/strategy terms
                      appear across a set of documents, surfacing what
                      is actually trending in the evidence

Both tools take a list of document dicts (the same shape returned by
ai_ceo_agent.retrieve_relevant_documents) and return a small, structured
result - no text generation, no randomness, fully reproducible.

This file has no main() / standalone run mode - it's a small library of
functions meant to be called by agent.py.
"""

import re
from collections import Counter

import sentiment


# TOOL 1: SENTIMENT TOOL
#
# This is a thin wrapper around sentiment.py, exposed under a name and
# shape the agent's tool-calling logic expects. The scoring logic isn't duplicated here - sentiment.py remains the single source of truth
# for how sentiment is calculated, which also still backs the dashboard's standalone Sentiment Analysis section.

def sentiment_tool(documents):
    """
    Tool: analyzes the sentiment of a set of documents.

    Input: list of document dicts (each with a "text" key)
    Output: a dict with average_score, overall_label, and counts -
    a factual signal the agent can reason over, not generated text.
    """
    result = sentiment.analyze_documents(documents)
    return {
        "tool": "sentiment_tool",
        "average_score": round(result["average_score"], 2),
        "overall_label": result["overall_label"],
        "positive_count": result["positive_count"],
        "negative_count": result["negative_count"],
        "neutral_count": result["neutral_count"],
    }


# TOOL 2: TREND / FREQUENCY TOOL
#
# Counts how often a fixed set of business/strategy terms appears
# across the retrieved documents. This gives the agent a simple,
# explainable, non-LLM signal about what topics are actually showing up
# most often in the evidence - useful input for deciding what to
# prioritize, independent of any one document's wording.

# Terms that are tracked
TREND_TERMS = [
    "ai", "artificial intelligence", "cloud", "sustainability",
    "partnership", "acquisition", "layoff", "lawsuit", "regulation",
    "antitrust", "security", "vulnerability", "revenue", "growth",
    "competition", "innovation", "automation", "data", "compliance",
]


def trend_tool(documents, top_n=5):
    """
    Tool: counts how often each tracked term appears across a set of
    documents, and returns the top_n most frequent terms.

    Input: list of document dicts (each with a "text" key)
    Output: a dict with the ranked term frequencies - a factual count,
    not an LLM-generated summary.

    How it works: combine all document text into one block, lowercase
    it, then for each tracked term count how many times it appears
    (using word boundaries so "ai" doesn't match inside "said" or
    "main", the same regex technique used in collect_hackernews.py's
    relevance filter).
    """
    combined_text = " ".join(doc.get("text", "") for doc in documents).lower()

    term_counts = Counter()
    for term in TREND_TERMS:
        pattern = r"\b" + re.escape(term) + r"\b"
        count = len(re.findall(pattern, combined_text))
        if count > 0:
            term_counts[term] = count

    top_terms = term_counts.most_common(top_n)

    return {
        "tool": "trend_tool",
        "top_terms": [{"term": term, "count": count} for term, count in top_terms],
        "documents_analyzed": len(documents),
    }



# TOOL REGISTRY
AVAILABLE_TOOLS = {
    "sentiment_tool": sentiment_tool,
    "trend_tool": trend_tool,
}


def run_tool(tool_name, documents):
    """
    Looks up and runs a tool by name. Returns None (rather than raising)
    if an unknown tool name is requested, so a planning mistake by the
    LLM doesn't crash the whole agent run.
    """
    tool_function = AVAILABLE_TOOLS.get(tool_name)
    if tool_function is None:
        return None
    return tool_function(documents)