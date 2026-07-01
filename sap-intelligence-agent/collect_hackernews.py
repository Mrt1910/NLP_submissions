import requests
import csv
import os
import re
import time
from datetime import datetime

# The search endpoint. We will add a `query` parameter for each search term.
SEARCH_URL = "https://hn.algolia.com/api/v1/search"

# Search terms: SAP itself, its main competitors, and broader trends.
# Same logic as collect_news.py
SEARCH_TERMS = [
    "SAP",
    "Oracle ERP",
    "Workday",
    "Microsoft Dynamics 365",
    "enterprise AI software",
]

# Number of results to request per search term
RESULTS_PER_TERM = 50

OUTPUT_PATH = "data/raw/hackernews_posts.csv"

# Is needed because words sucha as Sapphire or Sapling are similar to SAP but not the search term itself
def is_genuinely_relevant(text, term):

    # \\b in a regex means "word boundary" - the edge between a word character and a non-word character.
    # Wrapping the term in \\b...\\b means "term, but only when it stands as its own whole word".

    pattern = r"\b" + re.escape(term) + r"\b"
    return re.search(pattern, text, re.IGNORECASE) is not None

# Sends research request to API and returns POST response
def search_hackernews(query, max_results):
    print(f"Searching Hacker News for: {query}")

    params = {
        "query": query,
        "tags": "story",            
        "hitsPerPage": max_results,
    }

    response = requests.get(SEARCH_URL, params=params, timeout=15)

    if response.status_code != 200:
        print(f"  -> Failed (status code {response.status_code})")
        return []

    data = response.json()
    hits = data.get("hits", [])

    posts = []
    for hit in hits:
        title = hit.get("title", "")

        # Skip results when term only as part of a longer, unrelated word (e.g. "Sapling" matching "SAP").
        if not is_genuinely_relevant(title, query):
            continue

        post = {
            "search_term": query,
            "title": title,
            "link": hit.get("url", "") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
            "hn_discussion_link": f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
            "points": hit.get("points", 0),
            "num_comments": hit.get("num_comments", 0),
            "published": hit.get("created_at", ""),
            "summary": hit.get("story_text", "") or "",
            "collected_at": datetime.now().isoformat(),
        }
        posts.append(post)

    print(f"  -> Found {len(posts)} posts")
    return posts


def save_to_csv(posts, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_already_exists = os.path.exists(output_path)

    with open(output_path, mode="a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "search_term", "title", "link", "hn_discussion_link",
            "points", "num_comments", "published", "summary", "collected_at",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_already_exists:
            writer.writeheader()

        writer.writerows(posts)


def main():
    all_posts = []

    for term in SEARCH_TERMS:
        posts = search_hackernews(term, RESULTS_PER_TERM)
        all_posts.extend(posts)
        time.sleep(1) # delay in between requests

    # Remove duplicate posts. The HN discussion link is used as the unique ID.
    seen_links = set()
    unique_posts = []
    for post in all_posts:
        if post["hn_discussion_link"] not in seen_links:
            seen_links.add(post["hn_discussion_link"])
            unique_posts.append(post)

    save_to_csv(unique_posts, OUTPUT_PATH)
    print(f"\nDone. Saved {len(unique_posts)} unique posts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()