import feedparser
import csv
import os
from datetime import datetime

# SAP publishes official RSS feeds at predictable URLs. The main feed
# covers all news center content (press releases + feature articles).
# Topic-specific feeds follow the pattern:
#   https://news.sap.com/topics/<topic-name>/feed/


RSS_FEEDS = [
    "https://news.sap.com/feed/",                          
    "https://news.sap.com/topics/artificial-intelligence/feed/",
    "https://news.sap.com/topics/cloud/feed/",
    "https://news.sap.com/topics/sustainability/feed/",
]

OUTPUT_PATH = "data/raw/company_press_releases.csv"

# Downloads a RSS feed and returns a list of articles as dictionaries
def fetch_articles_from_feed(feed_url):
    print(f"Fetching: {feed_url}")
    feed = feedparser.parse(feed_url)

    if feed.bozo:
        print(f"  WARNING: could not parse this feed cleanly: {feed.bozo_exception}")

    articles = []

    for entry in feed.entries:
        article = {
            "source": feed_url,
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "summary": entry.get("summary", ""),
            "collected_at": datetime.now().isoformat(),
        }
        articles.append(article)

    print(f"  -> Found {len(articles)} articles")
    return articles


def save_to_csv(articles, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_already_exists = os.path.exists(output_path)

    with open(output_path, mode="a", newline="", encoding="utf-8") as f:
        fieldnames = ["source", "title", "link", "published", "summary", "collected_at"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_already_exists:
            writer.writeheader()

        writer.writerows(articles)


def main():
    all_articles = []

    for feed_url in RSS_FEEDS:
        articles = fetch_articles_from_feed(feed_url)
        all_articles.extend(articles)

    # Some articles may appear in more than one feed. Exact duplicate links are removed here.
    seen_links = set()
    unique_articles = []
    for article in all_articles:
        if article["link"] not in seen_links:
            seen_links.add(article["link"])
            unique_articles.append(article)

    save_to_csv(unique_articles, OUTPUT_PATH)
    print(f"\nDone. Saved {len(unique_articles)} unique articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()