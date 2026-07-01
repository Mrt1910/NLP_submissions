import feedparser
import csv
import os
import re
from html import unescape
from datetime import datetime

# Used Google News' search-as-RSS trick: Google News lets you turn ANY
# search query into an RSS feed of matching articles. This is far more
# reliable than general tech feeds, because it's already filtered to
# articles that actually mention the search terms.
#
# URL pattern: https://news.google.com/rss/search?q=YOUR+QUERY&hl=en-US&gl=US&ceid=US:en

RSS_FEEDS = [
    # Direct SAP news
    "https://news.google.com/rss/search?q=SAP&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%22SAP+SE%22+software&hl=en-US&gl=US&ceid=US:en",

    # Competitor coverage (useful for "what are competitors doing")
    "https://news.google.com/rss/search?q=Oracle+ERP&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Workday+software&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Microsoft+Dynamics+365&hl=en-US&gl=US&ceid=US:en",
    

    # General enterprise software trends (useful for "trends to monitor")
    "https://news.google.com/rss/search?q=enterprise+AI+software&hl=en-US&gl=US&ceid=US:en",
]

# Since these feeds are already search-filtered we just confirm "SAP" OR a known
# competitor name appears somewhere, mainly to drop totally unrelated
# results that occasionally slip into a search feed.
RELEVANT_TERMS = ["sap", "oracle", "workday", "microsoft dynamics", "erp", "enterprise software"]

OUTPUT_PATH = "data/raw/news_articles.csv"


# Format the html
def strip_html(raw_html):
    text_without_tags = re.sub(r"<[^>]+>", " ", raw_html)
    text_with_real_characters = unescape(text_without_tags)

    # Collapse any repeated whitespace (multiple spaces/newlines) into one space
    cleaned = re.sub(r"\s+", " ", text_with_real_characters).strip()
    return cleaned

# Get headline and publisher
def split_title_and_publisher(raw_title):
    if " - " in raw_title:
        headline, publisher = raw_title.rsplit(" - ", 1)
        return headline.strip(), publisher.strip()
    return raw_title.strip(), ""


# Filter for malformed headlines (e.g. image as a headline)
def is_real_headline(headline, min_words=3):
    word_count = len(headline.split())
    return word_count >= min_words



def fetch_articles_from_feed(feed_url):
    print(f"Fetching: {feed_url}")
    feed = feedparser.parse(feed_url)

    matching_articles = []

    # Each "entry" in the feed is one article
    for entry in feed.entries:
        raw_title = entry.get("title", "")
        raw_summary = entry.get("summary", "")

        # Combine title + summary into one string to search through.
        combined_text = (raw_title + " " + raw_summary).lower()

        if any(term in combined_text for term in RELEVANT_TERMS):
            # Google News title example: "Headline - Publisher Name".
            headline, publisher = split_title_and_publisher(raw_title)

            if not is_real_headline(headline):
                continue

            # The "summary" field from Google News is just an HTML-wrapped repeat of the headline, therefore it is removed.
            cleaned_summary = strip_html(raw_summary)

            article = {
                "source": feed_url,
                "publisher": publisher,
                "title": headline,
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "summary": cleaned_summary,
                "collected_at": datetime.now().isoformat(),
            }
            matching_articles.append(article)

    print(f"  -> Found {len(matching_articles)} matching articles")
    return matching_articles


def save_to_csv(articles, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    file_already_exists = os.path.exists(output_path)

    with open(output_path, mode="a", newline="", encoding="utf-8") as f:
        fieldnames = ["source", "publisher", "title", "link", "published", "summary", "collected_at"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_already_exists:
            writer.writeheader()

        writer.writerows(articles)


def main():
    all_articles = []

    for feed_url in RSS_FEEDS:
        articles = fetch_articles_from_feed(feed_url)
        all_articles.extend(articles)

    save_to_csv(all_articles, OUTPUT_PATH)
    print(f"\nDone. Saved {len(all_articles)} articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()