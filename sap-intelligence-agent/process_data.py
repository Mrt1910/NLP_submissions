import pandas as pd
import re
from html import unescape
import chromadb
from sentence_transformers import SentenceTransformer

NEWS_CSV = "data/raw/news_articles.csv"
COMPANY_CSV = "data/raw/company_press_releases.csv"
HACKERNEWS_CSV = "data/raw/hackernews_posts.csv"

# Where ChromaDB will store its data on disk (so it persists between runs)
CHROMA_PATH = "data/processed/chroma_db"

# The name of the "collection" inside ChromaDB
COLLECTION_NAME = "sap_intelligence"

# The embedding model. This runs locally via sentence-transformers.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


# LOAD - read each CSV and turn it into a list of plain Python dictionaries (Normalization step)

def load_news():
    """Loads news_articles.csv and normalizes it to our shared structure."""
    df = pd.read_csv(NEWS_CSV)
    documents = []
    for _, row in df.iterrows():
        documents.append({
            "source_type": "news",
            "title": str(row.get("title", "")),
            "text": str(row.get("summary", "")),
            "link": str(row.get("link", "")),
            "published": str(row.get("published", "")),
            "extra_info": str(row.get("publisher", "")),
        })
    return documents


def load_company():
    """Loads company_press_releases.csv and normalizes it."""
    df = pd.read_csv(COMPANY_CSV)
    documents = []
    for _, row in df.iterrows():
        documents.append({
            "source_type": "company",
            "title": str(row.get("title", "")),
            "text": str(row.get("summary", "")),
            "link": str(row.get("link", "")),
            "published": str(row.get("published", "")),
            "extra_info": "SAP Newsroom",
        })
    return documents


def load_hackernews():
    """Loads hackernews_posts.csv and normalizes it."""
    df = pd.read_csv(HACKERNEWS_CSV)
    documents = []
    for _, row in df.iterrows():
        # Hacker News posts often have an empty "summary", so the title is used as the main text if there's nothing else.
        summary = str(row.get("summary", "")) if pd.notna(row.get("summary")) else ""
        text = summary if summary.strip() else str(row.get("title", ""))

        documents.append({
            "source_type": "hackernews",
            "title": str(row.get("title", "")),
            "text": text,
            "link": str(row.get("link", "")),
            "published": str(row.get("published", "")),
            "extra_info": f"{row.get('points', 0)} points, {row.get('num_comments', 0)} comments",
        })
    return documents






# CLEAN - remove any leftover HTML, normalize whitespace, and drop documents that don't actually have usable text.

def clean_text(text):
    if not text or text == "nan":
        return ""
    text_without_tags = re.sub(r"<[^>]+>", " ", text)
    text_with_real_characters = unescape(text_without_tags)
    cleaned = re.sub(r"\s+", " ", text_with_real_characters).strip()
    return cleaned


def clean_documents(documents):
    cleaned = []
    for doc in documents:
        doc["title"] = clean_text(doc["title"])
        doc["text"] = clean_text(doc["text"])

        # A document needs at least a title to be worth keeping.
        if doc["title"]:
            cleaned.append(doc)

    return cleaned


# DEDUPLICATE - the same story can appear more than once, both within one source (e.g. two Google News queries both finding it) and across sources (e.g. SAP's own newsroom AND a news article about the same press release). 
# Deduplication happens by comparing cleaned titles.

def deduplicate_documents(documents):
    seen_titles = set()
    unique_documents = []

    for doc in documents:
        # .lower() makes comparison case-insensitive, so "SAP News" and "sap news" are treated as the same document.
        title_key = doc["title"].lower().strip()

        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_documents.append(doc)

    return unique_documents




# EMBED + STORE - turn each document's text into a vector, and save everything into ChromaDB.

def build_document_text(doc):
    if doc["text"] and doc["text"].lower() != doc["title"].lower():
        return f"{doc['title']}. {doc['text']}"
    return doc["title"]


# generates embedding for each document
def embed_and_store(documents):
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print("(First run downloads the model - this may take a minute.)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts_to_embed = [build_document_text(doc) for doc in documents]

    print(f"Generating embeddings for {len(texts_to_embed)} documents...")

    embeddings = model.encode(texts_to_embed, show_progress_bar=True)

    # Connect to (or create) the ChromaDB database, stored on disk at CHROMA_PATH so it persists between script runs.
    print(f"\nConnecting to ChromaDB at {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get_or_create_collection: if "sap_intelligence" already exists from a previous run, we delete and recreate it
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # ChromaDB needs: a unique id, the embedding vector, the document text itself, and a "metadata" dictionary of extra structured info for each entry.
    ids = [f"doc_{i}" for i in range(len(documents))]
    metadatas = [
        {
            "source_type": doc["source_type"],
            "title": doc["title"],
            "link": doc["link"],
            "published": doc["published"],
            "extra_info": doc["extra_info"],
        }
        for doc in documents
    ]

    print("Saving to ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),  # ChromaDB wants plain lists, not numpy arrays
        documents=texts_to_embed,
        metadatas=metadatas,
    )

    print(f"Done. Stored {collection.count()} documents in ChromaDB.")
    return collection






# MAIN

def main():
    print("Loading raw data")
    all_documents = load_news() + load_company() + load_hackernews()
    print(f"Loaded {len(all_documents)} raw documents")

    print("\n Cleaning")
    all_documents = clean_documents(all_documents)
    print(f"{len(all_documents)} documents remain after cleaning")

    print("\n Deduplicating")
    all_documents = deduplicate_documents(all_documents)
    print(f"{len(all_documents)} unique documents remain after deduplication")

    print("\n Embedding + storing")
    embed_and_store(all_documents)


if __name__ == "__main__":
    main()