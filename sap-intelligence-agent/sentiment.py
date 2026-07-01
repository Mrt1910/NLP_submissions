import re

# Word lists
POSITIVE_WORDS = [
    "growth", "success", "successful", "strong", "innovative", "innovation",
    "opportunity", "opportunities", "partnership", "win", "wins", "winning",
    "leader", "leading", "leadership", "improve", "improved", "improving",
    "improvement", "gain", "gains", "profit", "profitable", "expand",
    "expansion", "boost", "boosts", "breakthrough", "advance", "advances",
    "advancing", "achievement", "milestone", "record", "outperform",
    "exceed", "exceeds", "exceeded", "positive", "robust", "resilient",
    "thrive", "thriving", "upgrade", "upgraded", "best",
]

NEGATIVE_WORDS = [
    "risk", "risks", "decline", "declines", "declining", "loss", "losses",
    "lawsuit", "litigation", "concern", "concerns", "fail", "fails",
    "failure", "failed", "weak", "weakness", "threat", "threats",
    "vulnerability", "vulnerabilities", "breach", "breaches", "scandal",
    "investigation", "fine", "fines", "penalty", "penalties", "cut", "cuts",
    "layoff", "layoffs", "downturn", "drop", "drops", "falling", "fell",
    "fall", "antitrust", "violation", "violations", "warning", "warns",
    "recall", "delay", "delays", "delayed", "miss", "missed", "misses",
    "worst", "crisis", "controversy", "criticism", "criticized",
]


def calculate_sentiment_score(text):
    """
    Calculates a sentiment score for a piece of text, from -1 (very negative) to +1 (very positive). Returns 0 for completely neutral text (no sentiment words found at all).
    
    Count how many positive words and negative words appear in the text
    - Formula:
        score = (positive_count - negative_count) / (positive_count + negative_count)
    """

    # Lowercase and split into individual words. \b\w+\b matches sequences of word characters (letters/digits), which correctly ignores punctuation like commas and periods.
    words = re.findall(r"\b\w+\b", text.lower())

    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)

    total_sentiment_words = positive_count + negative_count

    if total_sentiment_words == 0:
        return 0.0

    score = (positive_count - negative_count) / total_sentiment_words
    return score


def classify_sentiment(score):
    """
    Converts a numeric score into a human-readable label.
    Small thresholds around 0 were used (rather than exactly 0) so that very weakly positive/negative text still counts as "Neutral" -
    this avoids the label flipping on a single stray word.
    """
    if score > 0.15:
        return "Positive"
    elif score < -0.15:
        return "Negative"
    else:
        return "Neutral"

# Scores and returns multiple document scores
def analyze_documents(documents):
    if not documents:
        return {
            "average_score": 0.0,
            "overall_label": "Neutral",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "scored_documents": [],
        }

    scored_documents = []
    for doc in documents:
        score = calculate_sentiment_score(doc.get("text", ""))
        label = classify_sentiment(score)
        scored_documents.append({**doc, "sentiment_score": score, "sentiment_label": label})

    average_score = sum(d["sentiment_score"] for d in scored_documents) / len(scored_documents)

    positive_count = sum(1 for d in scored_documents if d["sentiment_label"] == "Positive")
    negative_count = sum(1 for d in scored_documents if d["sentiment_label"] == "Negative")
    neutral_count = sum(1 for d in scored_documents if d["sentiment_label"] == "Neutral")

    return {
        "average_score": average_score,
        "overall_label": classify_sentiment(average_score),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "scored_documents": scored_documents,
    }