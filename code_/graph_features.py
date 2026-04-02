import pandas as pd
import numpy as np
import re

SENSATIONAL_WORDS = {
    "breaking", "shocking", "urgent", "exclusive", "confirmed", "boom",
    "dead", "killed", "dies", "died", "death", "murder", "arrested",
    "scandal", "exposed", "leaked", "secret", "conspiracy", "hoax",
    "fake", "fraud", "scam", "attack", "war", "bomb", "explosion",
    "crash", "destroyed", "emergency", "alert", "warning", "omg",
    "unbelievable", "incredible", "horrifying", "terrible", "shocking",
}

CREDIBILITY_WORDS = {
    "according", "source", "official", "statement", "reported", "confirmed",
    "evidence", "study", "research", "data", "analysis", "investigation",
}


def compute_text_intelligence(df):
    """NLP-based features that capture misinformation signals."""
    texts = df["text"].astype(str)

    df["exclamation_density"] = texts.apply(lambda t: t.count("!") / max(len(t), 1))
    df["question_density"] = texts.apply(lambda t: t.count("?") / max(len(t), 1))

    df["has_url"] = texts.apply(lambda t: int(bool(re.search(r"https?://", t))))
    df["has_mention"] = texts.apply(lambda t: int("@" in t))

    df["caps_word_count"] = texts.apply(
        lambda t: sum(1 for w in t.split() if w.isupper() and len(w) > 1)
    )
    df["caps_ratio"] = df["caps_word_count"] / df["word_count"].clip(lower=1)

    df["unique_word_ratio"] = texts.apply(
        lambda t: len(set(t.lower().split())) / max(len(t.split()), 1)
    )

    df["sensational_count"] = texts.apply(
        lambda t: len(set(t.lower().split()) & SENSATIONAL_WORDS)
    )
    df["credibility_count"] = texts.apply(
        lambda t: len(set(t.lower().split()) & CREDIBILITY_WORDS)
    )

    df["special_char_ratio"] = texts.apply(
        lambda t: sum(not c.isalnum() and not c.isspace() for c in t) / max(len(t), 1)
    )

    df["repeated_punct"] = texts.apply(
        lambda t: len(re.findall(r"[!?]{2,}", t))
    )

    return df


def compute_structural_features(df):
    """Features derived from tweet structure rather than fake propagation data."""
    texts = df["text"].astype(str)

    df["url_count"] = texts.apply(lambda t: len(re.findall(r"https?://\S+", t)))
    df["hashtag_word_ratio"] = df["hashtags"] / df["word_count"].clip(lower=1)
    df["mention_word_ratio"] = df["mentions"] / df["word_count"].clip(lower=1)

    df["text_complexity"] = texts.apply(
        lambda t: np.std([len(w) for w in t.split()]) if len(t.split()) > 1 else 0
    )

    df["digit_ratio"] = df["digit_count"] / df["length"].clip(lower=1)

    return df
