import pandas as pd
import re

from graph_features import compute_text_intelligence, compute_structural_features

# ---------------- LOAD BOTH DATASETS ---------------- #

t15 = pd.read_csv("../data_/rumour_detection/twitter15/source_tweets.txt",
                   sep="\t", names=["id", "text"], dtype={"id": str})
l15 = pd.read_csv("../data_/rumour_detection/twitter15/label.txt",
                   sep=":", names=["label", "id"], dtype={"id": str})
df15 = t15.merge(l15, on="id")
df15["source"] = "twitter15"

t16 = pd.read_csv("../data_/rumour_detection/twitter16/source_tweets.txt",
                   sep="\t", names=["id", "text"], dtype={"id": str})
l16 = pd.read_csv("../data_/rumour_detection/twitter16/label.txt",
                   sep=":", names=["label", "id"], dtype={"id": str})
df16 = t16.merge(l16, on="id")
df16["source"] = "twitter16"

df = pd.concat([df15, df16], ignore_index=True)
print(f"Combined dataset: {len(df)} tweets")
print(df["label"].value_counts())


# ---------------- BASIC TEXT FEATURES ---------------- #

def get_features(text):
    text = str(text)

    length = len(text)
    hashtags = text.count("#")
    mentions = text.count("@")
    urls = len(re.findall(r"https?://", text))
    uppercase = sum(1 for c in text if c.isupper())
    upper_ratio = uppercase / max(length, 1)

    words = text.split()
    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
    digits = sum(c.isdigit() for c in text)
    punct = sum(c in "!?.," for c in text)
    pvi = (hashtags + mentions + urls) / max(word_count, 1)

    return (
        length, hashtags, mentions, urls, upper_ratio,
        word_count, avg_word_len, digits, punct, pvi
    )


features = df["text"].apply(get_features)
features = pd.DataFrame(
    features.tolist(),
    columns=[
        "length", "hashtags", "mentions", "urls", "upper_ratio",
        "word_count", "avg_word_length", "digit_count",
        "punctuation_count", "pvi"
    ]
)
df = pd.concat([df, features], axis=1)

# ---------------- ADVANCED FEATURES ---------------- #

df = compute_text_intelligence(df)
df = compute_structural_features(df)

# ---------------- SAVE ---------------- #

final = df.drop(columns=["text"])
final.to_csv("../results_/features.csv", index=False)

print(f"\nFeature extraction completed! Shape: {final.shape}")
print(f"Features: {[c for c in final.columns if c not in ['id','label','source']]}")
print(final.head())
