import pandas as pd
import re

# ---------------- LOAD DATA ---------------- #

data = pd.read_csv("../data:/rumour_detection/twitter15/source_tweets.txt",
                   sep="\t",
                   names=["id","text"],
                   dtype={"id":str})

labels = pd.read_csv("../data:/rumour_detection/twitter15/label.txt",
                     sep=":",
                     names=["label","id"],
                     dtype={"id":str})

df = data.merge(labels, on="id")

# ---------------- FEATURE FUNCTIONS ---------------- #

def get_features(text):
    text = str(text)

    length = len(text)
    hashtags = text.count("#")
    mentions = text.count("@")
    urls = len(re.findall(r"http", text))
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

# ---------------- APPLY BASIC FEATURES ---------------- #

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

# ---------------- ADD GRAPH REQUIREMENTS ---------------- #

# Your dataset doesn't have real propagation structure,
# so we simulate minimal structure for now

df["parent_id"] = None
df["timestamp"] = range(len(df))

# ---------------- GRAPH + TEMPORAL FEATURES ---------------- #

from graph_features import compute_graph_features, compute_temporal_features

df = compute_graph_features(df)
df = compute_temporal_features(df)

# ---------------- SAVE FINAL FEATURES ---------------- #

final = df.drop(columns=["text"])

final.to_csv("../results:/features.csv", index=False)

# ---------------- OUTPUT ---------------- #

print("Feature extraction completed!")
print(final.head())