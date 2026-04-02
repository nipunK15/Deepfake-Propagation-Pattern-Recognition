import pandas as pd
from train_model import train_kmeans
from train_dbscan import train_dbscan
from hybrid_detect import hybrid_detection

# ---------------- LOAD FEATURES ---------------- #

df = pd.read_csv("../results_/features.csv")

# ---------------- PREPARE DATA ---------------- #

import numpy as np

X = df.drop(columns=["id", "label"])

# 🔥 FULL CLEANING (IMPORTANT)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# ---------------- BASIC FEATURES ---------------- #

basic_cols = [
    "length","hashtags","mentions","urls","upper_ratio",
    "word_count","avg_word_length","digit_count",
    "punctuation_count","pvi"
]

X_basic = X[basic_cols]

# ---------------- GRAPH FEATURES ---------------- #

graph_cols = basic_cols + [
    "num_children","cascade_depth","time_diff","avg_time_gap", "burstiness"
]

X_graph = X[graph_cols]

# ---------------- TRAIN MODELS ---------------- #

kmeans_1 = train_kmeans(X_basic)
dbscan_1 = train_dbscan(X_basic)

kmeans_2 = train_kmeans(X_graph)
dbscan_2 = train_dbscan(X_graph)

# ---------------- DETECT ---------------- #

result_basic = hybrid_detection(X_basic, kmeans_1, dbscan_1)
result_graph = hybrid_detection(X_graph, kmeans_2, dbscan_2)

# ---------------- RESULTS ---------------- #

print("\n===== RESULTS COMPARISON =====\n")

print("WITHOUT Graph Features:")
print("Total anomalies:", result_basic.sum())

print("\nWITH Graph + Temporal Features:")
print("Total anomalies:", result_graph.sum())


print("\n===== EARLY DETECTION EXPERIMENT =====\n")

sizes = [0.2, 0.5, 1.0]

for s in sizes:
    subset = df.sample(frac=s, random_state=42)

    X_sub = subset.drop(columns=["id", "label"])

    import numpy as np
    X_sub = X_sub.replace([np.inf, -np.inf], np.nan)
    X_sub = X_sub.fillna(0)

    X_sub = X_sub[graph_cols]  # use best features

    kmeans = train_kmeans(X_sub)
    dbscan = train_dbscan(X_sub)

    result = hybrid_detection(X_sub, kmeans, dbscan)

    print(f"\nData used: {int(s*100)}%")
    print("Anomalies detected:", result.sum())