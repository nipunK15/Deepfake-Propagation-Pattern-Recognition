import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from train_model import train_kmeans
from train_dbscan import train_dbscan
from hybrid_detect import hybrid_detection

st.set_page_config(page_title="Deepfake Propagation Pattern Recognition")

st.title("Deepfake Propagation Pattern Recognition")
st.write("Hybrid Unsupervised Detection with Explainable Risk Analysis")

# ---------------- CONFIGURATION ---------------- #

st.subheader("Model Configuration")

mode = st.radio(
    "Select Feature Mode",
    ["Baseline (Text Features Only)", "Proposed (Graph + Temporal)"]
)

percentage = st.slider(
    "Select Data Percentage (Early Detection)",
    20, 100, 100
)

# ---------------- LOAD DATA ---------------- #

df = pd.read_csv("../results:/features.csv")

# Apply early detection sampling
df = df.sample(frac=percentage/100, random_state=42)

# ---------------- FEATURE SETS ---------------- #

basic_cols = [
    "length","hashtags","mentions","urls","upper_ratio",
    "word_count","avg_word_length","digit_count",
    "punctuation_count","pvi"
]

graph_cols = basic_cols + [
    "num_children","cascade_depth","time_diff","avg_time_gap","burstiness"
]

# Select feature mode
if mode == "Baseline (Text Features Only)":
    X = df[basic_cols]
else:
    X = df[graph_cols]

# Clean data
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# ---------------- MODEL TRAINING ---------------- #

kmeans = train_kmeans(X)
dbscan = train_dbscan(X)

result = hybrid_detection(X, kmeans, dbscan)

df["prediction"] = result

data = df
flag_col = "prediction"

# ---------------- INFO ---------------- #

st.subheader("Experiment Info")
st.write("Mode:", mode)
st.write("Data Used:", f"{percentage}%")

# ---------------- DATASET PREVIEW ---------------- #

st.subheader("Dataset Preview")
st.dataframe(data.head())

# ---------------- SUMMARY ---------------- #

st.subheader("Detection Summary")
st.write(data[flag_col].value_counts())

# ---------------- BAR CHART ---------------- #

st.subheader("Abnormal vs Normal Patterns")
fig, ax = plt.subplots()
data[flag_col].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)

# ---------------- SUSPICIOUS POSTS ---------------- #

st.subheader("Suspicious Posts")

suspicious_df = data[data[flag_col] == True].reset_index(drop=True)

if len(suspicious_df) > 0:
    st.dataframe(suspicious_df)
else:
    st.info("No suspicious posts detected.")

# ---------------- EARLY DETECTION GRAPH ---------------- #

st.subheader("Early Detection Trend")

sizes = [0.2, 0.5, 1.0]
results = []

for s in sizes:
    temp_df = df.sample(frac=s, random_state=42)

    X_temp = temp_df[graph_cols] if mode != "Baseline (Text Features Only)" else temp_df[basic_cols]
    X_temp = X_temp.replace([np.inf, -np.inf], np.nan).fillna(0)

    k = train_kmeans(X_temp)
    d = train_dbscan(X_temp)
    r = hybrid_detection(X_temp, k, d)

    results.append(r.sum())

fig2, ax2 = plt.subplots()
ax2.plot([20, 50, 100], results, marker='o')
ax2.set_xlabel("Data %")
ax2.set_ylabel("Anomalies Detected")
ax2.set_title("Early Detection Capability")

st.pyplot(fig2)

# ---------------- COMPARISON ---------------- #

if st.checkbox("Show Baseline vs Proposed Comparison"):

    df_full = pd.read_csv("../results:/features.csv")
    df_full = df_full.fillna(0)

    X_basic = df_full[basic_cols]
    X_graph = df_full[graph_cols]

    k1 = train_kmeans(X_basic)
    d1 = train_dbscan(X_basic)
    r1 = hybrid_detection(X_basic, k1, d1)

    k2 = train_kmeans(X_graph)
    d2 = train_dbscan(X_graph)
    r2 = hybrid_detection(X_graph, k2, d2)

    st.write("Baseline anomalies:", r1.sum())
    st.write("Proposed anomalies:", r2.sum())

# ---------------- LIVE TWEET PREDICTION ---------------- #

st.subheader("Live Tweet Prediction")

user_tweet = st.text_area("Enter a tweet to analyze:")

if st.button("Predict"):

    import re

    def extract_features_live(text):
        text = str(text)

        length = len(text)
        hashtags = text.count("#")
        mentions = text.count("@")
        urls = len(re.findall(r"http", text))
        uppercase = sum(1 for c in text if c.isupper())
        upper_ratio = uppercase / max(length,1)

        words = text.split()
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / max(word_count,1)
        digit_count = sum(c.isdigit() for c in text)
        punctuation_count = sum(c in "!?.," for c in text)
        pvi = (hashtags + mentions + urls) / max(word_count,1)

        return np.array([
            length, hashtags, mentions, urls, upper_ratio,
            word_count, avg_word_length, digit_count,
            punctuation_count, pvi
        ]).reshape(1,-1)

    feats = extract_features_live(user_tweet)
    feats = np.nan_to_num(feats)

    k = train_kmeans(pd.DataFrame(feats, columns=basic_cols))
    d = train_dbscan(pd.DataFrame(feats, columns=basic_cols))

    result = hybrid_detection(pd.DataFrame(feats, columns=basic_cols), k, d)

    if result[0]:
        st.error("Prediction: SUSPICIOUS")
    else:
        st.success("Prediction: NORMAL")