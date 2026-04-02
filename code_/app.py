import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, accuracy_score

from train_model import train_kmeans
from train_dbscan import train_dbscan
from hybrid_detect import hybrid_detection

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Deepfake Propagation Detector",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.title("Configuration")

    st.markdown("---")
    st.subheader("Feature Mode")
    mode = st.radio(
        "Choose which features to use for detection:",
        ["Baseline (Text Only)", "Proposed (Text + NLP Intelligence)"],
        help=(
            "**Baseline**: 10 surface-level text features (length, hashtags, etc.).\n\n"
            "**Proposed**: 26 features including NLP intelligence signals "
            "(sensational words, credibility cues, caps analysis, text complexity)."
        )
    )

    st.markdown("---")
    st.subheader("Data Percentage")
    percentage = st.slider(
        "Simulate early detection with partial data:",
        min_value=20, max_value=100, value=100, step=10,
        help="Use a smaller percentage to test how early the model can detect anomalies."
    )

    st.markdown("---")
    st.subheader("About")
    st.caption(
        "Hybrid unsupervised detection combining KMeans clustering and "
        "DBSCAN density analysis to identify suspicious propagation patterns "
        "in social media posts. Trained on Twitter15 + Twitter16 rumor datasets."
    )

# ================================================================
# LOAD DATA
# ================================================================

@st.cache_data
def load_data():
    return pd.read_csv("../results_/features.csv")

df_full = load_data()
df = df_full.sample(frac=percentage / 100, random_state=42)

# ================================================================
# FEATURE SETS
# ================================================================

basic_cols = [
    "length", "hashtags", "mentions", "urls", "upper_ratio",
    "word_count", "avg_word_length", "digit_count",
    "punctuation_count", "pvi"
]

proposed_cols = [
    "length", "hashtags", "mentions", "urls", "upper_ratio",
    "word_count", "avg_word_length", "digit_count",
    "punctuation_count", "pvi",
    "exclamation_density", "question_density",
    "has_url", "has_mention",
    "caps_word_count", "caps_ratio", "unique_word_ratio",
    "sensational_count", "credibility_count",
    "special_char_ratio", "repeated_punct",
    "url_count", "hashtag_word_ratio", "mention_word_ratio",
    "text_complexity", "digit_ratio"
]

if mode == "Baseline (Text Only)":
    feature_cols = basic_cols
else:
    feature_cols = proposed_cols

X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# ================================================================
# TRAIN MODELS
# ================================================================

with st.spinner("Training models..."):
    kmeans, kmeans_scaler = train_kmeans(X, return_scaler=True)
    dbscan, dbscan_scaler = train_dbscan(X, return_scaler=True)
    predictions = hybrid_detection(X, kmeans, dbscan, scaler=kmeans_scaler)

df["prediction"] = predictions

label_map = {"false": 1, "unverified": 1, "true": 0, "non-rumor": 0}
df["ground_truth"] = df["label"].map(label_map)

# ================================================================
# HEADER
# ================================================================

st.title("Deepfake Propagation Pattern Detector")
st.caption("Hybrid Unsupervised Detection with Explainable Risk Analysis | Twitter15 + Twitter16 Datasets")

# ================================================================
# KPI METRICS
# ================================================================

total = len(df)
anomalies = int(predictions.sum())
normal = total - anomalies
anomaly_rate = anomalies / total * 100

X_scaled = kmeans_scaler.transform(X)
sil_score = silhouette_score(X_scaled, kmeans.labels_)

acc1 = accuracy_score(df["ground_truth"], df["prediction"].astype(int))
acc2 = accuracy_score(df["ground_truth"], 1 - df["prediction"].astype(int))
label_agreement = max(acc1, acc2)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Tweets", f"{total:,}")
col2.metric("Anomalies", f"{anomalies:,}", f"{anomaly_rate:.1f}%")
col3.metric("Normal", f"{normal:,}")
col4.metric("Silhouette Score", f"{sil_score:.3f}")
col5.metric("Label Agreement", f"{label_agreement:.1%}")

# ================================================================
# TABS
# ================================================================

tab_overview, tab_analysis, tab_early, tab_predict = st.tabs([
    "Overview", "Model Analysis", "Early Detection", "Live Prediction"
])

# ----------------------------------------------------------------
# TAB: OVERVIEW
# ----------------------------------------------------------------

with tab_overview:
    st.subheader("Dataset Overview")

    ov1, ov2 = st.columns(2)

    with ov1:
        st.markdown("**Label Distribution (Ground Truth)**")
        fig_labels, ax_labels = plt.subplots(figsize=(5, 3))
        colors = {"non-rumor": "#2ecc71", "true": "#3498db", "unverified": "#f39c12", "false": "#e74c3c"}
        label_counts = df["label"].value_counts()
        bars = ax_labels.bar(label_counts.index, label_counts.values,
                             color=[colors.get(l, "#95a5a6") for l in label_counts.index])
        ax_labels.set_ylabel("Count")
        ax_labels.set_title("Ground Truth Labels")
        for bar, val in zip(bars, label_counts.values):
            ax_labels.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                           str(val), ha="center", fontsize=9)
        fig_labels.tight_layout()
        st.pyplot(fig_labels)

    with ov2:
        st.markdown("**Detection Results**")
        fig_det, ax_det = plt.subplots(figsize=(5, 3))
        det_counts = df["prediction"].value_counts().sort_index()
        det_labels = ["Normal", "Suspicious"]
        det_colors = ["#2ecc71", "#e74c3c"]
        bars = ax_det.bar(det_labels, [det_counts.get(False, 0), det_counts.get(True, 0)],
                          color=det_colors)
        ax_det.set_ylabel("Count")
        ax_det.set_title("Model Predictions")
        for bar in bars:
            ax_det.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(int(bar.get_height())), ha="center", fontsize=9)
        fig_det.tight_layout()
        st.pyplot(fig_det)

    st.subheader("Dataset Preview")
    display_cols = ["id", "label", "prediction"] + feature_cols[:6]
    st.dataframe(
        df[display_cols].head(20).style.map(
            lambda v: "background-color: #fadbd8" if v is True else "",
            subset=["prediction"]
        ),
        use_container_width=True
    )

    with st.expander("Flagged Suspicious Posts"):
        suspicious_df = df[df["prediction"]].reset_index(drop=True)
        if len(suspicious_df) > 0:
            st.dataframe(suspicious_df[["id", "label", "source"] + feature_cols[:6]],
                         use_container_width=True)
        else:
            st.info("No suspicious posts detected with current settings.")

# ----------------------------------------------------------------
# TAB: MODEL ANALYSIS
# ----------------------------------------------------------------

with tab_analysis:
    st.subheader("Model Performance")

    ma1, ma2, ma3 = st.columns(3)
    ma1.metric("Features Used", len(feature_cols))
    ma2.metric("KMeans Clusters (k)", kmeans.n_clusters)
    ma3.metric("DBSCAN Noise %",
               f"{(dbscan.labels_ == -1).sum() / len(X) * 100:.1f}%")

    st.markdown("---")

    an1, an2 = st.columns(2)

    with an1:
        st.markdown("**Feature Importance (Variance After Scaling)**")
        variances = pd.Series(X_scaled.var(axis=0), index=feature_cols).sort_values(ascending=True)
        fig_var, ax_var = plt.subplots(figsize=(5, max(4, len(feature_cols) * 0.25)))
        ax_var.barh(variances.index, variances.values, color="#3498db")
        ax_var.set_xlabel("Variance")
        ax_var.set_title("Feature Variance (Scaled)")
        fig_var.tight_layout()
        st.pyplot(fig_var)

    with an2:
        st.markdown("**Detection Accuracy by Label**")
        if "ground_truth" in df.columns:
            cross = pd.crosstab(df["label"], df["prediction"], margins=True)
            cross.columns = [
                "Predicted Normal" if c is False else
                "Predicted Suspicious" if c is True else "Total"
                for c in cross.columns
            ]
            st.dataframe(cross, use_container_width=True)

    st.markdown("---")
    st.subheader("Baseline vs Proposed Comparison")

    with st.spinner("Running comparison..."):
        X_b = df_full[basic_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_p = df_full[proposed_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_all = df_full["label"].map(label_map)

        km_b, sc_b = train_kmeans(X_b, return_scaler=True)
        db_b = train_dbscan(X_b)
        r_b = hybrid_detection(X_b, km_b, db_b, scaler=sc_b)
        sil_b = silhouette_score(sc_b.transform(X_b), km_b.labels_)
        acc_b = max(accuracy_score(y_all, r_b.astype(int)),
                    accuracy_score(y_all, 1 - r_b.astype(int)))

        km_p, sc_p = train_kmeans(X_p, return_scaler=True)
        db_p = train_dbscan(X_p)
        r_p = hybrid_detection(X_p, km_p, db_p, scaler=sc_p)
        sil_p = silhouette_score(sc_p.transform(X_p), km_p.labels_)
        acc_p = max(accuracy_score(y_all, r_p.astype(int)),
                    accuracy_score(y_all, 1 - r_p.astype(int)))

    comp_df = pd.DataFrame({
        "Metric": ["Features", "Anomalies Detected", "Anomaly Rate",
                    "Silhouette Score", "Label Agreement"],
        "Baseline": [len(basic_cols), int(r_b.sum()),
                     f"{r_b.sum() / len(r_b) * 100:.1f}%",
                     f"{sil_b:.4f}", f"{acc_b:.1%}"],
        "Proposed": [len(proposed_cols), int(r_p.sum()),
                     f"{r_p.sum() / len(r_p) * 100:.1f}%",
                     f"{sil_p:.4f}", f"{acc_p:.1%}"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------
# TAB: EARLY DETECTION
# ----------------------------------------------------------------

with tab_early:
    st.subheader("Early Detection Experiment")
    st.caption(
        "Tests whether the model can detect anomalies with limited data. "
        "A stable anomaly rate across data sizes indicates robust detection."
    )

    sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    rates, counts, totals = [], [], []

    with st.spinner("Running early detection experiment..."):
        for s in sizes:
            temp_df = df_full.sample(frac=s, random_state=42)
            X_temp = temp_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            k = train_kmeans(X_temp)
            d = train_dbscan(X_temp)
            r = hybrid_detection(X_temp, k, d)
            counts.append(int(r.sum()))
            rates.append(r.sum() / len(X_temp) * 100)
            totals.append(len(X_temp))

    ed1, ed2 = st.columns(2)

    with ed1:
        fig_rate, ax_rate = plt.subplots(figsize=(5, 3.5))
        pcts = [int(s * 100) for s in sizes]
        ax_rate.plot(pcts, rates, marker="o", color="#e74c3c", linewidth=2)
        ax_rate.fill_between(pcts, rates, alpha=0.1, color="#e74c3c")
        ax_rate.set_xlabel("Data %")
        ax_rate.set_ylabel("Anomaly Rate (%)")
        ax_rate.set_title("Anomaly Rate Stability")
        ax_rate.set_ylim(bottom=0)
        ax_rate.grid(axis="y", alpha=0.3)
        fig_rate.tight_layout()
        st.pyplot(fig_rate)

    with ed2:
        fig_cnt, ax_cnt = plt.subplots(figsize=(5, 3.5))
        ax_cnt.bar(pcts, counts, color="#3498db", width=8)
        for i, (p, c, t) in enumerate(zip(pcts, counts, totals)):
            ax_cnt.text(p, c + 2, f"{c}/{t}", ha="center", fontsize=8)
        ax_cnt.set_xlabel("Data %")
        ax_cnt.set_ylabel("Anomaly Count")
        ax_cnt.set_title("Anomaly Count vs Data Size")
        ax_cnt.set_ylim(bottom=0)
        ax_cnt.grid(axis="y", alpha=0.3)
        fig_cnt.tight_layout()
        st.pyplot(fig_cnt)

    ed_df = pd.DataFrame({
        "Data %": [f"{int(s*100)}%" for s in sizes],
        "Samples": totals,
        "Anomalies": counts,
        "Rate (%)": [f"{r:.1f}" for r in rates],
    })
    st.dataframe(ed_df, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------
# TAB: LIVE PREDICTION
# ----------------------------------------------------------------

with tab_predict:
    st.subheader("Live Tweet Prediction")
    st.caption("Enter a tweet to analyze it for suspicious patterns using both cluster analysis and NLP content analysis.")

    user_tweet = st.text_area(
        "Enter a tweet to analyze:",
        placeholder="e.g., BREAKING: Major scandal exposed! Share before they delete this!!!",
        height=100
    )

    SENSATIONAL_WORDS = {
        "breaking", "shocking", "urgent", "exclusive", "confirmed", "boom",
        "dead", "killed", "dies", "died", "death", "murder", "arrested",
        "scandal", "exposed", "leaked", "secret", "conspiracy", "hoax",
        "fake", "fraud", "scam", "attack", "war", "bomb", "explosion",
        "crash", "destroyed", "emergency", "alert", "warning",
    }

    URGENCY_PHRASES = [
        r"share before.*deleted", r"they don'?t want you to know",
        r"must (watch|read|see|share)", r"happening now",
        r"wake up", r"open your eyes", r"spread the word",
        r"going viral", r"mainstream media won'?t",
        r"(100|thousand|million)s? (dead|killed|infected)",
    ]

    def extract_features_live(text):
        text = str(text)
        length = len(text)
        hashtags = text.count("#")
        mentions = text.count("@")
        urls = len(re.findall(r"https?://", text))
        uppercase = sum(1 for c in text if c.isupper())
        upper_ratio = uppercase / max(length, 1)
        words = text.split()
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        digit_count = sum(c.isdigit() for c in text)
        punctuation_count = sum(c in "!?.," for c in text)
        pvi = (hashtags + mentions + urls) / max(word_count, 1)

        basic = {
            "length": length, "hashtags": hashtags, "mentions": mentions,
            "urls": urls, "upper_ratio": upper_ratio, "word_count": word_count,
            "avg_word_length": avg_word_length, "digit_count": digit_count,
            "punctuation_count": punctuation_count, "pvi": pvi,
        }

        words_lower = set(text.lower().split())
        caps_words = [w for w in text.split() if w.isupper() and len(w) > 1]
        proposed = {
            "exclamation_density": text.count("!") / max(length, 1),
            "question_density": text.count("?") / max(length, 1),
            "has_url": int(bool(urls)),
            "has_mention": int(bool(mentions)),
            "caps_word_count": len(caps_words),
            "caps_ratio": len(caps_words) / max(word_count, 1),
            "unique_word_ratio": len(set(text.lower().split())) / max(word_count, 1),
            "sensational_count": len(words_lower & SENSATIONAL_WORDS),
            "credibility_count": len(words_lower & {"according", "source", "official",
                                                     "statement", "reported", "confirmed",
                                                     "evidence", "study", "research"}),
            "special_char_ratio": sum(not c.isalnum() and not c.isspace() for c in text) / max(length, 1),
            "repeated_punct": len(re.findall(r"[!?]{2,}", text)),
            "url_count": urls,
            "hashtag_word_ratio": hashtags / max(word_count, 1),
            "mention_word_ratio": mentions / max(word_count, 1),
            "text_complexity": float(np.std([len(w) for w in words])) if len(words) > 1 else 0,
            "digit_ratio": digit_count / max(length, 1),
        }

        return basic, proposed

    def compute_nlp_suspicion(text):
        text_lower = text.lower()
        words_lower = set(text_lower.split())
        signals = []
        score = 0.0

        sensational_hits = words_lower & SENSATIONAL_WORDS
        if sensational_hits:
            score += 0.25 * len(sensational_hits)
            signals.append(f"Sensational words: {', '.join(sorted(sensational_hits))}")

        for pattern in URGENCY_PHRASES:
            if re.search(pattern, text_lower):
                score += 0.3
                signals.append("Urgency/manipulation pattern detected")
                break

        excl = text.count("!")
        if excl >= 2:
            score += 0.15 * min(excl, 5)
            signals.append(f"Excessive exclamation marks: {excl}")

        caps = [w for w in text.split() if w.isupper() and len(w) > 1]
        if len(caps) >= 2:
            score += 0.2 * min(len(caps), 5)
            signals.append(f"ALL-CAPS words: {', '.join(caps[:5])}")

        if re.search(r"(is|are|was|has been|have been)\s+(dead|killed|arrested|fired)", text_lower):
            score += 0.35
            signals.append("Unverified claim pattern (X is dead/killed/arrested)")

        has_url = bool(re.search(r"https?://", text_lower))
        if not has_url and re.search(r"(confirmed|report|according|sources say)", text_lower):
            score += 0.2
            signals.append("Makes claims without providing source URL")

        return min(score, 1.0), signals

    if st.button("Analyze Tweet", type="primary", use_container_width=True):
        if not user_tweet.strip():
            st.warning("Please enter a tweet to analyze.")
        else:
            basic_feats, proposed_feats = extract_features_live(user_tweet)

            if mode == "Baseline (Text Only)":
                feat_values = basic_feats
            else:
                feat_values = {**basic_feats, **proposed_feats}

            live_df = pd.DataFrame([feat_values])
            live_df = live_df[feature_cols]
            live_df = live_df.replace([np.inf, -np.inf], np.nan).fillna(0)

            feats_scaled = kmeans_scaler.transform(live_df)

            distances = np.linalg.norm(feats_scaled - kmeans.cluster_centers_, axis=1)
            min_dist = distances.min()

            X_train_scaled = kmeans_scaler.transform(X)
            train_distances = np.min(
                np.linalg.norm(X_train_scaled[:, None] - kmeans.cluster_centers_, axis=2),
                axis=1
            )
            threshold = train_distances.mean() + 2 * train_distances.std()
            cluster_anomaly = min_dist > threshold

            nlp_score, nlp_signals = compute_nlp_suspicion(user_tweet)
            is_suspicious = cluster_anomaly or nlp_score >= 0.5

            st.markdown("---")

            if is_suspicious:
                st.error("**SUSPICIOUS** -- This tweet shows anomalous patterns")
            else:
                st.success("**NORMAL** -- No anomalous patterns detected")

            lp1, lp2 = st.columns(2)

            with lp1:
                st.markdown("##### Cluster Analysis")
                dist_pct = min_dist / threshold * 100
                st.progress(min(dist_pct / 100, 1.0))
                st.write(f"Distance: **{min_dist:.3f}** / threshold {threshold:.3f}")
                if cluster_anomaly:
                    st.error("Cluster: ANOMALOUS")
                else:
                    st.success("Cluster: NORMAL")

            with lp2:
                st.markdown("##### Content Analysis")
                st.progress(min(nlp_score, 1.0))
                st.write(f"Suspicion score: **{nlp_score:.2f}** / 1.00")
                if nlp_signals:
                    for sig in nlp_signals:
                        st.warning(f"- {sig}")
                else:
                    st.success("No suspicious content patterns")

            with st.expander("Feature Breakdown"):
                feat_df = pd.DataFrame({
                    "Feature": list(feat_values.keys()),
                    "Value": [f"{v:.4f}" if isinstance(v, float) else str(v)
                              for v in feat_values.values()]
                })
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
