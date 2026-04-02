# Deepfake Propagation Pattern Recognition

Hybrid unsupervised detection system that identifies suspicious propagation patterns in social media posts using KMeans clustering, DBSCAN density analysis, and NLP content intelligence.

## Dataset

Uses the **Twitter15** and **Twitter16** rumor detection datasets (2,308 tweets total) with four ground-truth labels:

| Label | Count | Description |
|-------|-------|-------------|
| non-rumor | 579 | Verified legitimate posts |
| true | 579 | Verified true claims |
| unverified | 575 | Unverified claims |
| false | 575 | Verified false claims |

## Features

The system extracts **26 features** across three categories:

**Basic Text Features (10):** length, hashtags, mentions, urls, upper\_ratio, word\_count, avg\_word\_length, digit\_count, punctuation\_count, pvi (propagation virality index)

**NLP Intelligence Features (11):** exclamation\_density, question\_density, has\_url, has\_mention, caps\_word\_count, caps\_ratio, unique\_word\_ratio, sensational\_count, credibility\_count, special\_char\_ratio, repeated\_punct

**Structural Features (5):** url\_count, hashtag\_word\_ratio, mention\_word\_ratio, text\_complexity, digit\_ratio

## Models

- **KMeans** -- Clustering with auto-optimized k (silhouette score selection over k=2..6)
- **DBSCAN** -- Density-based anomaly detection with auto-tuned eps (90th percentile of k-nearest-neighbor distances)
- **Hybrid Detection** -- AND logic combining both models; a tweet must be flagged by both detectors to be marked suspicious
- **NLP Content Analysis** -- Rule-based detection of sensational language, urgency patterns, unverified claims, and manipulation signals (used in live prediction)

## Project Structure

```
deepfake_project/
├── code_/
│   ├── app.py                 # Streamlit dashboard (main entry point)
│   ├── extract_features.py    # Feature extraction pipeline
│   ├── graph_features.py      # NLP intelligence & structural features
│   ├── train_model.py         # KMeans training with auto-k
│   ├── train_dbscan.py        # DBSCAN training with auto-eps
│   ├── hybrid_detect.py       # Hybrid anomaly detection
│   ├── detect_abnormal.py     # Standalone KMeans detection script
│   ├── detect_dbscan.py       # Standalone DBSCAN detection script
│   ├── evaluate.py            # Baseline vs proposed evaluation
│   ├── load_data.py           # Data loading utility
│   ├── plot.py                # Visualization utility
│   ├── requirements.txt       # Python dependencies
│   └── venv/                  # Virtual environment
├── data_/
│   └── rumour_detection/
│       ├── twitter15/         # 1,490 tweets + labels
│       └── twitter16/         # 818 tweets + labels
├── results_/
│   ├── features.csv           # Extracted features (generated)
│   ├── final_output.csv       # Detection results
│   └── *.pkl                  # Saved model artifacts
└── README.md
```

## Setup

```bash
# 1. Create and activate virtual environment
cd code_
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Extract features (generates results_/features.csv)
python extract_features.py

# 4. Launch the dashboard
streamlit run app.py
```

## Usage

### Streamlit Dashboard

The dashboard has four tabs:

1. **Overview** -- Dataset statistics, label distribution, detection results, and flagged posts
2. **Model Analysis** -- Feature importance, detection accuracy per label, and baseline vs proposed comparison
3. **Early Detection** -- Tests model stability across different data sizes (20%--100%)
4. **Live Prediction** -- Enter any tweet to analyze it in real-time with cluster analysis and NLP content scoring

### Configuration (Sidebar)

- **Feature Mode**: Switch between Baseline (10 text features) and Proposed (26 features with NLP intelligence)
- **Data Percentage**: Simulate early detection by training on partial data

### Standalone Scripts

```bash
# Run feature extraction
python extract_features.py

# Run KMeans detection with explainability
python detect_abnormal.py

# Run DBSCAN detection
python detect_dbscan.py

# Run full evaluation (baseline vs proposed)
python evaluate.py
```

## Requirements

- Python 3.10+
- pandas >= 3.0
- numpy >= 2.4
- matplotlib >= 3.10
- streamlit >= 1.56
- scikit-learn >= 1.8
- joblib >= 1.5
