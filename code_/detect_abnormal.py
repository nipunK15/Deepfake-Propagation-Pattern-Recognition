import pandas as pd
import joblib
import numpy as np

# Load data
data = pd.read_csv("../results_/features.csv")

# Use ALL numeric features (except id, label)
X = data.drop(columns=["id","label"])

# Load model
kmeans = joblib.load("../results_/kmeans_model.pkl")
scaler = joblib.load("../results_/scaler.pkl")

# Scale features
X_scaled = scaler.transform(X)

# Distance to nearest cluster center
distances = kmeans.transform(X_scaled).min(axis=1)

# ---------------- ADAPTIVE THRESHOLD ----------------
threshold = distances.mean() + 2 * distances.std()
print("Adaptive threshold:", threshold)

# ---------------- CONFIDENCE SCORE ----------------
confidence = (distances - distances.min()) / (distances.max() - distances.min())

# Detection
data["suspicious"] = distances > threshold
data["confidence"] = confidence

# ---------------- EXPLAINABILITY ----------------
feature_means = X.mean()
feature_stds = X.std()

def explain_row(row):
    explanations = []
    for col in X.columns:
        if row[col] > feature_means[col] + feature_stds[col]:
            explanations.append(col)
    return ", ".join(explanations)

data["explanation"] = data.apply(
    lambda row: explain_row(row) if row["suspicious"] else "",
    axis=1

)
def risk_level(conf):
    if conf < 0.3:
        return "Low"
    elif conf < 0.6:
        return "Medium"
    elif conf < 0.8:
        return "High"
    else:
        return "Critical"

data["risk_level"] = data["confidence"].apply(risk_level)

# Save
data.to_csv("../results_/final_output.csv", index=False)

print("Detection completed!")
print(data.head())

