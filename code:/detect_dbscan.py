import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib

# Load KMeans final output (contains confidence & explanations)
data = pd.read_csv("../results:/final_output.csv")

# Features
features = data.drop(columns=["id","label","suspicious","confidence","risk_level","explanation"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# -1 = anomaly
data["dbscan_suspicious"] = dbscan_labels == -1

# Save enriched DBSCAN file
data.to_csv("../results:/dbscan_output.csv", index=False)

print("DBSCAN detection completed!")
print(data["dbscan_suspicious"].value_counts())