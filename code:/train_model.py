import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def train_kmeans(X):

    # 🔥 CLEAN DATA
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # 🔥 SCALE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔥 TRAIN MODEL
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    print("Model training completed!")

    return kmeans