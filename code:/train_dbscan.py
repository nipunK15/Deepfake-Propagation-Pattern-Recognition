import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def train_dbscan(X):

    # 🔥 CLEAN DATA
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X_scaled)

    return dbscan