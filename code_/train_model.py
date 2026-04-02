import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def train_kmeans(X, return_scaler=False, k=None):

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if k is None:
        k = _find_optimal_k(X_scaled)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    sil = silhouette_score(X_scaled, kmeans.labels_)
    print(f"KMeans: k={k}, silhouette={sil:.4f}")

    if return_scaler:
        return kmeans, scaler
    return kmeans


def _find_optimal_k(X_scaled, k_range=range(2, 7)):
    best_k = 2
    best_sil = -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    return best_k
