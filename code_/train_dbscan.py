import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def _auto_eps(X_scaled, min_samples=5):
    """Compute eps from the knee of the k-distance graph."""
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    dists, _ = nn.kneighbors(X_scaled)
    k_dists = np.sort(dists[:, -1])
    return float(np.percentile(k_dists, 90))

def train_dbscan(X, return_scaler=False, eps=None, min_samples=5):

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if eps is None:
        eps = _auto_eps(X_scaled, min_samples)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X_scaled)

    noise_pct = (dbscan.labels_ == -1).sum() / len(X_scaled) * 100
    print(f"DBSCAN: eps={eps:.3f}, clusters={len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)}, noise={noise_pct:.1f}%")

    if return_scaler:
        return dbscan, scaler
    return dbscan