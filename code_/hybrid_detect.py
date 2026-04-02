import numpy as np
from sklearn.preprocessing import StandardScaler

def hybrid_detection(X, kmeans_model, dbscan_model, scaler=None, strategy="and"):
    """
    strategy: "and" = both must agree (precise), "or" = either flags (sensitive)
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    kmeans_labels = kmeans_model.predict(X_scaled)

    distances = np.min(
        np.linalg.norm(
            X_scaled[:, None] - kmeans_model.cluster_centers_, axis=2
        ),
        axis=1
    )

    threshold = distances.mean() + 2 * distances.std()
    kmeans_anomaly = distances > threshold

    dbscan_labels = dbscan_model.labels_
    dbscan_anomaly = dbscan_labels == -1

    if strategy == "and":
        hybrid = kmeans_anomaly & dbscan_anomaly
    else:
        hybrid = kmeans_anomaly | dbscan_anomaly

    km_count = kmeans_anomaly.sum()
    db_count = dbscan_anomaly.sum()
    print(f"Hybrid detection: KMeans={km_count}, DBSCAN={db_count}, combined({strategy})={hybrid.sum()}")

    return hybrid