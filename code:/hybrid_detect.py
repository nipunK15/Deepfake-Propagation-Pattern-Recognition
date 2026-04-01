import numpy as np
from sklearn.preprocessing import StandardScaler

def hybrid_detection(X, kmeans_model, dbscan_model):

    # 🔥 CLEAN DATA
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # 🔥 SCALE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔥 KMEANS DISTANCE (anomaly based on distance)
    kmeans_labels = kmeans_model.predict(X_scaled)

    # Distance from cluster centers
    distances = np.min(
        np.linalg.norm(
            X_scaled[:, None] - kmeans_model.cluster_centers_, axis=2
        ),
        axis=1
    )

    threshold = distances.mean() + 2 * distances.std()
    kmeans_anomaly = distances > threshold

    # 🔥 DBSCAN ANOMALY (-1 = outlier)
    dbscan_labels = dbscan_model.labels_
    dbscan_anomaly = dbscan_labels == -1

    # 🔥 HYBRID (OR logic)
    hybrid = kmeans_anomaly | dbscan_anomaly

    print("Hybrid detection completed!")

    return hybrid