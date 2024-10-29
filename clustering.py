from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def apply_dbscan(data, eps=0.5, min_samples=5):
    """
        Apply DBSCAN clustering algorithm to the provided data.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    return clusters

def calculate_silhouette_score(data, clusters):
    """
        Calculate the silhouette score for the given clusters.
    """
    if len(set(clusters)) > 1:  # Check if more than one cluster exists
        return silhouette_score(data, clusters)
    else:
        return None



