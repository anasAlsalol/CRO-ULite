from sklearn.cluster import KMeans
import numpy as np

def kmeans_cluster_nodes(node_data, n_clusters=5):
    """
    Perform K-means clustering on node data to identify potential cluster heads.
    
    Args:
        node_data: Array of node data (reduced dimensions from PCA)
        n_clusters: Number of clusters to form
        
    Returns:
        cluster_heads: List of potential cluster head candidates (centroids)
        labels: Cluster labels for each node
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(node_data)
    
    # Get the centroids as potential cluster head candidates
    cluster_heads = kmeans.cluster_centers_
    
    return cluster_heads, labels