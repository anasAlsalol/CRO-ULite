import random
import numpy as np
from src.energy_model import calculate_energy_consumption, update_energy, apply_pca
from src.clustering import kmeans_cluster_nodes

def leach_protocol(nodes, positions, transmission_range, G):
    """
    Standard LEACH protocol without PCA and K-means clustering.
    Randomly selects 20% of nodes as cluster heads.
    """
    cluster_heads = random.sample(nodes, int(len(nodes) / 5))  # 20% cluster heads
    for node in nodes:
        if node not in cluster_heads:
            closest_head = min(cluster_heads, key=lambda head: np.linalg.norm(np.array(positions[node]) - np.array(positions[head])))
            distance = np.linalg.norm(np.array(positions[node]) - np.array(positions[closest_head]))
            energy_spent = calculate_energy_consumption(distance, transmission_range)
            update_energy(node, energy_spent, G)
    return cluster_heads

def leach_with_pca_kmeans(nodes, positions, transmission_range, G):
    """
    Modified LEACH protocol that uses PCA and K-means clustering
    to select cluster heads more efficiently.
    """
    # Prepare node data for clustering
    node_data = []
    for node in nodes:
        if G.nodes[node]['energy'] > 0:  # Only consider nodes with energy
            node_data.append([
                positions[node][0],
                positions[node][1],
                G.nodes[node]['energy'],
                G.nodes[node].get('signal_strength', 0)
            ])

    # If no nodes with energy, return empty list
    if not node_data:
        return []

    # Apply PCA to reduce dimensions
    reduced_data = apply_pca(node_data)

    # Determine optimal number of clusters (20% of active nodes, same as standard LEACH)
    n_clusters = max(1, int(len(reduced_data) * 0.2))

    # Perform K-means clustering
    cluster_centers, labels = kmeans_cluster_nodes(reduced_data, n_clusters=n_clusters)

    # Select cluster heads based on energy and distance to centroid
    cluster_heads = []
    for i in range(n_clusters):
        # Get nodes in this cluster
        cluster_nodes = [nodes[j] for j in range(len(labels)) if labels[j] == i and j < len(nodes)]

        if cluster_nodes:
            # Select node with highest energy as cluster head
            ch = max(cluster_nodes, key=lambda node: G.nodes[node]['energy'])
            cluster_heads.append(ch)

    # For each non-cluster-head node, find closest cluster head and update energy
    for node in nodes:
        if node not in cluster_heads and G.nodes[node]['energy'] > 0:
            closest_head = min(cluster_heads, key=lambda head: np.linalg.norm(np.array(positions[node]) - np.array(positions[head])))
            distance = np.linalg.norm(np.array(positions[node]) - np.array(positions[closest_head]))
            energy_spent = calculate_energy_consumption(distance, transmission_range)
            update_energy(node, energy_spent, G)

    return cluster_heads
