import os
import numpy as np
import random
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from src.clustering import kmeans_cluster_nodes
from src.energy_model import calculate_energy_consumption, update_energy


def apply_pca_reduction(node_data, n_components=2):
    """
    Apply PCA dimensionality reduction to node data.

    Args:
        node_data (list or numpy.ndarray): Data to reduce dimensions
        n_components (int): Number of components to keep

    Returns:
        tuple: (pca_object, reduced_data)
    """
    node_data = np.array(node_data)
    n_samples, n_features = node_data.shape

    if n_samples >= n_components and n_features >= n_components:
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(node_data)
        return pca, reduced_data
    else:
        # If we don't have enough samples or features, create a pass-through PCA
        class PassThroughPCA:
            def transform(self, x): 
                return np.array(x)
        return PassThroughPCA(), node_data


def build_ann_model(input_dim):
    """
    Build an Artificial Neural Network model using TensorFlow.

    Args:
        input_dim (int): Input dimension for the model

    Returns:
        tensorflow.keras.Model: Compiled ANN model
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_ann_model(model, X_train, y_train, epochs=100):
    """
    Train the ANN model.

    Args:
        model (tensorflow.keras.Model): Model to train
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        epochs (int): Number of training epochs

    Returns:
        tensorflow.keras.Model: Trained model
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    return model


def predict_ch_performance(node_features, model):
    """
    Predict cluster head performance using the trained model.

    Args:
        node_features (numpy.ndarray): Features of the node
        model (tensorflow.keras.Model): Trained model

    Returns:
        float: Predicted performance score
    """
    return float(model.predict(np.array([node_features]), verbose=0)[0][0])


def prepare_node_data(nodes, positions, G):
    """
    Prepare node data for clustering.

    Args:
        nodes (list): List of node IDs
        positions (dict): Dictionary mapping node IDs to positions
        G (networkx.Graph): Network graph

    Returns:
        list: List of node data [x, y, energy, signal_strength]
    """
    node_data = []
    for node in nodes:
        if G.nodes[node]['energy'] > 0:
            node_data.append([
                positions[node][0], 
                positions[node][1], 
                G.nodes[node]['energy'],
                G.nodes[node].get('signal_strength', 0)
            ])
    return node_data


def select_cluster_heads(nodes, labels, n_clusters, G):
    """
    Select cluster heads based on energy levels.

    Args:
        nodes (list): List of node IDs
        labels (list): Cluster labels for each node
        n_clusters (int): Number of clusters
        G (networkx.Graph): Network graph

    Returns:
        list: Selected cluster heads
    """
    final_chs = []
    for i in range(n_clusters):
        cluster_nodes = [nodes[j] for j in range(len(labels)) if labels[j] == i]
        if cluster_nodes:
            ch = max(cluster_nodes, key=lambda node: G.nodes[node]['energy'])
            final_chs.append(ch)
    return final_chs


def prepare_training_data(nodes, positions, G):
    """
    Prepare training data for the ANN model.

    Args:
        nodes (list): List of node IDs
        positions (dict): Dictionary mapping node IDs to positions
        G (networkx.Graph): Network graph

    Returns:
        tuple: (X_train, y_train)
    """
    X_train = np.array([
        list(positions[node]) + [
            G.nodes[node]['energy'], 
            G.nodes[node].get('signal_strength', 0)
        ] for node in nodes
    ])
    y_train = np.array([G.nodes[node]['energy'] for node in nodes])
    return X_train, y_train


def metaheuristic_ch_selection(ch_candidates, positions, G):
    """
    Select the best cluster head using a metaheuristic approach.

    This function selects the node with the highest energy as the cluster head.

    Args:
        ch_candidates (list): List of cluster head candidates
        positions (dict): Dictionary mapping node IDs to positions
        G (networkx.Graph): Network graph

    Returns:
        int: Selected cluster head
    """
    if not ch_candidates:
        return None

    # Select the node with the highest energy
    return max(ch_candidates, key=lambda node: G.nodes[node]['energy'])


def cro_protocol(nodes, positions, transmission_range, G, model_save_path="../model/cro_model.h5"):
    """
    Chemical Reaction Optimization (CRO) protocol for wireless sensor networks.

    Args:
        nodes (list): List of node IDs
        positions (dict): Dictionary mapping node IDs to positions
        transmission_range (float): Maximum transmission range
        G (networkx.Graph): Network graph
        model_save_path (str): Path to save the trained model

    Returns:
        list: Selected cluster heads
    """
    # Prepare node data
    node_data = prepare_node_data(nodes, positions, G)
    if not node_data:
        return []

    # Apply PCA reduction
    node_data = np.array(node_data)
    n_samples, n_features = node_data.shape
    pca, reduced_data = apply_pca_reduction(node_data, n_components=2)
    if len(reduced_data) < 2:
        return [nodes[i] for i in range(len(reduced_data))]

    # Determine number of clusters and perform clustering
    n_clusters = max(1, int(len(reduced_data) * 0.15))
    cluster_heads, labels = kmeans_cluster_nodes(reduced_data, n_clusters)
    preliminary_chs = select_cluster_heads(nodes, labels, n_clusters, G)

    # Prepare training data for the ANN model
    X_train_full, y_train = prepare_training_data(nodes, positions, G)
    try:
        X_train = pca.transform(X_train_full)
    except Exception:
        X_train = X_train_full

    # Build and train the ANN model
    tf_model = build_ann_model(input_dim=X_train.shape[1])
    trained_model = train_ann_model(tf_model, X_train, y_train, epochs=200)

    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    trained_model.save(model_save_path)

    # Score cluster head candidates
    ch_scores = []
    for idx, ch in enumerate(preliminary_chs):
        node_idx = nodes.index(ch)
        features = reduced_data[node_idx]
        score = predict_ch_performance(features, trained_model)
        ch_scores.append((ch, score))

    # Select top cluster heads
    ch_scores.sort(key=lambda x: x[1], reverse=True)
    top_ch_candidates = [ch for ch, score in ch_scores[:n_clusters]]

    # Apply metaheuristic selection
    final_ch = metaheuristic_ch_selection(top_ch_candidates, positions, G)
    final_chs = [final_ch] if final_ch is not None else []

    # Update energy consumption for non-cluster-head nodes
    for node in nodes:
        if node not in final_chs and G.nodes[node]['energy'] > 0:
            closest_head = min(
                final_chs, 
                key=lambda head: np.linalg.norm(np.array(positions[node]) - np.array(positions[head]))
            )
            distance = np.linalg.norm(np.array(positions[node]) - np.array(positions[closest_head]))
            energy_spent = calculate_energy_consumption(distance, transmission_range)
            update_energy(node, energy_spent, G)

    return final_chs
