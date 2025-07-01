
from sklearn.decomposition import PCA
import numpy as np

def calculate_energy_consumption(distance, transmission_power):
    energy = transmission_power + 0.0001 * distance**2  # Free space model
    return energy

def update_energy(node, energy_spent, G):
    G.nodes[node]['energy'] -= energy_spent
    if G.nodes[node]['energy'] < 0:
        G.nodes[node]['energy'] = 0

def apply_pca(node_data, n_components=2):
    data = np.array(node_data)
    # Handle edge cases: if not enough samples or features, skip PCA
    if data.shape[0] < 2 or data.shape[1] < n_components:
        # Not enough samples or features for PCA, return data as-is
        return data
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data
