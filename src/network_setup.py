import random
import networkx as nx
import numpy as np

NUM_NODES = 100
TRANSMISSION_RANGE = 100  # meters
INITIAL_ENERGY = 1.0  # Joules

def calculate_signal_strength(position, transmission_range):
    """
    Calculate signal strength of a node based on its distance from the center.
    For demonstration, returns a value inversely proportional to the distance to the center.
    """
    center = (250, 250)
    distance = np.linalg.norm(np.array(position) - np.array(center))
    # Basic model: stronger in center, weaker at edges; normalized to [0, 1]
    signal_strength = max(0, 1 - distance / transmission_range)
    return signal_strength

def initialize_network():
    G = nx.Graph()
    positions = {i: (random.uniform(0, 500), random.uniform(0, 500)) for i in range(NUM_NODES)}
    node_energies = {i: INITIAL_ENERGY for i in range(NUM_NODES)}

    # Add nodes to the graph first
    G.add_nodes_from(range(NUM_NODES))

    # Now set attributes for each node, including signal strength
    for node in G.nodes:
        G.nodes[node]['position'] = positions[node]
        G.nodes[node]['energy'] = node_energies[node]
        G.nodes[node]['signal_strength'] = calculate_signal_strength(positions[node], TRANSMISSION_RANGE)

    return G, positions, node_energies