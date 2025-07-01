
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

def simulate_network_lifetime(nodes, protocol, max_rounds=100, G=None, positions=None, transmission_range=100):
    """
    Simulate network lifetime using the specified protocol.

    Args:
        nodes: List of node IDs
        protocol: Protocol function to use (e.g., leach_protocol, cro_protocol)
        max_rounds: Maximum number of rounds to simulate
        G: NetworkX graph representing the network
        positions: Dictionary mapping node IDs to (x, y) coordinates
        transmission_range: Maximum transmission range

    Returns:
        Dictionary containing simulation results:
        - rounds: List of round numbers
        - alive_nodes: List of number of alive nodes in each round
        - total_energy: List of total energy in the network in each round
        - lifetime: Number of rounds until first node death
    """
    # Make a deep copy of the graph to avoid modifying the original
    G_copy = copy.deepcopy(G)

    results = {
        'rounds': [],
        'alive_nodes': [],
        'total_energy': [],
        'lifetime': max_rounds
    }

    for round_num in range(max_rounds):
        # Record data before running the protocol
        results['rounds'].append(round_num)
        alive_nodes = [node for node in nodes if G_copy.nodes[node]['energy'] > 0]
        results['alive_nodes'].append(len(alive_nodes))
        total_energy = sum(G_copy.nodes[node]['energy'] for node in nodes)
        results['total_energy'].append(total_energy)

        # Run the protocol
        cluster_heads = protocol(nodes, positions, transmission_range, G_copy)

        # Check for dead nodes
        dead_nodes = [node for node in nodes if G_copy.nodes[node]['energy'] == 0]
        if dead_nodes and round_num < results['lifetime']:
            results['lifetime'] = round_num
            print(f"Network lifetime reached after {round_num} rounds.")

        # If all nodes are dead, break
        if len(dead_nodes) == len(nodes):
            break

    return results

def compare_protocols(nodes, protocols, protocol_names, max_rounds=100, G=None, positions=None, transmission_range=100):
    """
    Compare multiple protocols by simulating network lifetime for each.

    Args:
        nodes: List of node IDs
        protocols: List of protocol functions to compare
        protocol_names: List of names for the protocols (for plotting)
        max_rounds: Maximum number of rounds to simulate
        G: NetworkX graph representing the network
        positions: Dictionary mapping node IDs to (x, y) coordinates
        transmission_range: Maximum transmission range

    Returns:
        Dictionary containing comparison results:
        - protocol_results: Dictionary mapping protocol names to their simulation results
    """
    comparison_results = {
        'protocol_results': {}
    }

    for protocol, name in zip(protocols, protocol_names):
        print(f"Simulating {name}...")
        # Make a deep copy of the graph for each protocol
        G_copy = copy.deepcopy(G)
        results = simulate_network_lifetime(nodes, protocol, max_rounds, G_copy, positions, transmission_range)
        comparison_results['protocol_results'][name] = results
        print(f"{name} lifetime: {results['lifetime']} rounds")

    return comparison_results
