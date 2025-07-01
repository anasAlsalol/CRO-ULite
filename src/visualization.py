
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
os.makedirs('../outputs', exist_ok=True)

def plot_network_topology(positions, save_path='../outputs/network_topology.png'):
    """
    Plot the network topology and save to file.

    Args:
        positions: Dictionary mapping node IDs to (x, y) coordinates
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(*zip(*positions.values()))
    plt.title("Network Topology")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig(save_path)
    plt.close()
    print(f"Network topology plot saved to {save_path}")

def plot_benchmark_results(results, save_path='../outputs/benchmark_results.png'):
    """
    Plot benchmark results and save to file.

    Args:
        results: Dictionary containing simulation results
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['rounds'], results['total_energy'])
    plt.title("Energy Consumption Over Time")
    plt.xlabel("Rounds")
    plt.ylabel("Total Energy (Joules)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Benchmark results plot saved to {save_path}")

def plot_protocol_comparison(comparison_results, save_path='../outputs/protocol_comparison.png'):
    """
    Plot comparison of multiple protocols and save to file.

    Args:
        comparison_results: Dictionary containing comparison results
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Plot alive nodes over time
    plt.subplot(2, 1, 1)
    for protocol_name, results in comparison_results['protocol_results'].items():
        plt.plot(results['rounds'], results['alive_nodes'], label=protocol_name)
    plt.title("Number of Alive Nodes Over Time")
    plt.xlabel("Rounds")
    plt.ylabel("Number of Alive Nodes")
    plt.legend()
    plt.grid(True)

    # Plot total energy over time
    plt.subplot(2, 1, 2)
    for protocol_name, results in comparison_results['protocol_results'].items():
        plt.plot(results['rounds'], results['total_energy'], label=protocol_name)
    plt.title("Total Energy Over Time")
    plt.xlabel("Rounds")
    plt.ylabel("Total Energy (Joules)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Protocol comparison plot saved to {save_path}")
