
# Standard library imports
import os
import pandas as pd

# Local imports
from src.network_setup import initialize_network, NUM_NODES, TRANSMISSION_RANGE
from src.leach_protocol import leach_protocol, leach_with_pca_kmeans
from src.cro_protocol import cro_protocol
from src.benchmarking import compare_protocols
from src.visualization import plot_network_topology, plot_benchmark_results, plot_protocol_comparison
from src.edge_deployment import deploy_model_on_rpi


def initialize_and_save_network():
    """Initialize the network and save the initial state to CSV."""
    print("Initializing network...")
    G, positions, node_energies = initialize_network()

    # Create node data for analysis
    node_data = [
        (positions[node][0], positions[node][1], G.nodes[node]['energy'], G.nodes[node].get('signal_strength', 0)) 
        for node in range(NUM_NODES)
    ]

    # Plot the network topology
    print("Plotting network topology...")
    plot_network_topology(positions, save_path='../outputs/network_topology.png')

    # Save simulation results to CSV for analysis
    results_df = pd.DataFrame(node_data, columns=['x', 'y', 'energy', 'signal_strength'])
    os.makedirs('../data', exist_ok=True)
    results_df.to_csv('../data/results.csv', index=False)
    print("Simulation results saved to ../data/results.csv for analysis.")

    return G, positions


def run_protocol_comparison(G, positions):
    """Run and compare different protocols."""
    print("Comparing LEACH with PCA and K-means, CRO-ULite, and LEACH without PCA and K-means...")

    # Define model save path for CRO protocol
    model_save_path = '../model/cro_model.h5'

    # Create a wrapper for cro_protocol to pass the model_save_path
    def cro_protocol_with_path(nodes, positions, transmission_range, G):
        return cro_protocol(nodes, positions, transmission_range, G, model_save_path)

    protocols = [leach_with_pca_kmeans, cro_protocol_with_path, leach_protocol]
    protocol_names = ["LEACH with PCA and K-means", "CRO-ULite (Our Model)", "LEACH without PCA and K-means"]

    comparison_results = compare_protocols(
        list(range(NUM_NODES)),
        protocols,
        protocol_names,
        max_rounds=200,
        G=G,
        positions=positions,
        transmission_range=TRANSMISSION_RANGE
    )

    return comparison_results


def plot_results(comparison_results):
    """Plot the comparison results."""
    print("Plotting comparison results...")
    plot_protocol_comparison(comparison_results, save_path='../outputs/protocol_comparison.png')

    # Plot individual benchmark results
    for protocol_name, results in comparison_results['protocol_results'].items():
        plot_benchmark_results(
            results,
            save_path=f'../outputs/{protocol_name.replace(" ", "_").lower()}_results.png'
        )

    print("All plots saved to the outputs folder.")


def deploy_model():
    """Deploy the model to Raspberry Pi Pico."""
    try:
        model_path = '../model/cro_model.h5'
        output_path = '../model/cro_model_quantized.tflite'

        print("Deploying model to Raspberry Pi Pico...")
        deploy_model_on_rpi(model_path, output_path)
        print("Model deployment completed.")
    except Exception as e:
        print(f"Model deployment failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the simulation and analysis."""
    try:
        # Initialize network and save data
        G, positions = initialize_and_save_network()

        # Run protocol comparison
        comparison_results = run_protocol_comparison(G, positions)

        # Plot results
        plot_results(comparison_results)

        # Deploy model
        deploy_model()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
