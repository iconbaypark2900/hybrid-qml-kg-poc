# benchmarking/scalability_sim.py

"""
Scalability Simulation for Hybrid QML-KG System

Generates the algorithmic scaling projection plot comparing:
- Classical approach: O(N²) runtime for link prediction
- Quantum approach: O(log N) runtime based on quantum algorithm complexity

This demonstrates the theoretical quantum advantage as knowledge graph size increases.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_scaling_projection(
    output_path: str = "docs/scaling_projection.png",
    min_entities: int = 100,
    max_entities: int = 100000,
    classical_base_time: float = 0.001,  # ms per pairwise operation
    quantum_base_time: float = 10.0,     # ms base overhead
    quantum_scaling_factor: float = 5.0  # multiplier for log scaling
):
    """
    Generate and save the scaling projection plot.
    
    Args:
        output_path: Path to save the PNG file
        min_entities: Minimum number of entities to plot
        max_entities: Maximum number of entities to plot
        classical_base_time: Base time per classical operation (ms)
        quantum_base_time: Base overhead for quantum approach (ms)
        quantum_scaling_factor: Scaling factor for quantum log complexity
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Entity counts (logarithmically spaced)
    N = np.logspace(
        np.log10(min_entities), 
        np.log10(max_entities), 
        num=50, 
        dtype=int
    )
    N = np.unique(N)  # Remove duplicates from rounding
    
    # Classical runtime: O(N²) for pairwise link prediction
    # In real KG link prediction, you often need to evaluate many entity pairs
    classical_runtime = classical_base_time * (N ** 2)
    
    # Quantum runtime: O(log N) based on quantum algorithm complexity
    # This reflects the theoretical advantage of quantum algorithms for certain KG tasks
    quantum_runtime = quantum_base_time + quantum_scaling_factor * np.log2(N)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(N, classical_runtime, 'o-', 
             label='Classical Approach\nO(N²) Complexity', 
             linewidth=3, markersize=6, color='#2E86AB')
    plt.plot(N, quantum_runtime, 's-', 
             label='Quantum Approach\nO(log N) Complexity', 
             linewidth=3, markersize=6, color='#A23B72')
    
    plt.xlabel('Number of Entities in Knowledge Graph (N)', fontsize=14, fontweight='bold')
    plt.ylabel('Projected Runtime (milliseconds)', fontsize=14, fontweight='bold')
    plt.title('Algorithmic Scaling Advantage: Quantum vs Classical\nfor Knowledge Graph Link Prediction', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, which='both')
    plt.xscale('log')
    plt.yscale('log')
    
    # Find and mark crossover point
    crossover_indices = np.where(quantum_runtime < classical_runtime)[0]
    if len(crossover_indices) > 0:
        crossover_idx = crossover_indices[0]
        crossover_n = N[crossover_idx]
        crossover_time = quantum_runtime[crossover_idx]
        
        plt.axvline(x=crossover_n, color='red', linestyle='--', alpha=0.7, linewidth=2)
        plt.annotate(f'Crossover Point\n~{crossover_n:,} entities', 
                    xy=(crossover_n, crossover_time),
                    xytext=(crossover_n * 3, crossover_time * 10),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Add explanatory text box
    explanation = (
        "Note: This shows theoretical algorithmic complexity.\n"
        "Current NISQ hardware may be slower in absolute terms,\n"
        "but quantum approach scales more favorably as KG size grows."
    )
    plt.text(0.02, 0.02, explanation, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Scaling projection plot saved to {output_path}")
    logger.info(f"Crossover point at approximately {crossover_n:,} entities")
    
    return output_path

def run_scalability_simulation():
    """Run the full scalability simulation and generate the plot."""
    logger.info("Running scalability simulation...")
    
    plot_path = generate_scaling_projection(
        output_path="docs/scaling_projection.png",
        min_entities=100,
        max_entities=1000000,  # Up to 1M entities
        classical_base_time=0.001,
        quantum_base_time=50.0,   # Higher base overhead for realism
        quantum_scaling_factor=10.0
    )
    
    logger.info("Scalability simulation completed successfully!")
    return plot_path

# Example usage
if __name__ == "__main__":
    run_scalability_simulation()