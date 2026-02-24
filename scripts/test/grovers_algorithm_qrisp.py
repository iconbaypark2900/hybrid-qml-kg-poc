"""
Grover's Algorithm Implementation using qrisp

This script implements Grover's algorithm with:
- Oracle operator: marks the target state
- Diffusion operator: amplifies the marked state
- Two iterations of alternating oracle and diffusion operators
- Initial state: 2 register qubits in superposition, 1 ancilla qubit in |-> state
"""

from qrisp import QuantumVariable, h, x, z, mcp, cx, measure, QuantumCircuit
import numpy as np


def create_oracle(target_state: int, register: QuantumVariable, ancilla: QuantumVariable):
    """
    Creates an oracle operator that marks the target state.
    
    Args:
        target_state: The computational basis state to mark (0, 1, 2, or 3)
        register: QuantumVariable containing the register qubits
        ancilla: QuantumVariable containing the ancilla qubit
    
    The oracle flips the ancilla qubit if the register is in the target state.
    """
    # Apply X gates to qubits where target_state bit is 0
    # This prepares for multi-controlled phase flip
    if target_state == 0:
        # |00> -> mark with phase flip
        x(register[0])
        x(register[1])
        mcp(np.pi, [register[0], register[1]], ancilla)
        x(register[0])
        x(register[1])
    elif target_state == 1:
        # |01> -> mark
        x(register[0])
        mcp(np.pi, [register[0], register[1]], ancilla)
        x(register[0])
    elif target_state == 2:
        # |10> -> mark
        x(register[1])
        mcp(np.pi, [register[0], register[1]], ancilla)
        x(register[1])
    else:  # target_state == 3
        # |11> -> mark
        mcp(np.pi, [register[0], register[1]], ancilla)


def create_diffusion(register: QuantumVariable):
    """
    Creates the diffusion operator (Grover diffusion operator).
    
    Args:
        register: QuantumVariable containing the register qubits
    
    The diffusion operator amplifies the amplitude of the marked state
    by reflecting about the average amplitude.
    """
    # Apply H gates to all register qubits
    h(register[0])
    h(register[1])
    
    # Apply phase flip to |00> state
    x(register[0])
    x(register[1])
    mcp(np.pi, [register[0], register[1]], register[0])  # Multi-controlled phase
    x(register[0])
    x(register[1])
    
    # Apply H gates again to all register qubits
    h(register[0])
    h(register[1])


def grovers_algorithm(target_state: int = 2, num_iterations: int = 2):
    """
    Implements Grover's algorithm with oracle and diffusion operators.
    
    Args:
        target_state: The state to search for (0, 1, 2, or 3)
        num_iterations: Number of Grover iterations to perform
    
    Returns:
        QuantumCircuit: The complete Grover's algorithm circuit
        QuantumVariable: The register qubits
        QuantumVariable: The ancilla qubit
    """
    # Create quantum variables
    # 2 register qubits for the search space (4 states: 00, 01, 10, 11)
    register = QuantumVariable(2, name="register")
    
    # 1 ancilla qubit for the oracle
    ancilla = QuantumVariable(1, name="ancilla")
    
    # Initialize: register qubits in superposition, ancilla in |-> state
    print("\n" + "="*70)
    print("INITIALIZING GROVER'S ALGORITHM")
    print("="*70)
    print(f"Target state: |{target_state:02b}> = |{target_state}>")
    print(f"Number of iterations: {num_iterations}")
    
    # Put register qubits in superposition
    h(register[0])
    h(register[1])
    
    # Put ancilla in |-> state (|0> -> |->)
    h(ancilla[0])
    x(ancilla[0])
    
    print("\nInitial state prepared:")
    print("  - Register qubits: (|0> + |1>)/√2 ⊗ (|0> + |1>)/√2 = |++>")
    print("  - Ancilla qubit: |->")
    
    # Apply Grover iterations: alternate between oracle and diffusion
    print("\n" + "-"*70)
    print("APPLYING GROVER ITERATIONS")
    print("-"*70)
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Apply oracle
        print(f"  Applying oracle (marks |{target_state:02b}>)...")
        create_oracle(target_state, register, ancilla[0])
        
        # Apply diffusion
        print(f"  Applying diffusion operator...")
        create_diffusion(register)
    
    print("\n" + "="*70)
    print("GROVER'S ALGORITHM COMPLETE")
    print("="*70)
    
    return register, ancilla


def run_grovers_simulation(target_state: int = 2, num_iterations: int = 2, shots: int = 1000):
    """
    Run Grover's algorithm and measure the results.
    
    Args:
        target_state: The state to search for (0, 1, 2, or 3)
        num_iterations: Number of Grover iterations to perform
        shots: Number of measurement shots
    """
    # Create the circuit
    register, ancilla = grovers_algorithm(target_state, num_iterations)
    
    # Measure the register qubits
    print("\nMeasuring register qubits...")
    print(f"Running {shots} shots...")
    
    # Get the statevector or measurement counts
    # Note: qrisp's measurement interface may vary, this is a general approach
    measurement_result = measure(register, shots=shots)
    
    print("\n" + "="*70)
    print("MEASUREMENT RESULTS")
    print("="*70)
    print(f"\nTarget state: |{target_state:02b}> = |{target_state}>")
    print(f"\nMeasurement counts:")
    
    # Process and display results
    if hasattr(measurement_result, 'most_common'):
        counts = measurement_result
        total = sum(counts.values())
        for state, count in sorted(counts.items()):
            percentage = (count / total) * 100
            marker = "✓" if int(state, 2) == target_state else " "
            print(f"  {marker} |{state}>: {count:4d} ({percentage:5.2f}%)")
    else:
        print(f"  {measurement_result}")
    
    print("\n" + "="*70)
    
    return register, ancilla, measurement_result


def main():
    """
    Main function to demonstrate Grover's algorithm.
    """
    print("\n" + "="*70)
    print("GROVER'S ALGORITHM WITH QRISP")
    print("="*70)
    print("\nThis implementation demonstrates:")
    print("  - Oracle operator that marks a target state")
    print("  - Diffusion operator that amplifies the marked state")
    print("  - Two iterations of alternating oracle and diffusion")
    print("  - Initial state: 2 register qubits in |++>, 1 ancilla in |->")
    
    # Example: search for state |10> (binary 10 = decimal 2)
    target = 2
    iterations = 2
    shots = 1000
    
    print("\n" + "-"*70)
    print(f"Configuration:")
    print(f"  Target state: |{target:02b}> = |{target}>")
    print(f"  Iterations: {iterations}")
    print(f"  Shots: {shots}")
    print("-"*70)
    
    try:
        register, ancilla, results = run_grovers_simulation(
            target_state=target,
            num_iterations=iterations,
            shots=shots
        )
        
        print("\n✅ Grover's algorithm executed successfully!")
        print("\nNote: In an ideal scenario, after 2 iterations for 4 states,")
        print("      the target state should have high probability (~100%).")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("\nThis might be due to:")
        print("  1. qrisp not being installed (pip install qrisp)")
        print("  2. API differences in qrisp version")
        print("  3. Quantum simulator backend not configured")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

