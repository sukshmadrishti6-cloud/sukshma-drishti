"""Parameterized Ansatz Construction Module for Variational Quantum Circuits."""
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal

from ml.exceptions import QuantumCircuitError
from ml.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_ANSATZ_TYPES = {
    "realamplitudes": RealAmplitudes,
    "efficientsu2": EfficientSU2,
    "twolocal": TwoLocal,
}


def create_ansatz(
    name: str = "RealAmplitudes",
    num_qubits: int = 4,
    reps: int = 2,
    entanglement: str = "full",
) -> QuantumCircuit:
    """Constructs a real Qiskit parameterized variational ansatz circuit.

    Args:
        name: Name of the ansatz ('RealAmplitudes', 'EfficientSU2', 'TwoLocal').
        num_qubits: Number of qubits (must equal number of reduced quantum features).
        reps: Number of parameterized variational layers.
        entanglement: Entanglement topology ('full', 'linear', 'circular').

    Returns:
        QuantumCircuit: Real Qiskit parameterized QuantumCircuit instance.
    """
    if num_qubits < 1:
        raise QuantumCircuitError(f"num_qubits must be >= 1, got {num_qubits}.")

    if reps < 1:
        raise QuantumCircuitError(f"reps must be >= 1, got {reps}.")

    name_clean = name.strip().lower()
    if name_clean not in SUPPORTED_ANSATZ_TYPES:
        raise QuantumCircuitError(
            f"Unsupported ansatz type: '{name}'. Supported: {list(SUPPORTED_ANSATZ_TYPES.keys())}"
        )

    try:
        if name_clean == "realamplitudes":
            ansatz = RealAmplitudes(
                num_qubits=num_qubits, reps=reps, entanglement=entanglement
            )
        elif name_clean == "efficientsu2":
            ansatz = EfficientSU2(
                num_qubits=num_qubits, reps=reps, entanglement=entanglement
            )
        elif name_clean == "twolocal":
            ansatz = TwoLocal(
                num_qubits=num_qubits,
                rotation_blocks=["ry", "rz"],
                entanglement_blocks="cz",
                reps=reps,
                entanglement=entanglement,
            )
        else:
            ansatz = RealAmplitudes(
                num_qubits=num_qubits, reps=reps, entanglement=entanglement
            )

        logger.info(
            f"Constructed Qiskit ansatz '{name}' ({num_qubits} qubits, {reps} reps, "
            f"entanglement='{entanglement}', parameters={ansatz.num_parameters}, "
            f"depth={ansatz.depth()})"
        )
        return ansatz
    except Exception as e:
        raise QuantumCircuitError(f"Failed to create ansatz '{name}': {e!s}")
