"""Quantum Kernel Construction Module for SIH26139."""
from typing import Any

from qiskit import QuantumCircuit
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from ml.exceptions import QuantumCircuitError
from ml.logger import get_logger

logger = get_logger(__name__)


def create_quantum_kernel(
    feature_map: QuantumCircuit,
    fidelity: Any | None = None,
) -> FidelityQuantumKernel:
    """Constructs a real Qiskit Machine Learning FidelityQuantumKernel.

    Args:
        feature_map: Real Qiskit QuantumCircuit representing the quantum feature map.
        fidelity: Optional custom fidelity primitive.

    Returns:
        FidelityQuantumKernel: Configured quantum kernel instance.
    """
    if feature_map is None or not isinstance(feature_map, QuantumCircuit):
        raise QuantumCircuitError(
            "Valid Qiskit QuantumCircuit feature_map is required to create a Quantum Kernel."
        )

    try:
        kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
        logger.info(
            f"Created FidelityQuantumKernel with feature map '{feature_map.name}' "
            f"({feature_map.num_qubits} qubits)"
        )
        return kernel
    except Exception as e:
        raise QuantumCircuitError(f"Failed to instantiate FidelityQuantumKernel: {e!s}")
