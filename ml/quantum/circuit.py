"""Quantum Circuit and Feature Map Construction Module for SIH26139."""
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit_aer import AerSimulator

from ml.exceptions import QuantumBackendError, QuantumCircuitError
from ml.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FEATURE_MAPS = {
    "zzfeaturemap": ZZFeatureMap,
    "zfeaturemap": ZFeatureMap,
    "paulifeaturemap": PauliFeatureMap,
}


def create_feature_map(
    name: str = "ZZFeatureMap",
    num_qubits: int = 4,
    reps: int = 2,
    entanglement: str = "linear",
) -> QuantumCircuit:
    """Constructs a real Qiskit feature map quantum circuit for quantum kernel computation.

    Args:
        name: Name of the feature map ('ZZFeatureMap', 'ZFeatureMap', 'PauliFeatureMap').
        num_qubits: Number of qubits (must equal number of reduced features).
        reps: Number of circuit layer repetitions.
        entanglement: Entanglement topology ('linear', 'full', 'circular').

    Returns:
        QuantumCircuit: Real Qiskit QuantumCircuit instance.
    """
    if num_qubits < 1:
        raise QuantumCircuitError(f"num_qubits must be >= 1, got {num_qubits}.")

    if reps < 1:
        raise QuantumCircuitError(f"reps must be >= 1, got {reps}.")

    name_clean = name.strip().lower()
    if name_clean not in SUPPORTED_FEATURE_MAPS:
        raise QuantumCircuitError(
            f"Unsupported feature map: '{name}'. Supported: {list(SUPPORTED_FEATURE_MAPS.keys())}"
        )

    try:
        if name_clean == "zfeaturemap":
            feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=reps)
        elif name_clean == "zzfeaturemap":
            feature_map = ZZFeatureMap(
                feature_dimension=num_qubits, reps=reps, entanglement=entanglement
            )
        elif name_clean == "paulifeaturemap":
            feature_map = PauliFeatureMap(
                feature_dimension=num_qubits, reps=reps, entanglement=entanglement
            )
        else:
            feature_map = ZZFeatureMap(
                feature_dimension=num_qubits, reps=reps, entanglement=entanglement
            )

        logger.info(
            f"Constructed Qiskit feature map '{name}' ({num_qubits} qubits, {reps} reps, "
            f"entanglement='{entanglement}', depth={feature_map.depth()})"
        )
        return feature_map
    except Exception as e:
        raise QuantumCircuitError(f"Failed to create feature map '{name}': {e!s}")


def get_quantum_backend(
    name: str = "aer_simulator",
    shots: int = 1024,
    random_seed: int = 42,
) -> AerSimulator:
    """Returns a configured local Qiskit Aer simulator backend.

    Args:
        name: Backend name ('aer_simulator').
        shots: Default number of simulation shots.
        random_seed: Random seed for deterministic simulation.

    Returns:
        AerSimulator: Configured Qiskit Aer simulator backend.
    """
    try:
        backend = AerSimulator(seed_simulator=random_seed)
        backend.set_options(shots=shots)
        logger.info(f"Initialized local Aer simulator backend (shots={shots}, seed={random_seed})")
        return backend
    except Exception as e:
        raise QuantumBackendError(f"Failed to initialize Aer simulator backend: {e!s}")
