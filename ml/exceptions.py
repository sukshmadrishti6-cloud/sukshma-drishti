"""Custom Exception hierarchy for SIH26139 ML & QML package."""


class QMLError(Exception):
    """Base exception for all errors originating from the ML/QML engine."""


class ValidationError(QMLError):
    """Raised when dataset schema, column names, or feature dimensions are invalid."""


class PreprocessingError(QMLError):
    """Raised when data transformation, scaling, or imputation fails."""


class ModelError(QMLError):
    """Raised when classical model training or evaluation fails."""


class QuantumCircuitError(QMLError):
    """Raised when quantum feature mapping, ansatz construction, or circuit execution fails."""


class QuantumBackendError(QuantumCircuitError):
    """Raised when connection to Qiskit Aer or hardware backend fails."""
