"""Dataset Loading and Schema Validation Module for SIH26139."""
import hashlib
from pathlib import Path

import pandas as pd

from ml.exceptions import ValidationError
from ml.logger import get_logger
from ml.preprocessing.pipeline import RAW_FEATURE_NAMES, TARGET_COLUMN

logger = get_logger(__name__)

# Expected Pima Indians Diabetes Dataset columns derived from canonical schema
EXPECTED_PIMA_COLUMNS = list(RAW_FEATURE_NAMES) + [TARGET_COLUMN]

# Standard column alias mapping for flexible CSV headers
COLUMN_ALIASES = {
    "DiabetesPedigree": "DiabetesPedigreeFunction",
}


def compute_file_hash(filepath: str | Path) -> str:
    """Computes SHA-256 hash of a file for dataset version tracking."""
    path = Path(filepath)
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class DataLoader:
    """Loads and validates biomedical datasets from CSV files."""

    def __init__(
        self,
        filepath: str | Path,
        target_column: str = "Outcome",
        expected_columns: list[str] | None = None,
    ):
        self.filepath = Path(filepath)
        self.target_column = target_column
        self.expected_columns = expected_columns or EXPECTED_PIMA_COLUMNS
        self.logger = logger

    def load(self) -> tuple[pd.DataFrame, str]:
        """Loads dataset CSV, validates schema, and returns DataFrame and SHA-256 hash."""
        if not self.filepath.exists():
            raise ValidationError(f"Dataset file not found at path: '{self.filepath}'")

        if not self.filepath.is_file() or self.filepath.suffix.lower() != ".csv":
            raise ValidationError(
                f"Invalid dataset file format. Expected CSV file at: '{self.filepath}'"
            )

        try:
            df = pd.read_csv(self.filepath)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            raise ValidationError(f"Failed to parse CSV dataset at '{self.filepath}': {e!s}")

        if df.empty:
            raise ValidationError(f"Dataset file at '{self.filepath}' is empty.")

        # Normalize column aliases if present
        df = df.rename(columns=COLUMN_ALIASES)

        self.validate_schema(df)
        file_hash = compute_file_hash(self.filepath)

        self.logger.info(
            f"Successfully loaded dataset '{self.filepath.name}' "
            f"({len(df)} rows, {len(df.columns)} columns, hash: {file_hash[:12]})"
        )
        return df, file_hash

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validates columns, target column existence, numeric feature types, and non-empty rows."""
        missing_cols = [col for col in self.expected_columns if col not in df.columns]
        if missing_cols:
            raise ValidationError(
                f"Dataset missing required columns: {missing_cols}. Found columns: {list(df.columns)}"
            )

        if self.target_column not in df.columns:
            raise ValidationError(f"Target column '{self.target_column}' missing from dataset.")

        # Verify numeric feature types
        feature_cols = [col for col in df.columns if col != self.target_column]
        non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            raise ValidationError(f"Non-numeric feature columns detected: {non_numeric}")
