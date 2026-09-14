"""Unit tests for ml.data module (Loader and Validator)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data import DataLoader, DataValidator
from ml.exceptions import QMLError, ValidationError


@pytest.fixture
def mock_pima_df() -> pd.DataFrame:
    """Fixture producing synthetic Pima-like DataFrame for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "Pregnancies": np.random.randint(0, 10, size=n),
            "Glucose": np.random.uniform(70, 180, size=n),
            "BloodPressure": np.random.uniform(60, 90, size=n),
            "SkinThickness": np.random.uniform(10, 40, size=n),
            "Insulin": np.random.uniform(15, 200, size=n),
            "BMI": np.random.uniform(18, 35, size=n),
            "DiabetesPedigreeFunction": np.random.uniform(0.1, 1.5, size=n),
            "Age": np.random.randint(21, 65, size=n),
            "Outcome": np.random.choice([0, 1], size=n, p=[0.65, 0.35]),
        }
    )
    # Inject some invalid zeros into Glucose, BP, BMI
    df.loc[5, "Glucose"] = 0
    df.loc[12, "BloodPressure"] = 0
    df.loc[20, "BMI"] = 0
    return df


def test_data_package_import():
    """Verify data module package structure imports cleanly."""
    import ml.data
    assert hasattr(ml.data, "DataLoader")
    assert hasattr(ml.data, "DataValidator")


def test_custom_exception_hierarchy():
    """Verify custom exception hierarchy."""
    err = ValidationError("Invalid column")
    assert isinstance(err, QMLError)


def test_data_loader_file_not_found():
    """Verify DataLoader raises ValidationError on non-existent file."""
    loader = DataLoader("data/raw/non_existent_file.csv")
    with pytest.raises(ValidationError, match="Dataset file not found"):
        loader.load()


def test_data_loader_invalid_schema(tmp_path: Path):
    """Verify DataLoader raises ValidationError when required columns are missing."""
    bad_csv = tmp_path / "bad.csv"
    bad_df = pd.DataFrame({"Pregnancies": [1, 2], "Outcome": [0, 1]})
    bad_df.to_csv(bad_csv, index=False)

    loader = DataLoader(bad_csv)
    with pytest.raises(ValidationError, match="missing required columns"):
        loader.load()


def test_data_loader_empty_file_rejection(tmp_path: Path):
    """Verify DataLoader raises ValidationError on empty CSV files."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")

    loader = DataLoader(empty_csv)
    with pytest.raises(ValidationError):
        loader.load()


def test_data_validator_valid_dataset(mock_pima_df: pd.DataFrame):
    """Verify DataValidator inspects valid dataset successfully."""
    validator = DataValidator(target_column="Outcome")
    report = validator.inspect_quality(mock_pima_df)

    assert report["is_valid"] is True
    assert report["total_rows"] == 100
    assert report["feature_count"] == 8
    assert "Outcome" not in report["feature_names"]
    assert report["target_distribution"][0] > 0
    assert report["target_distribution"][1] > 0


def test_data_validator_invalid_non_binary_target(mock_pima_df: pd.DataFrame):
    """Verify DataValidator rejects non-binary target labels."""
    df_bad = mock_pima_df.copy()
    df_bad.loc[0, "Outcome"] = 2  # Invalid target label

    validator = DataValidator(target_column="Outcome")
    with pytest.raises(ValidationError, match="non-binary values"):
        validator.inspect_quality(df_bad)


def test_data_validator_lacks_class_diversity(mock_pima_df: pd.DataFrame):
    """Verify DataValidator rejects single-class datasets."""
    df_single = mock_pima_df.copy()
    df_single["Outcome"] = 1  # Only class 1

    validator = DataValidator(target_column="Outcome")
    with pytest.raises(ValidationError, match="lacks class diversity"):
        validator.inspect_quality(df_single)


def test_data_validator_duplicate_detection(mock_pima_df: pd.DataFrame):
    """Verify duplicate row counting."""
    df_dup = pd.concat([mock_pima_df, mock_pima_df.iloc[[0, 1]]], ignore_index=True)
    validator = DataValidator(target_column="Outcome")
    report = validator.inspect_quality(df_dup)

    assert report["duplicate_rows"] == 2
