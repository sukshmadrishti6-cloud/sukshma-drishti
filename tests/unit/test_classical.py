"""Unit tests for ml.classical module."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.classical import (
    ClassicalModelFactory,
    LogisticRegressionModel,
    RandomForestModel,
    SVMModel,
    XGBoostModel,
    load_model,
    save_model,
)
from ml.exceptions import ModelError, ValidationError
from ml.preprocessing import BiomedicalPreprocessor, split_data


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
    return df


@pytest.fixture
def processed_data(mock_pima_df: pd.DataFrame):
    """Fixture producing preprocessed feature matrix and targets."""
    X_train, X_test, y_train, y_test = split_data(
        mock_pima_df, target_column="Outcome", test_size=0.2, random_seed=42
    )
    preprocessor = BiomedicalPreprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    return X_train_trans, X_test_trans, y_train, y_test


def test_classical_package_import():
    """Verify classical module package structure imports cleanly."""
    import ml.classical
    assert hasattr(ml.classical, "ClassicalModelFactory")
    assert hasattr(ml.classical, "LogisticRegressionModel")
    assert hasattr(ml.classical, "SVMModel")
    assert hasattr(ml.classical, "RandomForestModel")
    assert hasattr(ml.classical, "XGBoostModel")


def test_factory_model_instantiation():
    """Verify ClassicalModelFactory creates all 4 classical model families."""
    for key in ["logistic_regression", "svm", "random_forest", "xgboost"]:
        model = ClassicalModelFactory.create_model(key, random_seed=42)
        assert model is not None
        assert model.random_seed == 42


def test_factory_invalid_model_name():
    """Verify factory raises ValidationError for unknown model names."""
    with pytest.raises(ValidationError, match="Unsupported classical model name"):
        ClassicalModelFactory.create_model("invalid_model")


def test_logistic_regression_training_and_prediction(processed_data):
    """Verify LogisticRegressionModel trains, predicts labels, and predicts probabilities."""
    X_train, X_test, y_train, _y_test = processed_data
    model = LogisticRegressionModel(random_seed=42)
    model.fit(X_train, y_train)

    assert model.is_fitted_ is True

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    assert preds.shape == (len(X_test),)
    assert set(preds).issubset({0, 1})
    assert probs.shape == (len(X_test), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_svm_rbf_kernel_training_and_prediction(processed_data):
    """Verify SVMModel uses RBF kernel, trains, predicts labels and probabilities."""
    X_train, X_test, y_train, _y_test = processed_data
    model = SVMModel(params={"C": 1.5}, random_seed=42)
    model.fit(X_train, y_train)

    assert model.is_fitted_ is True
    assert model.estimator_.kernel == "rbf"
    assert model.estimator_.probability is True

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    assert preds.shape == (len(X_test),)
    assert set(preds).issubset({0, 1})
    assert probs.shape == (len(X_test), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_random_forest_training_and_prediction(processed_data):
    """Verify RandomForestModel trains and produces valid predictions."""
    X_train, X_test, y_train, _y_test = processed_data
    model = RandomForestModel(params={"n_estimators": 50, "max_depth": 5}, random_seed=42)
    model.fit(X_train, y_train)

    assert model.is_fitted_ is True

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    assert preds.shape == (len(X_test),)
    assert set(preds).issubset({0, 1})
    assert probs.shape == (len(X_test), 2)


def test_xgboost_training_and_prediction(processed_data):
    """Verify XGBoostModel trains and produces valid predictions."""
    X_train, X_test, y_train, _y_test = processed_data
    model = XGBoostModel(params={"n_estimators": 40, "learning_rate": 0.1}, random_seed=42)
    model.fit(X_train, y_train)

    assert model.is_fitted_ is True

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    assert preds.shape == (len(X_test),)
    assert set(preds).issubset({0, 1})
    assert probs.shape == (len(X_test), 2)


def test_unfitted_model_raises_error(processed_data):
    """Verify calling predict on an unfitted model raises ModelError."""
    _, X_test, _, _ = processed_data
    model = LogisticRegressionModel()
    with pytest.raises(ModelError, match="must be fitted"):
        model.predict(X_test)


def test_dimension_mismatch_raises_error(processed_data):
    """Verify feature dimension mismatch raises ValidationError."""
    X_train, _, y_train, _ = processed_data
    model = LogisticRegressionModel()
    model.fit(X_train, y_train)

    bad_X = np.random.rand(10, 5)  # 5 features instead of 8
    with pytest.raises(ValidationError, match="Feature dimension mismatch"):
        model.predict(bad_X)


def test_nan_input_raises_validation_error(processed_data):
    """Verify NaN in input matrix raises ValidationError."""
    X_train, _, y_train, _ = processed_data
    X_nan = X_train.copy()
    X_nan[0, 0] = np.nan

    model = LogisticRegressionModel()
    with pytest.raises(ValidationError, match="contains NaN or infinite values"):
        model.fit(X_nan, y_train)


def test_model_persistence(tmp_path: Path, processed_data):
    """Verify save_model and load_model serialize and restore predicting model correctly."""
    X_train, X_test, y_train, _ = processed_data
    model = RandomForestModel(random_seed=42)
    model.fit(X_train, y_train)

    original_preds = model.predict(X_test)
    original_probs = model.predict_proba(X_test)

    save_file = tmp_path / "rf_model.joblib"
    save_model(model, save_file)

    loaded_model = load_model(save_file)

    loaded_preds = loaded_model.predict(X_test)
    loaded_probs = loaded_model.predict_proba(X_test)

    np.testing.assert_array_equal(original_preds, loaded_preds)
    np.testing.assert_allclose(original_probs, loaded_probs)
