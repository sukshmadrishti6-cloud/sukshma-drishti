"""Unit tests for ml.preprocessing module."""
import numpy as np
import pandas as pd
import pytest

from ml.exceptions import PreprocessingError
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
    # Inject invalid zero values
    df.loc[5, "Glucose"] = 0.0
    df.loc[12, "BloodPressure"] = 0.0
    df.loc[20, "BMI"] = 0.0
    return df


def test_preprocessing_package_import():
    """Verify preprocessing module package structure imports cleanly."""
    import ml.preprocessing
    assert hasattr(ml.preprocessing, "BiomedicalPreprocessor")
    assert hasattr(ml.preprocessing, "split_data")


def test_stratified_train_test_split(mock_pima_df: pd.DataFrame):
    """Verify train/test split size, stratification, and target column removal."""
    X_train, X_test, y_train, y_test = split_data(
        mock_pima_df, target_column="Outcome", test_size=0.2, random_seed=42, stratify=True
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

    # Ensure target column is NOT in X_train or X_test
    assert "Outcome" not in X_train.columns
    assert "Outcome" not in X_test.columns

    # Verify stratified target ratio
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_train_test_split_reproducibility(mock_pima_df: pd.DataFrame):
    """Verify seed=42 produces identical splits."""
    X_train1, X_test1, _, _ = split_data(mock_pima_df, random_seed=42)
    X_train2, X_test2, _, _ = split_data(mock_pima_df, random_seed=42)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)


def test_invalid_zero_handling_and_imputation(mock_pima_df: pd.DataFrame):
    """Verify invalid zeros in Glucose/BP/BMI are converted to NaN and imputed."""
    X_train, X_test, _, _ = split_data(mock_pima_df, random_seed=42)

    preprocessor = BiomedicalPreprocessor(scale_features=False)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Confirm no NaN or Inf values remain
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()
    assert not np.isinf(X_train_trans).any()
    assert not np.isinf(X_test_trans).any()


def test_data_leakage_prevention(mock_pima_df: pd.DataFrame):
    """CRITICAL TEST: Verify scaler parameters are fit STRICTLY on X_train only."""
    X_train, X_test, _, _ = split_data(mock_pima_df, random_seed=42)

    # Create outlier in X_test that should NOT affect X_train fitted mean/scaler
    X_test_polluted = X_test.copy()
    X_test_polluted.loc[X_test_polluted.index[0], "Glucose"] = 999999.0

    preprocessor1 = BiomedicalPreprocessor(scale_features=True)
    preprocessor1.fit(X_train)

    fitted_mean = preprocessor1.scaler_.mean_.copy()

    # Preprocessor 2 fits on X_train only (ignoring X_test_polluted)
    preprocessor2 = BiomedicalPreprocessor(scale_features=True)
    preprocessor2.fit(X_train)

    # Scaler mean must be identical regardless of X_test content
    np.testing.assert_array_equal(fitted_mean, preprocessor2.scaler_.mean_)

    # Transforming polluted test set should use train mean
    X_test_trans = preprocessor1.transform(X_test_polluted)
    assert X_test_trans.shape == (20, 11)


def test_unfitted_preprocessor_raises_error(mock_pima_df: pd.DataFrame):
    """Verify transform() on unfitted preprocessor raises PreprocessingError."""
    X_train, _, _, _ = split_data(mock_pima_df)
    preprocessor = BiomedicalPreprocessor()
    with pytest.raises(PreprocessingError, match="must be fitted"):
        preprocessor.transform(X_train)


def test_canonical_schemas():
    """Verify raw (8), engineered (3), and classical (11) schemas and exact ordering."""
    from ml.preprocessing import (
        RAW_FEATURE_NAMES,
        ENGINEERED_FEATURE_NAMES,
        CLASSICAL_FEATURE_NAMES,
        TARGET_COLUMN,
        FEATURES_WITH_INVALID_ZEROS,
    )

    assert len(RAW_FEATURE_NAMES) == 8
    assert RAW_FEATURE_NAMES == [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    assert TARGET_COLUMN == "Outcome"

    assert len(ENGINEERED_FEATURE_NAMES) == 3
    assert ENGINEERED_FEATURE_NAMES == ["Glucose_BMI", "Insulin_Glucose", "AgeGroup"]

    assert len(CLASSICAL_FEATURE_NAMES) == 11
    assert CLASSICAL_FEATURE_NAMES == RAW_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES
    assert len(set(CLASSICAL_FEATURE_NAMES)) == 11  # No duplicates

    assert set(FEATURES_WITH_INVALID_ZEROS) == {
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    }
    assert "Pregnancies" not in FEATURES_WITH_INVALID_ZEROS


def test_invalid_zero_handling_exact_columns():
    """Verify zeros in invalid_zero_columns become NaN, while legitimate Pregnancies zeros remain 0."""
    from ml.preprocessing import BiomedicalPreprocessor

    sample_df = pd.DataFrame(
        {
            "Pregnancies": [0.0, 3.0, 5.0],
            "Glucose": [0.0, 120.0, 140.0],
            "BloodPressure": [0.0, 70.0, 80.0],
            "SkinThickness": [0.0, 25.0, 30.0],
            "Insulin": [0.0, 100.0, 150.0],
            "BMI": [0.0, 28.0, 32.0],
            "DiabetesPedigreeFunction": [0.2, 0.4, 0.6],
            "Age": [22.0, 35.0, 48.0],
        }
    )

    preproc = BiomedicalPreprocessor(scale_features=False)
    preproc.fit(sample_df)

    # Medians computed on non-zero values
    assert preproc.impute_medians_["Glucose"] == 130.0  # median of [120, 140]
    assert preproc.impute_medians_["BloodPressure"] == 75.0
    assert preproc.impute_medians_["SkinThickness"] == 27.5
    assert preproc.impute_medians_["Insulin"] == 125.0
    assert preproc.impute_medians_["BMI"] == 30.0

    # Pregnancies zero is legitimate and kept
    assert preproc.impute_medians_["Pregnancies"] == 3.0

    trans = preproc.transform(sample_df)
    # Row 0 Pregnancies was 0.0 and should remain 0.0
    assert trans[0, 0] == 0.0
    # Row 0 Glucose was 0.0 (invalid) and should become 130.0
    assert trans[0, 1] == 130.0


def test_missing_value_imputation_train_only():
    """Verify test set missing values use train-fitted medians and cannot modify them."""
    train_df = pd.DataFrame(
        {
            "Pregnancies": [1.0, 2.0, 3.0],
            "Glucose": [100.0, 120.0, 140.0],
            "BloodPressure": [70.0, 80.0, 90.0],
            "SkinThickness": [20.0, 25.0, 30.0],
            "Insulin": [50.0, 100.0, 150.0],
            "BMI": [25.0, 30.0, 35.0],
            "DiabetesPedigreeFunction": [0.2, 0.4, 0.6],
            "Age": [25.0, 35.0, 45.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "Pregnancies": [0.0],
            "Glucose": [0.0],  # Invalid zero -> will be imputed with train median (120.0)
            "BloodPressure": [80.0],
            "SkinThickness": [25.0],
            "Insulin": [100.0],
            "BMI": [30.0],
            "DiabetesPedigreeFunction": [0.3],
            "Age": [30.0],
        }
    )

    preproc = BiomedicalPreprocessor(scale_features=False)
    preproc.fit(train_df)
    saved_medians = dict(preproc.impute_medians_)

    trans_test = preproc.transform(test_df)

    # Train medians must not be altered by test set transform
    assert preproc.impute_medians_ == saved_medians
    # Test sample Glucose was 0.0 -> imputed to train median 120.0
    assert trans_test[0, 1] == 120.0


def test_iqr_outlier_bounds_train_only():
    """Verify IQR outlier bounds are computed strictly on train and clip test outliers."""
    train_df = pd.DataFrame(
        {
            "Pregnancies": [1.0, 2.0, 2.0, 3.0, 4.0],
            "Glucose": [80.0, 90.0, 100.0, 110.0, 120.0],
            "BloodPressure": [60.0, 70.0, 70.0, 80.0, 90.0],
            "SkinThickness": [20.0, 20.0, 25.0, 30.0, 30.0],
            "Insulin": [50.0, 60.0, 80.0, 90.0, 100.0],
            "BMI": [20.0, 22.0, 25.0, 28.0, 30.0],
            "DiabetesPedigreeFunction": [0.1, 0.2, 0.3, 0.4, 0.5],
            "Age": [22.0, 25.0, 30.0, 35.0, 40.0],
        }
    )

    preproc = BiomedicalPreprocessor(scale_features=False)
    preproc.fit(train_df)

    glucose_bounds = preproc.iqr_bounds_["Glucose"]
    # Q1 = 90, Q3 = 110, IQR = 20 -> lower = 90 - 30 = 60, upper = 110 + 30 = 140
    assert glucose_bounds["lower"] == 60.0
    assert glucose_bounds["upper"] == 140.0

    # Outlier in test set: 500.0 should be clipped to upper bound (140.0)
    extreme_test = train_df.iloc[[0]].copy()
    extreme_test["Glucose"] = 500.0

    trans = preproc.transform(extreme_test)
    assert trans[0, 1] == 140.0


def test_feature_engineering_exact_formulas():
    """Verify exact deterministic formulas for Glucose_BMI, Insulin_Glucose, and AgeGroup."""
    from ml.preprocessing import engineer_features

    df = pd.DataFrame(
        {
            "Glucose": [100.0, 150.0, 200.0, 120.0],
            "BMI": [25.0, 30.0, 20.0, 35.0],
            "Insulin": [50.0, 150.0, 0.0, 300.0],
            "Age": [25.0, 35.0, 50.0, 70.0],
        }
    )

    out = engineer_features(df)

    # 1. Glucose_BMI = Glucose * BMI
    assert out.loc[0, "Glucose_BMI"] == 100.0 * 25.0  # 2500.0
    assert out.loc[1, "Glucose_BMI"] == 150.0 * 30.0  # 4500.0

    # 2. Insulin_Glucose = Insulin / (Glucose + 1e-9)
    np.testing.assert_allclose(out.loc[0, "Insulin_Glucose"], 50.0 / (100.0 + 1e-9), rtol=1e-6)
    np.testing.assert_allclose(out.loc[1, "Insulin_Glucose"], 150.0 / (150.0 + 1e-9), rtol=1e-6)
    np.testing.assert_allclose(out.loc[2, "Insulin_Glucose"], 0.0, atol=1e-9)

    # 3. AgeGroup binning: <=30 -> 0, 31-45 -> 1, 46-60 -> 2, >60 -> 3
    assert out.loc[0, "AgeGroup"] == 0.0  # Age 25
    assert out.loc[1, "AgeGroup"] == 1.0  # Age 35
    assert out.loc[2, "AgeGroup"] == 2.0  # Age 50
    assert out.loc[3, "AgeGroup"] == 3.0  # Age 70

    # Edge cases: Age > 100 or Age = 30
    edge_df = pd.DataFrame(
        {
            "Glucose": [100.0, 100.0],
            "BMI": [25.0, 25.0],
            "Insulin": [50.0, 50.0],
            "Age": [30.0, 105.0],
        }
    )
    edge_out = engineer_features(edge_df)
    assert edge_out.loc[0, "AgeGroup"] == 0.0  # 30 is upper bound of bin 0
    assert edge_out.loc[1, "AgeGroup"] == 3.0  # >60 is bin 3, no NaN


def test_feature_ordering_stability():
    """Verify output columns always match exact 11 classical feature order regardless of input column order."""
    from ml.preprocessing import CLASSICAL_FEATURE_NAMES

    # Create shuffled column DataFrame
    shuffled_cols = [
        "Age",
        "BMI",
        "Glucose",
        "Pregnancies",
        "Insulin",
        "BloodPressure",
        "DiabetesPedigreeFunction",
        "SkinThickness",
    ]
    df = pd.DataFrame({col: [np.random.uniform(10, 100)] for col in shuffled_cols})

    preproc = BiomedicalPreprocessor(scale_features=False)
    # Fit on canonical order
    canonical_df = df[[col for col in CLASSICAL_FEATURE_NAMES if col in df.columns]]
    preproc.fit(canonical_df)

    # Transform with shuffled order input
    trans_shuffled = preproc.transform(df)

    assert preproc.get_feature_names() == CLASSICAL_FEATURE_NAMES
    assert trans_shuffled.shape == (1, 11)


def test_quantum_feature_reducer_k_selection():
    """Verify QuantumFeatureReducer SelectKBest works for k=4, 6, 8 with angle scaling in [-pi, pi]."""
    from ml.quantum import QuantumFeatureReducer

    np.random.seed(42)
    X = np.random.randn(50, 11)
    y = np.random.choice([0, 1], size=50)

    for k in [4, 6, 8]:
        reducer = QuantumFeatureReducer(n_qubits=k, random_seed=42)
        X_q = reducer.fit_transform(X, y)

        assert X_q.shape == (50, k)
        assert np.all(X_q >= -np.pi - 1e-5)
        assert np.all(X_q <= np.pi + 1e-5)

        stats = reducer.get_selection_stats()
        assert stats["n_qubits"] == k
        assert stats["method"] == "SelectKBest(mutual_info_classif)"
        assert len(stats["selected_indices"]) == k
        assert len(stats["selected_feature_names"]) == k


def test_anti_double_preprocessing_guard(mock_pima_df: pd.DataFrame):
    """Verify ModelArtifact predict/predict_proba guards against double preprocessing."""
    from ml.artifacts import ModelArtifact
    from ml.classical import ClassicalModelFactory

    X_train, X_test, y_train, y_test = split_data(mock_pima_df, random_seed=42)

    preproc = BiomedicalPreprocessor()
    X_train_trans = preproc.fit_transform(X_train)

    model = ClassicalModelFactory.create("lr", random_seed=42)
    model.fit(X_train_trans, y_train)

    artifact = ModelArtifact(
        artifact_id="lr_test",
        model_id="lr_test",
        model_name="Logistic Regression",
        model_family="classical",
        model_type="logistic_regression",
        model=model,
        preprocessor=preproc,
    )

    # 1. Calling predict with RAW DataFrame (8 features)
    preds_raw = artifact.predict(X_test)
    assert preds_raw.shape == (len(X_test),)

    # 2. Calling predict with ALREADY PREPROCESSED array (11 features)
    X_test_trans = preproc.transform(X_test)
    preds_trans = artifact.predict(X_test_trans)
    assert preds_trans.shape == (len(X_test),)

    # Both predictions must match exactly
    np.testing.assert_array_equal(preds_raw, preds_trans)

