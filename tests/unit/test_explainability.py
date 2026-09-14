"""Unit tests for ml.explainability module and Responsible AI safeguards."""
import numpy as np
import pandas as pd
import pytest

from ml.artifacts import ModelArtifact
from ml.classical import LogisticRegressionModel, RandomForestModel, SVMModel, XGBoostModel
from ml.explainability import (
    ExplanationResult,
    LogisticRegressionExplainer,
    QuantumModelExplainer,
    SVMRBFExplainer,
    TreeEnsembleExplainer,
    explanation_service,
)
from ml.preprocessing import BiomedicalPreprocessor
from ml.quantum import QuantumFeatureReducer, QuantumSVCModel, VariationalQuantumClassifierModel


@pytest.fixture
def trained_artifacts():
    """Deterministic fixtures for testing explainers across all models."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame(
        {
            "Pregnancies": np.random.randint(0, 5, size=n),
            "Glucose": np.random.uniform(70, 160, size=n),
            "BloodPressure": np.random.uniform(60, 90, size=n),
            "SkinThickness": np.random.uniform(10, 30, size=n),
            "Insulin": np.random.uniform(15, 150, size=n),
            "BMI": np.random.uniform(18, 32, size=n),
            "DiabetesPedigreeFunction": np.random.uniform(0.1, 1.2, size=n),
            "Age": np.random.randint(21, 60, size=n),
            "Outcome": np.array([0, 1] * 15),
        }
    )
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    prep = BiomedicalPreprocessor().fit(X)
    X_trans = prep.transform(X)

    # 1. LR
    lr = LogisticRegressionModel(random_seed=42).fit(X_trans, y)
    art_lr = ModelArtifact("lr_art", "lr_v1", "Logistic Regression", "classical", "logistic_regression", lr, prep)

    # 2. RF
    rf = RandomForestModel(random_seed=42).fit(X_trans, y)
    art_rf = ModelArtifact("rf_art", "rf_v1", "Random Forest", "classical", "random_forest", rf, prep)

    # 3. XGB
    xgb = XGBoostModel(random_seed=42).fit(X_trans, y)
    art_xgb = ModelArtifact("xgb_art", "xgb_v1", "XGBoost", "classical", "xgboost", xgb, prep)

    # 4. SVM
    svm = SVMModel(random_seed=42).fit(X_trans, y)
    art_svm = ModelArtifact("svm_art", "svm_v1", "SVM", "classical", "svm", svm, prep)

    # 5. QSVC
    reducer = QuantumFeatureReducer(n_qubits=4, random_seed=42).fit(X_trans, y)
    X_q = reducer.transform(X_trans)
    qsvc = QuantumSVCModel(num_qubits=4, reps=1, shots=128, random_seed=42).fit(X_q, y)
    art_qsvc = ModelArtifact(
        "qsvc_art",
        "qsvc_v1",
        "QSVC",
        "quantum",
        "qsvc",
        qsvc,
        prep,
        quantum_reducer=reducer,
        metadata=qsvc.get_circuit_metadata(),
    )

    # 6. VQC
    vqc = VariationalQuantumClassifierModel(num_qubits=4, max_iter=5, shots=128, random_seed=42).fit(X_q, y)
    art_vqc = ModelArtifact(
        "vqc_art",
        "vqc_v1",
        "VQC",
        "quantum",
        "vqc",
        vqc,
        prep,
        quantum_reducer=reducer,
        metadata=vqc.get_circuit_metadata(),
    )

    sample_df = X.iloc[[0]].copy()
    return {
        "lr": art_lr,
        "rf": art_rf,
        "xgb": art_xgb,
        "svm": art_svm,
        "qsvc": art_qsvc,
        "vqc": art_vqc,
        "sample": sample_df,
    }


def test_logistic_regression_explainer(trained_artifacts):
    """Verify Logistic Regression additive coefficient decomposition."""
    art = trained_artifacts["lr"]
    sample = trained_artifacts["sample"]

    explainer = LogisticRegressionExplainer()
    res = explainer.explain_instance(art, sample)

    assert isinstance(res, ExplanationResult)
    assert res.status == "available"
    assert res.method == "linear_additive_attribution"
    assert len(res.features) == 11
    assert res.baseline_value is not None
    assert "Responsible AI Notice" in res.responsible_ai_disclaimer

    # Total normalized weight should sum to ~100%
    total_wt = sum(f.normalized_weight for f in res.features)
    assert pytest.approx(100.0, rel=1e-2) == total_wt

    for f in res.features:
        assert f.direction in ["positive", "negative", "neutral"]


def test_tree_ensemble_explainer(trained_artifacts):
    """Verify Tree Ensemble marginal perturbation attribution for RF and XGBoost."""
    art_rf = trained_artifacts["rf"]
    sample = trained_artifacts["sample"]

    explainer = TreeEnsembleExplainer()
    res_rf = explainer.explain_instance(art_rf, sample)

    assert res_rf.status == "available"
    assert res_rf.method == "marginal_feature_perturbation"
    assert len(res_rf.features) == 11

    art_xgb = trained_artifacts["xgb"]
    res_xgb = explainer.explain_instance(art_xgb, sample)
    assert res_xgb.status == "available"
    assert len(res_xgb.features) == 11


def test_svm_rbf_explainer(trained_artifacts):
    """Verify SVM RBF local numerical sensitivity gradient."""
    art = trained_artifacts["svm"]
    sample = trained_artifacts["sample"]

    explainer = SVMRBFExplainer()
    res = explainer.explain_instance(art, sample)

    assert res.status == "available"
    assert res.method == "local_sensitivity_gradient"
    assert len(res.features) == 11


def test_quantum_model_explainer(trained_artifacts):
    """Verify Quantum explainer exposes circuit metadata and quantum feature sensitivity."""
    art_qsvc = trained_artifacts["qsvc"]
    sample = trained_artifacts["sample"]

    explainer = QuantumModelExplainer()
    res = explainer.explain_instance(art_qsvc, sample)

    assert res.model_family == "quantum"
    assert res.status == "partial"
    assert res.method == "quantum_feature_sensitivity_projection"
    assert res.quantum_metadata is not None
    assert "qubit_count" in res.quantum_metadata
    assert len(res.features) == 11
    assert len(res.limitations) >= 2


def test_explanation_service_dispatch(trained_artifacts):
    """Verify ExplanationService dispatches correctly across model types."""
    sample = trained_artifacts["sample"]

    for m_key, art in trained_artifacts.items():
        if m_key == "sample":
            continue
        res = explanation_service.explain(art, sample, top_k=4)
        assert res.model_name == art.model_name
        assert len(res.features) <= 4
        assert res.responsible_ai_disclaimer != ""


def test_svm_rbf_explainer_global(trained_artifacts):
    """Verify SVM RBF global permutation feature importance."""
    art = trained_artifacts["svm"]
    explainer = SVMRBFExplainer()
    res = explainer.explain_global(art)

    assert res.status == "available"
    assert res.method == "permutation_importance"
    assert len(res.features) == 11

    # Feature names must match preprocessor engineered feature names exactly
    expected_names = art.preprocessor.get_feature_names()
    actual_names = [f.name for f in res.features]
    assert set(actual_names) == set(expected_names)

    # Weights must sum to ~100%
    total_wt = sum(f.normalized_weight for f in res.features)
    assert pytest.approx(100.0, rel=1e-1) == total_wt

    # Metadata must be populated
    assert res.metadata["scoring_metric"] == "roc_auc"
    assert res.metadata["random_seed"] == 42
    assert res.metadata["sample_count"] == 154
    assert len(res.limitations) >= 2


def test_svm_permutation_importance_no_leakage(trained_artifacts):
    """Verify that permutation importance does NOT refit model or preprocessor."""
    art = trained_artifacts["svm"]
    preprocessor = art.preprocessor
    medians_before = dict(preprocessor.impute_medians_)
    scaler_mean_before = preprocessor.scaler_.mean_.copy() if preprocessor.scaler_ is not None else None

    explainer = SVMRBFExplainer()
    _ = explainer.explain_global(art)

    # Preprocessor medians and scaler must be strictly unmodified
    assert preprocessor.impute_medians_ == medians_before
    if scaler_mean_before is not None:
        np.testing.assert_array_equal(preprocessor.scaler_.mean_, scaler_mean_before)

