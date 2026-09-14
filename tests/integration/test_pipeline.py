"""Integration test for end-to-end pipeline import and setup."""


def test_full_package_integration_import():
    """Verify all top-level packages import cleanly together."""
    import ml
    import ml.classical
    import ml.data
    import ml.evaluation
    import ml.explainability
    import ml.preprocessing
    import ml.quantum

    assert ml.__version__ == "0.1.0"
