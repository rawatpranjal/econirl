"""Test that legacy API imports emit DeprecationWarning."""
import pytest


def test_nfxp_estimator_deprecation_warning():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from econirl import NFXPEstimator
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, "NFXPEstimator import should emit DeprecationWarning"
        assert "NFXP" in str(dep_warnings[0].message)


def test_ccp_estimator_deprecation_warning():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from econirl import CCPEstimator
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, "CCPEstimator import should emit DeprecationWarning"
        assert "CCP" in str(dep_warnings[0].message)
