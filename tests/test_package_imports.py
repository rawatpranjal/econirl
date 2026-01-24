"""Tests for package-level imports.

Verifies that the sklearn-style API is properly exported at the top level.
"""

import pytest


class TestSklearnStyleImports:
    """Test that sklearn-style classes can be imported from econirl."""

    def test_import_nfxp(self) -> None:
        """Can import NFXP from econirl."""
        from econirl import NFXP

        # Verify it's the right class
        assert hasattr(NFXP, "fit")
        assert hasattr(NFXP, "summary")

    def test_import_ccp(self) -> None:
        """Can import CCP from econirl."""
        from econirl import CCP

        # Verify it's the right class
        assert hasattr(CCP, "fit")
        assert hasattr(CCP, "summary")

    def test_import_utilities(self) -> None:
        """Can import LinearCost and Utility from econirl."""
        from econirl import LinearCost, Utility

        # Verify LinearCost is a subclass of Utility
        assert issubclass(LinearCost, Utility)

        # Verify LinearCost has the expected interface
        assert hasattr(LinearCost, "n_params")
        assert hasattr(LinearCost, "param_names")
        assert hasattr(LinearCost, "__call__")

    def test_import_make_utility(self) -> None:
        """Can import make_utility factory function from econirl."""
        from econirl import make_utility

        # Verify it's callable
        assert callable(make_utility)

    def test_import_transition_estimator(self) -> None:
        """Can import TransitionEstimator from econirl."""
        from econirl import TransitionEstimator

        # Verify it has the expected interface
        assert hasattr(TransitionEstimator, "fit")
        assert hasattr(TransitionEstimator, "summary")


class TestBackwardCompatibility:
    """Test that old API still works for backward compatibility."""

    def test_import_nfxp_estimator(self) -> None:
        """Can still import NFXPEstimator (old name)."""
        from econirl import NFXPEstimator

        assert hasattr(NFXPEstimator, "estimate")

    def test_import_ccp_estimator(self) -> None:
        """Can still import CCPEstimator (old name)."""
        from econirl import CCPEstimator

        assert hasattr(CCPEstimator, "estimate")

    def test_import_linear_utility(self) -> None:
        """Can still import LinearUtility (old name)."""
        from econirl import LinearUtility

        # Verify it works
        assert hasattr(LinearUtility, "compute")


class TestAllExports:
    """Test that __all__ is properly updated."""

    def test_all_contains_new_exports(self) -> None:
        """__all__ includes all new sklearn-style exports."""
        import econirl

        expected_exports = [
            "NFXP",
            "CCP",
            "LinearCost",
            "Utility",
            "make_utility",
            "TransitionEstimator",
        ]

        for name in expected_exports:
            assert name in econirl.__all__, f"{name} not in __all__"

    def test_all_contains_old_exports(self) -> None:
        """__all__ still includes old exports for backward compatibility."""
        import econirl

        expected_exports = [
            "NFXPEstimator",
            "CCPEstimator",
            "LinearUtility",
        ]

        for name in expected_exports:
            assert name in econirl.__all__, f"{name} not in __all__"
