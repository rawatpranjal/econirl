"""Estimator protocol for econirl.

All sklearn-style estimator wrappers implement this protocol.
This is a protocol (structural typing), NOT a base class.
Estimators are independent modules that happen to satisfy the same interface.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class EstimatorProtocol(Protocol):
    """Every econirl estimator implements this contract.

    This protocol defines the unified interface for all estimators,
    whether they are structural (NFXP, CCP), IRL (MCE-IRL), or neural (NNES, TD-CCP).

    Attributes (set after fit):
        params_: Estimated parameters as {name: value} dict
        se_: Standard errors as {name: value} dict, or NaN values with warning for estimators without inference
        pvalues_: P-values as {name: value} dict
        policy_: Learned policy π(a|s) as ndarray of shape (S, A)
        value_: Value function V(s) as ndarray of shape (S,)
    """

    params_: dict[str, float] | None
    se_: dict[str, float] | None
    pvalues_: dict[str, float] | None
    policy_: np.ndarray | None
    value_: np.ndarray | None

    def fit(self, data, state: str, action: str, id: str, **kwargs) -> "EstimatorProtocol":
        """Fit the estimator to data."""
        ...

    def summary(self) -> str:
        """Return StatsModels-style summary table."""
        ...

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Predict choice probabilities for given states."""
        ...

    def conf_int(self, alpha: float = 0.05) -> dict:
        """Compute confidence intervals for parameters."""
        ...
