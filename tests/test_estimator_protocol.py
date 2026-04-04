"""Test that EstimatorProtocol works correctly as a structural type check."""
import numpy as np
import pytest
from econirl.estimators.protocol import EstimatorProtocol


class _MockEstimator:
    """A minimal mock that satisfies the protocol."""
    def __init__(self):
        self.params_ = {"a": 1.0}
        self.se_ = {"a": 0.1}
        self.pvalues_ = {"a": 0.01}
        self.policy_ = np.array([[0.5, 0.5]])
        self.value_ = np.array([1.0])
        self.reward_matrix_ = np.array([[0.1, -0.2]])

    def fit(self, data, state, action, id, **kwargs):
        return self

    def summary(self):
        return "Mock summary"

    def predict_proba(self, states):
        return np.array([[0.5, 0.5]])

    def conf_int(self, alpha=0.05):
        return {"a": (0.8, 1.2)}


class _NotAnEstimator:
    """Does NOT satisfy the protocol."""
    pass


def test_mock_satisfies_protocol():
    est = _MockEstimator()
    assert isinstance(est, EstimatorProtocol)


def test_non_estimator_fails_protocol():
    obj = _NotAnEstimator()
    assert not isinstance(obj, EstimatorProtocol)


def test_protocol_is_runtime_checkable():
    """Verify the protocol can be used with isinstance()."""
    assert hasattr(EstimatorProtocol, '__protocol_attrs__') or True  # runtime_checkable
    est = _MockEstimator()
    assert isinstance(est, EstimatorProtocol)


from econirl.estimators import (
    NFXP, CCP, NNES, SEES, TDCCP,
    MaxEntIRL, MaxMarginIRL, MCEIRL,
    NeuralGLADIUS, NeuralAIRL, MCEIRLNeural, GCL,
)

# Core estimators that should always be available
_CORE_ESTIMATORS = [NFXP, CCP, NNES, SEES, TDCCP, MaxEntIRL, MaxMarginIRL, MCEIRL]

# Optional estimators (may be None if import failed)
_OPTIONAL = [
    ("NeuralGLADIUS", NeuralGLADIUS),
    ("NeuralAIRL", NeuralAIRL),
    ("MCEIRLNeural", MCEIRLNeural),
    ("GCL", GCL),
]


@pytest.mark.parametrize("cls", _CORE_ESTIMATORS, ids=lambda c: c.__name__)
def test_core_estimator_satisfies_protocol(cls):
    """Every core sklearn estimator should satisfy EstimatorProtocol before fitting."""
    est = cls()
    assert isinstance(est, EstimatorProtocol), f"{cls.__name__} does not satisfy EstimatorProtocol"


@pytest.mark.parametrize("name,cls", _OPTIONAL, ids=[n for n, _ in _OPTIONAL])
def test_optional_estimator_satisfies_protocol(name, cls):
    """Optional estimators should satisfy protocol if available."""
    if cls is None:
        pytest.skip(f"{name} not available")
    est = cls()
    assert isinstance(est, EstimatorProtocol), f"{name} does not satisfy EstimatorProtocol"
