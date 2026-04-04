import econirl._jax_config  # noqa: F401  # enable float64 before any JAX usage

"""
econirl: The StatsModels of IRL

A Python package bridging Structural Econometrics and Inverse Reinforcement Learning.
Provides economist-friendly APIs for estimating dynamic discrete choice models with
rich statistical inference.

Key Features:
- Sklearn-style API with fit()/predict() interface
- StatsModels-style summary() output with standard errors and hypothesis tests
- Multiple estimation methods (NFXP, CCP)
- Gymnasium-compatible environments
- Rich visualization and counterfactual analysis

Sklearn-style API (recommended):
    >>> from econirl import NFXP, CCP, LinearCost, TransitionEstimator
    >>> import pandas as pd
    >>>
    >>> # Load your data as a DataFrame
    >>> df = pd.read_csv("bus_data.csv")
    >>>
    >>> # First stage: estimate transition probabilities
    >>> trans = TransitionEstimator(n_states=90)
    >>> trans.fit(data=df, state="mileage_bin", id="bus_id", action="replaced")
    >>>
    >>> # Second stage: estimate utility parameters
    >>> model = NFXP(n_states=90, discount=0.9999, utility=LinearCost())
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id",
    ...           transitions=trans.matrix_)
    >>> print(model.summary())
    >>>
    >>> # Or use the faster CCP estimator
    >>> model_ccp = CCP(n_states=90, discount=0.9999)
    >>> model_ccp.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>> print(model_ccp.summary())

Legacy API (deprecated, for backward compatibility):
    >>> from econirl import RustBusEnvironment, LinearUtility, NFXPEstimator
    >>> from econirl.simulation import simulate_panel
    >>>
    >>> env = RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0)
    >>> panel = simulate_panel(env, n_individuals=500, n_periods=100)
    >>> utility = LinearUtility(feature_matrix=env.feature_matrix)
    >>> result = NFXPEstimator().estimate(panel, utility, env.problem_spec)
    >>> print(result.summary())
"""

__version__ = "0.0.3"

# Core types
from econirl.core.types import DDCProblem, Panel, Trajectory, TrajectoryPanel
from econirl.core.reward_spec import RewardSpec
from econirl.core.sufficient_stats import SufficientStats

# Environments
from econirl.environments.rust_bus import RustBusEnvironment

# Preferences
from econirl.preferences.linear import LinearUtility

# Legacy Estimators — handled by __getattr__ with deprecation warnings

# Sklearn-style Estimators (JAX backend)
from econirl.estimators import NFXP, CCP, MaxEntIRL, MaxMarginIRL, MCEIRL, NNES, SEES, TDCCP
from econirl.estimators import GLADIUS, NeuralGLADIUS
from econirl.estimators import AIRL, NeuralAIRL
from econirl.estimation import IQLearnEstimator as IQLearn
from econirl.estimators import MCEIRLNeural

# Sklearn-style Utilities
from econirl.utilities import Utility, LinearCost, make_utility

# Sklearn-style Transition Estimator
from econirl.transitions import TransitionEstimator

# Datasets
from econirl import datasets

# Preprocessing
try:
    from econirl import preprocessing
except ImportError:
    pass

# Replication
try:
    from econirl import replication
except ImportError:
    pass

__all__ = [
    # Version
    "__version__",
    # Core types
    "DDCProblem",
    "Panel",
    "Trajectory",
    # Environments
    "RustBusEnvironment",
    # Sklearn-style Estimators (recommended)
    "NFXP",
    "CCP",
    "MaxEntIRL",
    "MaxMarginIRL",
    "MCEIRL",
    "NNES",
    "SEES",
    "TDCCP",
    "GLADIUS",
    "AIRL",
    "IQLearn",
    "NeuralGLADIUS",
    "NeuralAIRL",
    "MCEIRLNeural",
    # Core types (new)
    "RewardSpec",
    "TrajectoryPanel",
    "SufficientStats",
    # Sklearn-style Utilities
    "Utility",
    "LinearCost",
    "make_utility",
    # Sklearn-style Transition Estimator
    "TransitionEstimator",
    # Legacy API (for backward compatibility)
    "LinearUtility",
    "NFXPEstimator",
    "CCPEstimator",
    # Datasets
    "datasets",
    # Preprocessing
    "preprocessing",
    # Replication
    "replication",
]

_DEPRECATED_LEGACY = {
    "NFXPEstimator": ("econirl.estimation.nfxp", "NFXPEstimator", "NFXP"),
    "CCPEstimator": ("econirl.estimation.ccp", "CCPEstimator", "CCP"),
}


def __getattr__(name: str):
    if name in _DEPRECATED_LEGACY:
        module_path, class_name, replacement = _DEPRECATED_LEGACY[name]
        import warnings
        import importlib
        warnings.warn(
            f"{name} is deprecated. Use econirl.{replacement} (sklearn-style API) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    raise AttributeError(f"module 'econirl' has no attribute {name!r}")
