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

__version__ = "0.1.0"

# Core types
from econirl.core.types import DDCProblem, Panel, Trajectory

# Environments
from econirl.environments.rust_bus import RustBusEnvironment

# Preferences
from econirl.preferences.linear import LinearUtility

# Legacy Estimators (for backward compatibility)
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator

# Sklearn-style Estimators (recommended)
from econirl.estimators import NFXP, CCP

# Sklearn-style Utilities
from econirl.utilities import Utility, LinearCost, make_utility

# Sklearn-style Transition Estimator
from econirl.transitions import TransitionEstimator

# Datasets
from econirl import datasets

# Replication
from econirl import replication

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
    # Replication
    "replication",
]
