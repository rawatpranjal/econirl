"""
econirl: The StatsModels of IRL

A Python package bridging Structural Econometrics and Inverse Reinforcement Learning.
Provides economist-friendly APIs for estimating dynamic discrete choice models with
rich statistical inference.

Key Features:
- Economist-friendly terminology (utility, preferences, characteristics)
- StatsModels-style summary() output with standard errors and hypothesis tests
- Multiple estimation methods (NFXP, CCP, MaxEnt IRL)
- Gymnasium-compatible environments
- Rich visualization and counterfactual analysis

Example:
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

# Estimators
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator

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
    # Preferences
    "LinearUtility",
    # Estimators
    "NFXPEstimator",
    "CCPEstimator",
    # Datasets
    "datasets",
    # Replication
    "replication",
]
