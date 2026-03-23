"""Utility/reward type adapters for benchmark harness.

Maps each estimator class to the correct utility/reward specification,
all derived from the same DDCEnvironment.
"""

from __future__ import annotations

import torch

from econirl.environments.base import DDCEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction
from econirl.preferences.linear import LinearUtility
from econirl.preferences.reward import LinearReward

# Estimator classes that need ActionDependentReward
_ACTION_DEPENDENT_NAMES = {
    "MCEIRLEstimator",
    "MaxEntIRLEstimator",
    "MaxMarginPlanningEstimator",
    "MaxMarginIRLEstimator",
    "AIRLEstimator",
    "GAILEstimator",
}

# Estimator classes that need LinearReward (state-only features)
_LINEAR_REWARD_NAMES: set[str] = set()


def project_state_features(env: DDCEnvironment) -> torch.Tensor:
    """Project 3D feature matrix to 2D state-only features.

    Extracts the keep-action (action=0) features from the 3D
    (n_states, n_actions, n_features) matrix, yielding (n_states, n_features).

    Args:
        env: DDC environment with feature_matrix property.

    Returns:
        State features of shape (n_states, n_features).
    """
    return env.feature_matrix[:, 0, :].clone()


def build_utility_for_estimator(
    env: DDCEnvironment,
    estimator_class: type,
) -> BaseUtilityFunction:
    """Build the correct utility/reward type for a given estimator.

    Args:
        env: DDC environment providing features and parameter names.
        estimator_class: The estimator class to build utility for.

    Returns:
        Utility function of the appropriate type.
    """
    class_name = estimator_class.__name__

    if class_name in _ACTION_DEPENDENT_NAMES:
        return ActionDependentReward(
            feature_matrix=env.feature_matrix,
            parameter_names=env.parameter_names,
        )

    if class_name in _LINEAR_REWARD_NAMES:
        state_features = project_state_features(env)
        return LinearReward(
            state_features=state_features,
            parameter_names=env.parameter_names,
            n_actions=env.num_actions,
        )

    # Default: LinearUtility (NFXP, CCP, TD-CCP, GLADIUS, GAIL, GCL)
    return LinearUtility.from_environment(env)
