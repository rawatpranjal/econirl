"""Estimation algorithms for dynamic discrete choice models."""

from econirl.estimation.base import Estimator, EstimationResult
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.maxent_irl import MaxEntIRLEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimation.max_margin_planning import MaxMarginPlanningEstimator, MMPConfig
from econirl.estimation.max_margin_irl import MaxMarginIRLEstimator
from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
from econirl.estimation.gcl import GCLEstimator, GCLConfig
from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig
from econirl.estimation.transitions import (
    estimate_transition_probs,
    estimate_transition_probs_by_group,
)

# Adversarial methods
from econirl.estimation.adversarial import (
    GAILEstimator,
    GAILConfig,
    AIRLEstimator,
    AIRLConfig,
    TabularDiscriminator,
    LinearDiscriminator,
)

# Neural network IRL
from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig

# Bayesian IRL
from econirl.estimation.bayesian_irl import BayesianIRLEstimator

# Deep MaxEnt IRL
from econirl.estimation.deep_maxent_irl import DeepMaxEntIRLEstimator

# NNES
from econirl.estimation.nnes import NNESEstimator

# f-IRL
from econirl.estimation.f_irl import FIRLEstimator

# SEES
from econirl.estimation.sees import SEESEstimator

__all__ = [
    # Base
    "Estimator",
    "EstimationResult",
    # Forward estimation
    "NFXPEstimator",
    "CCPEstimator",
    # IRL methods
    "MaxEntIRLEstimator",
    "MCEIRLEstimator",
    "MCEIRLConfig",
    "MaxMarginPlanningEstimator",
    "MMPConfig",
    "MaxMarginIRLEstimator",
    "GCLEstimator",
    "GCLConfig",
    # Supervised baseline
    "BehavioralCloningEstimator",
    # TD-CCP Neural
    "TDCCPEstimator",
    "TDCCPConfig",
    # Adversarial methods
    "GAILEstimator",
    "GAILConfig",
    "AIRLEstimator",
    "AIRLConfig",
    "TabularDiscriminator",
    "LinearDiscriminator",
    # Neural network IRL
    "GLADIUSEstimator",
    "GLADIUSConfig",
    # Bayesian IRL
    "BayesianIRLEstimator",
    # Deep MaxEnt IRL
    "DeepMaxEntIRLEstimator",
    # NNES
    "NNESEstimator",
    # f-IRL
    "FIRLEstimator",
    # SEES
    "SEESEstimator",
    # Utilities
    "estimate_transition_probs",
    "estimate_transition_probs_by_group",
]
