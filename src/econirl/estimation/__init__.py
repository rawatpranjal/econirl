"""Estimation algorithms for dynamic discrete choice models.

Production estimators (12):
    NFXP, MPEC, CCP, MCE-IRL, TD-CCP, NNES, SEES, GLADIUS, IQ-Learn, AIRL, f-IRL, BC

Contrib estimators (moved to econirl.contrib):
    MaxEnt IRL, Deep MaxEnt, Max Margin, Max Margin IRL, GAIL, GCL, BIRL, IQ-Learn
"""

import warnings

from econirl.estimation.base import Estimator, EstimationResult
from econirl.estimation.categories import (
    EstimatorCategory,
    ProblemCapabilities,
    ESTIMATOR_REGISTRY,
    get_estimators_by_category,
    get_estimators_with_capability,
    get_category,
    get_capabilities,
)

# Structural
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.estimation.ccp import CCPEstimator
NFXP = NFXPEstimator
MPEC = MPECEstimator
CCP = CCPEstimator

# Entropy IRL
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
MCEIRL = MCEIRLEstimator

# Structural approximation
from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig
from econirl.estimation.nnes import NNESEstimator, NNESNFXPEstimator
from econirl.estimation.sees import SEESEstimator
TDCCP = TDCCPEstimator
NNES = NNESEstimator
SEES = SEESEstimator

# Q-learning IRL
from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig
from econirl.estimation.iq_learn import IQLearnEstimator, IQLearnConfig
IQLearn = IQLearnEstimator
GLADIUS = GLADIUSEstimator

# Adversarial IRL
from econirl.estimation.adversarial import (
    AIRLEstimator,
    AIRLConfig,
    AIRLHetEstimator,
    AIRLHetConfig,
    TabularDiscriminator,
    LinearDiscriminator,
)
AIRL = AIRLEstimator
AIRLHet = AIRLHetEstimator

# Distribution-matching IRL
from econirl.estimation.f_irl import FIRLEstimator
FIRL = FIRLEstimator

# Imitation baseline
from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
BC = BehavioralCloningEstimator

# Utilities
from econirl.estimation.transitions import (
    estimate_transition_probs,
    estimate_transition_probs_by_group,
)

__all__ = [
    # Base
    "Estimator",
    "EstimationResult",
    # Structural
    "NFXP",
    "NFXPEstimator",
    "MPEC",
    "MPECEstimator",
    "MPECConfig",
    "CCP",
    "CCPEstimator",
    # Entropy IRL
    "MCEIRL",
    "MCEIRLEstimator",
    "MCEIRLConfig",
    # Structural approximation
    "TDCCP",
    "TDCCPEstimator",
    "TDCCPConfig",
    "NNES",
    "NNESEstimator",
    "NNESNFXPEstimator",
    "SEES",
    "SEESEstimator",
    # Q-learning IRL
    "GLADIUS",
    "GLADIUSEstimator",
    "GLADIUSConfig",
    "IQLearn",
    "IQLearnEstimator",
    "IQLearnConfig",
    # Adversarial IRL
    "AIRL",
    "AIRLEstimator",
    "AIRLConfig",
    "AIRLHet",
    "AIRLHetEstimator",
    "AIRLHetConfig",
    "TabularDiscriminator",
    "LinearDiscriminator",
    # Distribution-matching IRL
    "FIRL",
    "FIRLEstimator",
    # Imitation baseline
    "BC",
    "BehavioralCloningEstimator",
    # Taxonomy
    "EstimatorCategory",
    "ProblemCapabilities",
    "ESTIMATOR_REGISTRY",
    "get_estimators_by_category",
    "get_estimators_with_capability",
    "get_category",
    "get_capabilities",
    # Utilities
    "estimate_transition_probs",
    "estimate_transition_probs_by_group",
]

# Backward-compatibility shim: moved estimators import from contrib with warning
_MOVED_TO_CONTRIB = {
    "MaxEntIRLEstimator": "maxent_irl",
    "DeepMaxEntIRLEstimator": "deep_maxent_irl",
    "MaxMarginIRLEstimator": "max_margin_irl",
    "MaxMarginPlanningEstimator": "max_margin_planning",
    "MMPConfig": "max_margin_planning",
    "GCLEstimator": "gcl",
    "GCLConfig": "gcl",
    "BayesianIRLEstimator": "bayesian_irl",
    "GAILEstimator": "gail",
    "GAILConfig": "gail",
}


def __getattr__(name: str):
    if name in _MOVED_TO_CONTRIB:
        mod_name = _MOVED_TO_CONTRIB[name]
        warnings.warn(
            f"{name} has moved to econirl.contrib.{mod_name}. "
            f"Update your import to: from econirl.contrib.{mod_name} import {name}",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib
        mod = importlib.import_module(f"econirl.contrib.{mod_name}")
        return getattr(mod, name)
    raise AttributeError(f"module 'econirl.estimation' has no attribute {name!r}")
