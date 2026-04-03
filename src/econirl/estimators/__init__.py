"""Sklearn-style estimators for dynamic discrete choice models.

This module provides high-level estimators with a scikit-learn style API:
- NFXP: Nested Fixed Point estimator (Rust 1987, 1988)
- NNES: Neural Network Estimation of Structural models (Nguyen 2025)
- CCP: Conditional Choice Probability estimator (Hotz-Miller 1993, NPL)
- MaxEntIRL: Maximum Entropy IRL estimator (Ziebart 2008)
- MaxMarginIRL: Maximum Margin IRL estimator (Abbeel & Ng 2004)
- MCEIRL: Maximum Causal Entropy IRL estimator (Ziebart 2010)

Example:
    >>> from econirl.estimators import NFXP, NNES, CCP, MaxEntIRL, MaxMarginIRL, MCEIRL
    >>> import pandas as pd
    >>>
    >>> # Load your data
    >>> df = pd.read_csv("bus_data.csv")
    >>>
    >>> # Create and fit the NFXP estimator
    >>> model = NFXP(n_states=90, discount=0.9999)
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Or use the neural NNES estimator (avoids inner fixed-point)
    >>> model_nnes = NNES(n_states=90, discount=0.9999)
    >>> model_nnes.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Or use the faster CCP estimator (Hotz-Miller)
    >>> model_ccp = CCP(n_states=90, discount=0.9999)
    >>> model_ccp.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # For IRL: recover reward from expert demonstrations
    >>> model_irl = MaxEntIRL(n_states=90, n_actions=2, discount=0.99)
    >>> model_irl.fit(data=df, state="state", action="action", id="agent_id")
    >>> print(model_irl.reward_)  # Recovered reward function
    >>>
    >>> # MCE IRL (Maximum Causal Entropy)
    >>> model_mce = MCEIRL(n_states=90, discount=0.99)
    >>> model_mce.fit(data=df, state="state", action="action", id="agent_id")
    >>> print(model_mce.reward_)  # Recovered reward function
    >>>
    >>> # Access results (same interface)
    >>> print(model.params_)
    >>> print(model.summary())
"""

from econirl.estimators.ccp import CCP
from econirl.estimators.max_margin_irl import MaxMarginIRL
from econirl.estimators.maxent_irl import MaxEntIRL
from econirl.estimators.mce_irl import MCEIRL
from econirl.estimators.nfxp import NFXP
from econirl.estimators.nnes import NNES
from econirl.estimators.sees import SEES
from econirl.estimators.tdccp import TDCCP
from econirl.estimators.protocol import EstimatorProtocol

try:
    from econirl.estimators.gcl import GCL
except ImportError:
    GCL = None

try:
    from econirl.estimators.neural_gladius import NeuralGLADIUS
except ImportError:
    NeuralGLADIUS = None

GLADIUS = NeuralGLADIUS

try:
    from econirl.estimators.neural_airl import NeuralAIRL
except ImportError:
    NeuralAIRL = None

AIRL = NeuralAIRL

try:
    from econirl.estimators.mceirl_neural import MCEIRLNeural
except ImportError:
    MCEIRLNeural = None

__all__ = [
    "NFXP",
    "NNES",
    "CCP",
    "SEES",
    "TDCCP",
    "MaxEntIRL",
    "MaxMarginIRL",
    "MCEIRL",
    "GLADIUS",
    "AIRL",
    "NeuralGLADIUS",
    "NeuralAIRL",
    "MCEIRLNeural",
    "EstimatorProtocol",
]
