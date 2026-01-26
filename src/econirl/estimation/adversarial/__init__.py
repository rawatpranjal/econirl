"""Adversarial imitation learning methods for discrete choice models.

This module implements adversarial IRL algorithms adapted for tabular MDPs:
- GAIL (Generative Adversarial Imitation Learning) - Ho & Ermon 2016
- AIRL (Adversarial IRL) - Fu et al. 2018

These methods learn reward functions by training a discriminator to distinguish
expert demonstrations from policy-generated behavior.
"""

from econirl.estimation.adversarial.discriminator import (
    TabularDiscriminator,
    LinearDiscriminator,
)
from econirl.estimation.adversarial.gail import GAILEstimator, GAILConfig
from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig

__all__ = [
    "TabularDiscriminator",
    "LinearDiscriminator",
    "GAILEstimator",
    "GAILConfig",
    "AIRLEstimator",
    "AIRLConfig",
]
