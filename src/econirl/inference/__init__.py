"""Statistical inference for estimation results."""

from econirl.inference.results import EstimationSummary
from econirl.inference.standard_errors import compute_standard_errors
from econirl.inference.identification import check_identification
from econirl.inference.hypothesis_tests import (
    likelihood_ratio_test,
    score_test,
    vuong_test,
)
from econirl.inference.fit_metrics import (
    brier_score,
    kl_divergence,
    efron_pseudo_r_squared,
    ccp_consistency_test,
)
from econirl.inference.reward_comparison import (
    epic_distance,
    detect_reward_shaping,
)
from econirl.inference.etable import etable

__all__ = [
    "EstimationSummary",
    "compute_standard_errors",
    "check_identification",
    # Hypothesis tests
    "likelihood_ratio_test",
    "score_test",
    "vuong_test",
    # Fit metrics
    "brier_score",
    "kl_divergence",
    "efron_pseudo_r_squared",
    "ccp_consistency_test",
    # Reward comparison
    "epic_distance",
    "detect_reward_shaping",
    # Tables
    "etable",
]
