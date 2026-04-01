"""Preprocessing utilities for DDC estimation.

This module provides transparent preprocessing functions for:
- State discretization (continuous to discrete bins)
- Panel data validation
- Next-state computation
- Running normalization for feature standardization
"""

from econirl.preprocessing.discretization import (
    discretize_state,
    discretize_mileage,
)
from econirl.preprocessing.validation import (
    check_panel_structure,
    compute_next_states,
    PanelValidationResult,
)
from econirl.preprocessing.running_norm import RunningNorm

__all__ = [
    "discretize_state",
    "discretize_mileage",
    "check_panel_structure",
    "compute_next_states",
    "PanelValidationResult",
    "RunningNorm",
]
