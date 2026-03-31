"""Linear utility specification.

This module implements the most common utility specification in discrete
choice models: linear-in-parameters utility.

    U(s, a; θ) = θ · φ(s, a)

where φ(s, a) is a vector of observable characteristics (features) and
θ is the parameter vector to be estimated.

This specification is used in:
- Rust (1987) bus engine replacement
- Most logit/probit discrete choice models
- Maximum entropy IRL (Ziebart 2008)
"""

from __future__ import annotations

import torch

from econirl.preferences.base import BaseUtilityFunction


class LinearUtility(BaseUtilityFunction):
    """Linear-in-parameters utility function.

    Implements the utility specification:
        U(s, a; θ) = θ · φ(s, a) = Σ_k θ_k φ_k(s, a)

    where:
    - θ ∈ R^K is the parameter vector (preferences to estimate)
    - φ: S × A → R^K maps state-action pairs to feature vectors

    This is the workhorse specification for structural estimation.
    The gradient ∂U/∂θ = φ(s, a) is constant in θ, making optimization
    relatively straightforward.

    Attributes:
        feature_matrix: Tensor of shape (num_states, num_actions, num_features)

    Example:
        >>> # Create from environment
        >>> env = RustBusEnvironment()
        >>> utility = LinearUtility.from_environment(env)
        >>>
        >>> # Or manually specify
        >>> features = torch.randn(100, 2, 5)  # 100 states, 2 actions, 5 features
        >>> utility = LinearUtility(features, parameter_names=["a", "b", "c", "d", "e"])
        >>>
        >>> # Compute utility
        >>> theta = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> U = utility.compute(theta)  # shape (100, 2)
    """

    def __init__(
        self,
        feature_matrix: torch.Tensor,
        parameter_names: list[str] | None = None,
        anchor_action: int | None = None,
    ):
        """Initialize the linear utility function.

        Args:
            feature_matrix: Feature tensor of shape (num_states, num_actions, num_features)
                           where feature_matrix[s, a, k] = φ_k(s, a)
            parameter_names: Names for each parameter. If None, uses "θ_0", "θ_1", etc.
            anchor_action: Action to use for normalization. If set, the utility of
                          this action will be subtracted from all actions (for identification).
        """
        if feature_matrix.ndim != 3:
            raise ValueError(
                f"feature_matrix must be 3D (states, actions, features), "
                f"got shape {feature_matrix.shape}"
            )

        num_states, num_actions, num_features = feature_matrix.shape

        if parameter_names is None:
            parameter_names = [f"θ_{k}" for k in range(num_features)]
        elif len(parameter_names) != num_features:
            raise ValueError(
                f"parameter_names must have {num_features} elements, "
                f"got {len(parameter_names)}"
            )

        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            parameter_names=parameter_names,
            anchor_action=anchor_action,
        )

        # Store feature matrix (potentially normalized)
        if anchor_action is not None:
            # Normalize: subtract anchor action's features from all actions
            # This ensures U(s, anchor_action; θ) = 0 for all s, θ
            anchor_features = feature_matrix[:, anchor_action : anchor_action + 1, :]
            self._feature_matrix = feature_matrix - anchor_features
        else:
            self._feature_matrix = feature_matrix.clone()

    @property
    def feature_matrix(self) -> torch.Tensor:
        """Return the feature matrix (potentially normalized)."""
        return self._feature_matrix

    def compute(self, parameters: torch.Tensor) -> torch.Tensor:
        """Compute utility matrix U(s, a; θ) = θ · φ(s, a).

        Args:
            parameters: Parameter vector θ of shape (num_parameters,)

        Returns:
            Utility matrix of shape (num_states, num_actions)
        """
        self.validate_parameters(parameters)

        # U[s, a] = Σ_k θ[k] * φ[s, a, k]
        return torch.einsum("sak,k->sa", self._feature_matrix, parameters)

    def compute_gradient(self, parameters: torch.Tensor) -> torch.Tensor:
        """Compute gradient ∂U/∂θ = φ(s, a).

        For linear utility, the gradient is simply the feature matrix,
        independent of the parameter values.

        Args:
            parameters: Parameter vector (unused, but kept for interface consistency)

        Returns:
            Gradient tensor of shape (num_states, num_actions, num_parameters)
        """
        # For linear utility, gradient is constant
        return self._feature_matrix.clone()

    def compute_hessian(self, parameters: torch.Tensor) -> torch.Tensor:
        """Compute Hessian ∂²U/∂θ².

        For linear utility, the Hessian is zero (utility is linear in θ).

        Args:
            parameters: Parameter vector (unused)

        Returns:
            Zero tensor of shape (num_states, num_actions, num_parameters, num_parameters)
        """
        return torch.zeros(
            (self.num_states, self.num_actions, self.num_parameters, self.num_parameters),
            dtype=self._feature_matrix.dtype,
            device=self._feature_matrix.device,
        )

    def get_parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return parameter bounds for all parameters.

        Returns unbounded (-inf, inf) for all parameters. Estimators that
        use bounded optimization (e.g., L-BFGS-B in CCP) will use these
        bounds. Users can override by passing explicit bounds to the
        estimator.
        """
        n_params = self.num_parameters
        lower = torch.full((n_params,), float("-inf"))
        upper = torch.full((n_params,), float("inf"))
        return lower, upper

    @classmethod
    def from_environment(
        cls,
        env,
        anchor_action: int | None = None,
    ) -> "LinearUtility":
        """Create LinearUtility from a DDCEnvironment.

        Convenience constructor that extracts the feature matrix and
        parameter names from an environment.

        Args:
            env: A DDCEnvironment instance
            anchor_action: Action to normalize (default: use env's setting if any)

        Returns:
            LinearUtility instance configured for this environment
        """
        return cls(
            feature_matrix=env.feature_matrix,
            parameter_names=env.parameter_names,
            anchor_action=anchor_action,
        )

    def to(self, device: torch.device | str) -> "LinearUtility":
        """Move feature matrix to specified device."""
        new_utility = LinearUtility(
            feature_matrix=self._feature_matrix.to(device),
            parameter_names=self._parameter_names.copy(),
            anchor_action=None,  # Already normalized if needed
        )
        new_utility._anchor_action = self._anchor_action
        return new_utility

    def subset_states(self, state_indices: torch.Tensor) -> "LinearUtility":
        """Create a new utility function for a subset of states.

        Useful for state aggregation or focusing on specific regions.

        Args:
            state_indices: Indices of states to keep

        Returns:
            New LinearUtility with only selected states
        """
        new_features = self._feature_matrix[state_indices, :, :]
        return LinearUtility(
            feature_matrix=new_features,
            parameter_names=self._parameter_names.copy(),
            anchor_action=self._anchor_action,
        )
