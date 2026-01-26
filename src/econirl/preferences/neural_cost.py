"""Neural network cost function for Guided Cost Learning.

This module implements a neural network cost function c(s,a) that maps
discrete state-action pairs to costs via learned embeddings and an MLP.

Used in Guided Cost Learning (Finn et al. 2016) where the cost function
is parameterized as a neural network instead of linear features.

Reference:
    Finn, C., Levine, S., & Abbeel, P. (2016). Guided Cost Learning:
    Deep Inverse Optimal Control via Policy Optimization. ICML.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from econirl.preferences.base import BaseUtilityFunction


class NeuralCostFunction(nn.Module, BaseUtilityFunction):
    """Neural network cost function for discrete state-action spaces.

    Implements c(s, a) = MLP(Embed(s) || Embed(a)) where:
    - Embed(s) is a learned state embedding
    - Embed(a) is a learned action embedding
    - || denotes concatenation
    - MLP is a multi-layer perceptron

    This is used in Guided Cost Learning (GCL) where the cost function
    is learned from demonstrations via importance sampling.

    Attributes
    ----------
    state_embedding : nn.Embedding
        Embedding layer for states.
    action_embedding : nn.Embedding
        Embedding layer for actions.
    mlp : nn.Sequential
        MLP that maps concatenated embeddings to scalar cost.

    Parameters
    ----------
    n_states : int
        Number of discrete states.
    n_actions : int
        Number of discrete actions.
    embed_dim : int, default=32
        Dimension of state and action embeddings.
    hidden_dims : list[int], default=[64, 64]
        Hidden layer dimensions for the MLP.
    activation : str, default="relu"
        Activation function: "relu", "tanh", or "leaky_relu".

    Examples
    --------
    >>> cost_fn = NeuralCostFunction(n_states=100, n_actions=4)
    >>> states = torch.tensor([0, 5, 10])
    >>> actions = torch.tensor([1, 2, 0])
    >>> costs = cost_fn(states, actions)  # shape: (3,)
    >>> cost_matrix = cost_fn.compute()   # shape: (100, 4)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        embed_dim: int = 32,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ):
        # Initialize nn.Module first
        nn.Module.__init__(self)

        if hidden_dims is None:
            hidden_dims = [64, 64]

        # Initialize BaseUtilityFunction
        # Parameter names represent the network weights (not used for optimization directly)
        param_names = ["neural_cost_params"]
        BaseUtilityFunction.__init__(
            self,
            num_states=n_states,
            num_actions=n_actions,
            parameter_names=param_names,
        )

        self._embed_dim = embed_dim
        self._hidden_dims = hidden_dims

        # Embedding layers
        self.state_embedding = nn.Embedding(n_states, embed_dim)
        self.action_embedding = nn.Embedding(n_actions, embed_dim)

        # Initialize embeddings with small values
        nn.init.normal_(self.state_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.1)

        # Build MLP
        layers = []
        input_dim = 2 * embed_dim  # concatenated state + action embeddings

        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_fn())
            input_dim = hidden_dim

        # Output layer (scalar cost)
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize MLP weights
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def embed_dim(self) -> int:
        """Embedding dimension."""
        return self._embed_dim

    @property
    def hidden_dims(self) -> list[int]:
        """Hidden layer dimensions."""
        return self._hidden_dims.copy()

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute costs for batched state-action pairs.

        Parameters
        ----------
        states : torch.Tensor
            State indices, shape (batch_size,) or scalar.
        actions : torch.Tensor
            Action indices, shape (batch_size,) or scalar.

        Returns
        -------
        costs : torch.Tensor
            Cost values, shape (batch_size,) or scalar.
        """
        # Handle scalar inputs
        if states.dim() == 0:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Get embeddings
        state_embed = self.state_embedding(states)  # (batch, embed_dim)
        action_embed = self.action_embedding(actions)  # (batch, embed_dim)

        # Concatenate and pass through MLP
        combined = torch.cat([state_embed, action_embed], dim=-1)
        costs = self.mlp(combined).squeeze(-1)  # (batch,)

        if squeeze_output:
            costs = costs.squeeze(0)

        return costs

    def compute(self, parameters: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the full cost matrix c(s, a) for all state-action pairs.

        For GCL, the cost function is parameterized by neural network weights,
        not an explicit parameter vector. The `parameters` argument is ignored.

        Parameters
        ----------
        parameters : torch.Tensor, optional
            Ignored. Kept for interface compatibility.

        Returns
        -------
        cost_matrix : torch.Tensor
            Cost matrix of shape (n_states, n_actions) where
            cost_matrix[s, a] = c(s, a).
        """
        n_states = self.num_states
        n_actions = self.num_actions

        # Create all state-action pairs
        states = torch.arange(n_states, device=self.state_embedding.weight.device)
        actions = torch.arange(n_actions, device=self.action_embedding.weight.device)

        # Compute cost for each (state, action) pair
        # Expand to create all combinations
        states_expanded = states.unsqueeze(1).expand(n_states, n_actions)
        actions_expanded = actions.unsqueeze(0).expand(n_states, n_actions)

        # Flatten for batch processing
        states_flat = states_expanded.reshape(-1)
        actions_flat = actions_expanded.reshape(-1)

        # Compute costs
        costs_flat = self(states_flat, actions_flat)

        # Reshape to matrix
        cost_matrix = costs_flat.reshape(n_states, n_actions)

        return cost_matrix

    def compute_gradient(self, parameters: torch.Tensor | None = None) -> torch.Tensor:
        """Compute gradient of cost w.r.t. parameters.

        For neural network cost functions, gradients are computed via
        backpropagation through the network, not analytically.
        This method returns zeros as a placeholder.

        Parameters
        ----------
        parameters : torch.Tensor, optional
            Ignored.

        Returns
        -------
        gradient : torch.Tensor
            Zero tensor of shape (n_states, n_actions, 1).
        """
        # For neural networks, we use autograd instead of analytic gradients
        return torch.zeros(
            (self.num_states, self.num_actions, 1),
            dtype=torch.float32,
            device=self.state_embedding.weight.device,
        )

    def get_reward_matrix(self) -> torch.Tensor:
        """Get the reward matrix R(s, a) = -c(s, a).

        In IRL, we typically work with rewards (higher is better),
        while GCL learns costs (lower is better). This method returns
        the negated cost matrix.

        Returns
        -------
        reward_matrix : torch.Tensor
            Reward matrix of shape (n_states, n_actions).
        """
        return -self.compute()

    def to_device(self, device: torch.device | str) -> "NeuralCostFunction":
        """Move the cost function to specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        self : NeuralCostFunction
            Self (moved in-place).
        """
        self.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"NeuralCostFunction("
            f"n_states={self.num_states}, "
            f"n_actions={self.num_actions}, "
            f"embed_dim={self._embed_dim}, "
            f"hidden_dims={self._hidden_dims})"
        )
