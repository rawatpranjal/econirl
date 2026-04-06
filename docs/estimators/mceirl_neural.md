# MCE-IRL (Neural)

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Wulfmeier et al. (2015) | Neural | Yes | No | No |

## What this estimator does

Linear MCE-IRL assumes the reward takes the form $R(s,a) = \theta^\top \phi(s,a)$, which works when the analyst knows the correct features. When the true reward depends on nonlinear transformations of state variables, a linear model misspecifies the reward and produces suboptimal policies. Wulfmeier et al. (2015) replace the linear reward with a neural network $R_\psi(s,a)$ while keeping the rest of the MCE framework unchanged. The soft Bellman recursion, occupancy measure computation, and feature matching structure all stay the same. The only change is the reward parameterization and how the gradient propagates through it.

Neural rewards inherit the same identification limitations as linear rewards. Rewards are identified only up to potential-based shaping. Post-hoc interpretability is achieved through sieve projection, where the neural reward surface is projected onto a linear feature basis via least squares. The projection $R^2$ measures how well the neural reward can be explained by the chosen features. Standard errors from the projection regression are approximate because they do not account for estimation error in the neural reward itself.

## How it works

The gradient flows through the network via a surrogate loss that treats the occupancy mismatch as a stop-gradient constant:

$$
\mathcal{L}(\psi) = -\sum_{s,a} [\mu_D(s,a) - \mu_\pi(s,a)]_{\text{sg}} \cdot R_\psi(s,a).
$$

Each iteration computes the neural reward matrix, solves the soft Bellman equation for the policy (backward pass), computes the occupancy measure (forward pass), and backpropagates through the surrogate loss to update the network weights. Early stopping monitors the feature matching residual and restores the best model. Sieve projection after training recovers interpretable parameters by regressing the neural reward onto linear features.

## When to use it

Deep MCE-IRL is designed for environments where the reward depends on complex nonlinear structure that is difficult to specify as hand-crafted features. On small state spaces, the neural network tends to overfit because the occupancy mismatch provides a low-dimensional gradient signal relative to the number of network parameters. Linear MCE-IRL outperforms the deep variant on small grids where two or three linear parameters suffice. Deep MCE-IRL is expected to outperform linear MCE-IRL on larger state spaces where nonlinear reward structure matters more.

## References

- Wulfmeier, M., Ondruska, P., and Posner, I. (2015). Maximum Entropy Deep Inverse Reinforcement Learning. *arXiv:1507.04888*.

The full derivation, algorithm, and simulation results are in the [Deep MCE-IRL primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/deep_mce_irl.pdf).
