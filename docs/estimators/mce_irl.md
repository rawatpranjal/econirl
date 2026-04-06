# MCE-IRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Ziebart (2010) | Linear | Yes | Bootstrap | No |

## What this estimator does

Standard MaxEnt IRL uses non-causal state visitation frequencies that propagate future information backward, violating the agent's information structure. Ziebart (2010) introduced Maximum Causal Entropy IRL, which conditions each action only on information available at decision time by replacing Shannon entropy with causal entropy. The resulting softmax Bellman equation is simultaneously the MCE optimal policy and the structural logit choice model of Rust (1987), making MCE-IRL the bridge between econometrics and IRL.

The MCE distribution has a minimax robustness property. Among all distributions that match the feature constraints, it minimizes the worst-case KL divergence to the true unknown distribution. This gives formal protection against model misspecification. Rewards are identified only up to potential-based shaping, but the parametric restriction $r = \theta^\top \phi$ with the feature matching dual resolves this by selecting the maximum-entropy-compatible reward from the equivalence class. MCE-IRL and NFXP optimize the same objective and converge to the same parameters.

## How it works

The gradient is the gap between expert and model feature averages:

$$
\nabla_\theta G(\theta) = \bar\phi_\pi(\theta) - \bar\phi_{\text{data}}.
$$

Each iteration solves the soft Bellman equation (backward pass) to get the policy, then computes the occupancy measure $D_\pi = (I - \beta F_\pi^\top)^{-1} \rho_0$ (forward pass) to get the expected features under that policy. The optimizer updates $\theta$ until the feature expectations match the data. Standard errors are computed via bootstrap by resampling trajectories and re-estimating. The deep variant replaces the linear reward with a neural network, which handles nonlinear preferences but loses parameter interpretability and valid standard errors.

## When to use it

MCE-IRL is the right choice for learning interpretable reward weights from demonstrations when you want standard errors and a direct connection to structural econometrics. It requires explicit transition matrices, so it does not apply to problems where transitions are unknown. For nonlinear reward functions, use the neural variant at the cost of interpretability. For settings without known transitions, consider AIRL or GLADIUS.

## References

- Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. *PhD Thesis, Carnegie Mellon University*.

The full derivation, algorithm, and simulation results are in the [MCE-IRL primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/mce_irl.pdf).
