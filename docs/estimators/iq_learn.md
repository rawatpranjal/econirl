# IQ-Learn

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Q-Learning IRL | Garg et al. (2021) | Tabular | Yes (exact) | No | Moderate |

## Background

IQ-Learn recovers the reward function from a single optimization over Q-function parameters, avoiding the adversarial min-max loop of AIRL and the Bellman inner loop of NFXP. The key observation is that the soft Bellman equation can be rearranged to express the reward as a function of Q alone. This is the inverse Bellman operator. IQ-Learn builds a concave objective around this operator and solves it with standard gradient methods.

Unlike the reduced-form logit (which also learns Q by maximum likelihood), IQ-Learn uses transition data to enforce Bellman consistency. This means the recovered Q-function respects the temporal structure of the MDP, and the implied reward separates immediate payoffs from continuation values. The reduced-form logit bundles the two together.

## Key Equations

The inverse Bellman operator expresses reward as a function of $Q$ alone,

$$
r_Q(s,a) = Q(s,a) - \beta \sum_{s'} P(s' \mid s,a)\, V_Q(s'),
$$

where $V_Q(s) = \log \sum_{a'} \exp Q(s,a')$. IQ-Learn estimates $Q$ by solving

$$
\max_Q \; \mathbb{E}_{(s,a) \sim \rho^*}\bigl[\phi(r_Q(s,a))\bigr] - (1-\beta)\,\mathbb{E}_{s_0 \sim \mu_0}[V_Q(s_0)],
$$

where $\rho^*$ is the expert occupancy measure and $\phi$ is a concave function that determines the divergence. The default is $\phi(x) = x - x^2/2$ for chi-squared divergence.

## Pseudocode

```
IQ-Learn(D, P, beta, sigma):
  1. Initialize tabular Q(s,a)
  2. For each iteration:
     a. Compute V(s) = sigma * logsumexp(Q / sigma)
     b. Compute implied reward: r(s,a) = Q(s,a) - beta * E[V(s')]
     c. Compute loss = -E_expert[phi(r)] + (1-beta) * E_s0[V(s0)]
     d. Update Q via gradient descent (or L-BFGS-B)
  3. Extract policy: pi(a|s) = softmax(Q(s,:) / sigma)
  4. Extract reward: r(s,a) = Q(s,a) - beta * E[V(s')]
  5. Return r, pi, Q
```

## Strengths and Limitations

IQ-Learn replaces the adversarial loop with a single concave optimization. This makes training more stable than AIRL and avoids the mode collapse issues of GAN-based methods. It requires the transition kernel $P(s' \mid s,a)$ to compute the inverse Bellman operator, which is a stronger data requirement than GLADIUS (which learns $E[V(s')]$ from empirical transitions).

In finite samples with tabular Q parameterization, the chi-squared objective can have multiple local optima when the state space is large. The global finite-sample optimum may differ from the population optimum. This issue does not affect the theoretical guarantee (population-level recovery is exact) but matters in practice.

IQ-Learn is the right choice when you have the transition matrix available and want non-adversarial reward recovery with Bellman consistency.

## References

- Garg, D., Chakraborty, S., Cundy, C., Song, J., and Ermon, S. (2021). IQ-Learn: Inverse soft-Q Learning for Imitation. NeurIPS.

## Diagnostics and Guarantees

At the population level, IQ-Learn recovers the true soft Q-function exactly under the chi-squared divergence when the expert policy is the unique optimizer of the true reward. The reward is then recovered from Q via the inverse Bellman operator, r(s,a) = Q(s,a) minus beta times the expected next-period soft value. This identification requires the transition kernel P(s' given s,a) to be known. Unlike MCE-IRL, IQ-Learn identifies Q directly rather than reward parameters, so the reward is a derived quantity.

Convergence depends on the chosen optimizer. With L-BFGS-B (the default), the estimator terminates when the gradient norm falls below 1e-6 or after 500 iterations. With Adam, it runs for a fixed number of iterations (default 500) and checks whether the gradient norm has dropped below the convergence tolerance of 1e-6. The chi-squared objective includes a regularization term controlled by the alpha parameter (default 1.0) that ensures the objective is strictly concave in Q.

IQ-Learn does not produce standard errors on the recovered parameters. The Q-function is estimated via a non-standard objective (not a likelihood), so there is no natural Hessian to invert for variance estimation. The standard errors are reported as NaN. If inference is needed, the user should consider NFXP or CCP, which provide valid standard errors through maximum likelihood.

In finite samples with tabular Q parameterization, the chi-squared objective can have multiple local optima when the state space is large. The global finite-sample optimum may differ from the population optimum because the empirical occupancy measure may not have full support. The simple (total variation) divergence avoids this issue but provides weaker finite-sample guarantees. When using linear Q parameterization (q_type set to "linear"), the recovered parameters are in the structural parameter space, but the Bellman consistency is only approximate.

The default configuration uses tabular Q parameterization, chi-squared divergence with alpha of 1.0, L-BFGS-B as the optimizer with a maximum of 500 iterations, and a convergence tolerance of 1e-6. The learning rate of 0.01 applies only when Adam is selected as the optimizer.
