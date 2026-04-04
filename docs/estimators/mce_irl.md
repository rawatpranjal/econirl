# MCE-IRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Ziebart (2010) | Linear or Neural | Yes | Bootstrap | Deep variant only |

## Background

Many different payoff functions can explain the same behavior. Ziebart (2010) proposed a principled way to pick one: among all policies that match the expert's average feature usage, choose the one that is most random (highest entropy). This "maximum entropy" principle avoids over-committing to structure the data does not support. The key refinement was using causal entropy, which only counts randomness in the agent's own decisions and ignores randomness from the environment. The solution turns out to be exactly the logit model from the [Theory](../estimators_theory.md) section, so MCE-IRL is really just maximum likelihood from a different angle.

## Key Equations

Let $\mu_\mathcal{D}$ be the expert's average discounted feature usage. The problem is

$$
\max_\pi H_c(\pi) \quad \text{subject to} \quad E_\pi\bigg\{\sum_{t} \beta^t \vec{r}(s_t, a_t)\bigg\} = \mu_\mathcal{D}.
$$

The gradient is simply the gap between expert and model feature averages:

$$
\nabla_\theta \ell^p(\theta) = \mu_\mathcal{D} - E_{\pi_\theta}\bigg\{\sum_{t} \beta^t \vec{r}(s_t, a_t)\bigg\}.
$$

At convergence, the two match.

## Pseudocode

```
MCE-IRL(D, features, p, beta, sigma):
  1. Compute expert feature averages from data
  2. Initialize theta
  3. Repeat until convergence:
     a. Backward pass: solve soft Bellman for Q, V, pi
     b. Forward pass: compute how often each state is visited under pi
     c. Compute model feature averages
     d. Gradient = expert averages - model averages
     e. Update theta
  4. Bootstrap standard errors by resampling trajectories
  5. Return theta, SEs
```

## Strengths and Limitations

MCE-IRL bridges structural logit estimation and machine learning. It returns interpretable linear reward weights with exact feature matching and robust predictive behavior. The worst-case robustness guarantee (Ziebart 2010, Theorem 3) provides formal protection against model misspecification. Bootstrap standard errors allow inference on the recovered parameters.

The deep variant replaces the linear reward with a neural network, which handles nonlinear preferences but loses parameter interpretability and valid standard errors. The standard (linear) variant requires explicit transition matrices, so it does not apply to problems where transitions are unknown.

MCE-IRL is the right choice for learning reward weights from demonstrations when you want standard errors and interpretability. It is the bridge between IRL and structural econometrics.

## References

- Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. *PhD Thesis, Carnegie Mellon University*.

## Diagnostics and Guarantees

Reward parameters are identified only up to an additive constant and an overall scale factor, following Kim et al. (2021) and Cao and Cohen (2021). This means absolute parameter values are not meaningful on their own. Evaluate recovered parameters using cosine similarity or policy quality rather than raw RMSE. Features should be normalized to the range of negative one to one for well-conditioned optimization.

The estimator uses dual stopping criteria following the imitation library (Gleave and Toyer 2022). The gradient path checks both the gradient norm and the occupancy distance. Convergence is declared when either the maximum absolute gradient falls below outer_tol (default 1e-6) or when the L-infinity distance between the demonstration and policy state visitation frequencies drops below occupancy_tol (default 1e-3). The inner soft value iteration uses the hybrid SA-then-NK solver with a tolerance of 1e-8 and a maximum of 10,000 iterations. The inner solver switches from successive approximation to Newton-Kantorovich when the error drops below 1e-3.

Standard errors are computed via bootstrap by default, resampling trajectories and re-estimating parameters. The default number of bootstrap replications is 100. Hessian-based standard errors are also available by setting se_method to "hessian" in the configuration. Bootstrap standard errors are preferred because the IRL objective is not a standard likelihood, and the Hessian may not capture the full sampling variability from the feature-matching gradient.

The linear variant returns interpretable reward weights with exact feature matching and a worst-case robustness guarantee (Ziebart 2010, Theorem 3). The deep variant replaces the linear reward with a neural network, which handles nonlinear preferences but loses parameter interpretability and invalidates the bootstrap standard errors. Both variants require explicit transition matrices.

The default configuration uses L-BFGS-B as the outer optimizer with a maximum of 200 iterations. The gradient descent path uses Adam with a learning rate of 0.02, beta1 of 0.9, beta2 of 0.999, and a gradient clipping norm of 1.0. The state visitation computation runs for up to 1000 iterations with a tolerance of 1e-8.
