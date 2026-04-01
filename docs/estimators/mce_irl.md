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
