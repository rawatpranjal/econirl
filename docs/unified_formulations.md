# Estimators

This page presents the theory, mathematical formulations, and selection guide for all econirl estimators using common notation from Rust and Rawat (2026). For worked code examples, see the [examples](examples/index.rst).

## Theory

### The Problem

An agent makes sequential decisions under uncertainty. At each period the agent observes a state $s \in \mathcal{S}$, chooses an action $a \in \mathcal{A}(s)$, receives a flow payoff $r(s,a)$, and transitions to a new state $s'$ drawn from $p(s' \mid s,a)$. The agent discounts the future by $\beta \in (0,1)$. The goal of the agent is to maximize the expected discounted sum of payoffs $E\{\sum_{t=0}^\infty \beta^t r(s_t, a_t)\}$.

This is a Markov decision process. Its solution is an optimal policy $\delta^*(s) = \arg\max_a Q^*(s,a)$ where $Q^*(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) V^*(s')$ is the choice-specific value and $V^*(s) = \max_a Q^*(s,a)$ is the value function. Both satisfy the Bellman equation, a fixed-point condition that balances current payoff against discounted continuation value.

The structural estimation problem is to run this logic in reverse. We observe agents' states and choices but not their preferences. We want to recover the reward function $r$ (or its parameters $\theta$) that rationalizes the observed behavior. This is what the econometrics literature calls the identification and estimation of dynamic discrete choice (DDC) models, and what the machine learning literature calls inverse reinforcement learning (IRL).

### Statistical Degeneracy and Taste Shocks

The optimal policy $\delta^*(s)$ is a deterministic function of the state. But in data, different agents in the same observed state often make different choices. A deterministic model cannot explain this heterogeneity. Rust (1987) resolved this by augmenting the state with unobserved idiosyncratic taste shocks $\varepsilon = (\varepsilon_1, \ldots, \varepsilon_{|\mathcal{A}|})$, one per action, observed by the agent but not by the econometrician. The agent's full state is $(s, \varepsilon)$ and the reward for action $a$ is $r(s,a) + \varepsilon_a$. Even though the agent's decision rule $\delta^*(s, \varepsilon)$ is still deterministic from the agent's standpoint, it appears stochastic from the observer's standpoint because $\varepsilon$ is unobserved.

Following McFadden (1973), assume $\varepsilon$ has a multivariate extreme value type 1 (Gumbel) distribution with scale parameter $\sigma > 0$ and mean zero. Assume also that transitions factor as $p(s', \varepsilon' \mid s, \varepsilon, a) = g(\varepsilon' \mid s') \, p(s' \mid s,a)$, so the taste shocks are conditionally independent across periods. Under these assumptions, the value function $V(s,\varepsilon) = \max_a [Q(s,a) + \varepsilon_a]$ can be integrated analytically over $\varepsilon$.

### Deriving the Soft Bellman Equations

The key property of the EV1 distribution is that the expected maximum has a closed-form expression. Integrating $V(s,\varepsilon)$ over $\varepsilon$ yields the expected value function

$$
V(s) = E_\varepsilon\!\big[\max_a (Q(s,a) + \varepsilon_a)\big] = \sigma \log \sum_{a \in \mathcal{A}(s)} \exp\!\big(Q(s,a)/\sigma\big).
$$

This is the log-sum-exp or "social surplus" function. Substituting back into the Bellman equation, the choice-specific value $Q$ is the unique fixed point of the soft Bellman operator $\Lambda_\sigma$,

$$
\Lambda_\sigma(Q)(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) \; \sigma \log \sum_{a'} \exp\!\big(Q(s',a')/\sigma\big).
$$

The operator $\Lambda_\sigma$ is a $\beta$-contraction on the space of bounded functions, so the fixed point $Q^* = \Lambda_\sigma(Q^*)$ exists and is unique. As $\sigma \to 0$ the soft operator converges to the hard Bellman operator $\max_a Q(s,a)$, recovering the standard MDP.

The probability that the agent chooses action $a$ in state $s$, called the conditional choice probability (CCP) in economics or the policy in reinforcement learning, is the multinomial logit (softmax),

$$
\pi(a \mid s) = \frac{\exp\!\big(Q(s,a)/\sigma\big)}{\sum_{b \in \mathcal{A}(s)} \exp\!\big(Q(s,b)/\sigma\big)}.
$$

More generally, with a non-uniform base measure $\mu(a \mid s) > 0$, the CCP becomes $\pi(a \mid s) = \mu(a \mid s) \exp(Q(s,a)/\sigma) / \sum_b \mu(b \mid s) \exp(Q(s,b)/\sigma)$. The uniform case $\mu = 1/|\mathcal{A}|$ is the standard Rust model.

### The Variational Interpretation

The same equations arise from a completely different starting point. Consider an agent who maximizes expected reward minus a penalty for deviating from a reference policy $\mu$,

$$
V(s) = \max_{\pi(\cdot \mid s)} \bigg\{ \sum_a \pi(a \mid s) \, Q(s,a) - \sigma \, \mathrm{KL}\!\big(\pi(\cdot \mid s) \| \mu(\cdot \mid s)\big) \bigg\}.
$$

By the Fenchel-Moreau (Donsker-Varadhan) duality, the maximum equals the log-sum-exp $V(s) = \sigma \log \sum_a \mu(a \mid s) \exp(Q(s,a)/\sigma)$ and the maximizer is the softmax CCP $\pi(a \mid s)$ above. This establishes a fundamental equivalence. The DDC multinomial logit model with EV1 taste shocks solves the same optimization problem as entropy-regularized reinforcement learning. The temperature $\sigma$ has a dual interpretation. In economics it is the scale of the unobserved taste shocks. In machine learning it is the weight on KL regularization that prevents the policy from collapsing to a deterministic rule. Both interpretations are mathematically identical, as shown in Appendix A of Rust and Rawat (2026).

### Notation Summary

| Symbol | Definition |
|--------|-----------|
| $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$ | State, action |
| $\beta \in (0,1)$ | Discount factor |
| $\sigma > 0$ | EV1 scale parameter (equivalently, KL regularization weight) |
| $\mu(a \mid s) > 0$ | Base policy with full support |
| $\vec{r}(s,a) = (r_1(s,a), \ldots, r_K(s,a))$ | Feature vector |
| $r_\theta(s,a) = \vec{r}(s,a) \cdot \theta$ | Linear reward parameterized by $\theta \in \mathbb{R}^K$ |
| $p(s' \mid s,a)$ | Transition probability |
| $Q(s,a)$, $V(s)$ | Choice-specific value, expected value |
| $\pi(a \mid s)$ | Conditional choice probability (CCP) or policy |
| $\mathcal{D} = \{\tau_i\}_{i=1}^N$ | Observed trajectories, $\tau_i = \{(s_{it}, a_{it})\}_{t=0}^{T_i}$ |
| $R(s,a) \in \mathbb{R}^K$ | Discounted continuation feature vector |
| $Q_r(s,a)$, $Q_\varepsilon(s,a)$ | Reward component and entropy correction of $Q$ |

### The Q Decomposition

When the reward is linear in parameters, $r_\theta(s,a) = \vec{r}(s,a) \cdot \theta$, the choice-specific value decomposes as $Q(s,a) = Q_r(s,a) + Q_\varepsilon(s,a)$. The reward component satisfies $Q_r(s,a) = R(s,a)^\top \theta$ where the continuation feature vector $R$ solves

$$
R(s,a) = \vec{r}(s,a) + \beta \sum_{s'} p(s' \mid s,a) \sum_{a'} \pi(a' \mid s') \, R(s',a').
$$

Each element $R_k(s,a)$ is the discounted expected sum of feature $k$ starting from $(s,a)$ under policy $\pi$. The entropy correction $Q_\varepsilon$ captures the expected future value of having choice flexibility,

$$
Q_\varepsilon(s,a) = \beta \sum_{s'} p(s' \mid s,a) \sum_{a'} \pi(a' \mid s') \big[-\sigma \log \pi(a' \mid s') + Q_\varepsilon(s',a')\big].
$$

This decomposition is central to the CCP and TD-CCP estimators. It separates the part of $Q$ that depends on $\theta$ (through $R$) from the part that depends only on the policy (through $Q_\varepsilon$).

### Likelihood Functions

Suppose we observe trajectories $\mathcal{D} = \{\tau_i\}_{i=1}^N$ where $\tau_i = \{(s_{it}, a_{it})\}_{t=0}^{T_i}$. Parametrize the reward as $r_\theta(s,a) = \vec{r}(s,a) \cdot \theta$. The full likelihood conditions on both choices and transitions,

$$
\mathcal{L}^f_\mathcal{D}(\theta) = \prod_{i=1}^N \prod_{t=1}^{T_i} \pi_\theta(a_{it} \mid s_{it}) \, p_\theta(s_{it} \mid s_{it-1}, a_{it-1}).
$$

The partial likelihood conditions on choices only,

$$
\mathcal{L}^p_\mathcal{D}(\theta) = \prod_{i=1}^N \prod_{t=1}^{T_i} \pi_\theta(a_{it} \mid s_{it}).
$$

Most estimators maximize the log partial likelihood $\ell^p(\theta) = \log \mathcal{L}^p_\mathcal{D}(\theta)$ because the transition component does not depend on $\theta$ when transitions are estimated nonparametrically.

### Identification

Rewards are not uniquely identified from observed behavior without additional restrictions. From the softmax CCP, only Q-value differences are recoverable,

$$
\log\!\big(\pi(a \mid s) / \pi(a' \mid s)\big) = \big(Q(s,a) - Q(s,a')\big) / \sigma.
$$

The level of $Q$ is not identified, so neither is the level of $r$. Ng, Harada, and Russell (1999) showed that the set of observationally equivalent rewards forms an equivalence class. For any function $h \colon \mathcal{S} \to \mathbb{R}$,

$$
r_h(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) \, h(s') - h(s)
$$

yields $\pi_{r_h} = \pi_r$ under the same dynamics. All rewards in this class generate identical behavior. But under different dynamics $\lambda(s' \mid s,a) \neq p(s' \mid s,a)$, the counterfactual policies diverge. This is why identification matters for policy evaluation.

Rust and Rawat (2026) adopt two gauges. The state-potential gauge removes the additive state constant, $E_{a \sim \mu(\cdot \mid s)}[r_\theta(s,a)] = 0$ for all $s$. The reference Q-gap fixes the scale, $Q^*_\theta(\bar{s}, a^+) - Q^*_\theta(\bar{s}, a^-) = \Delta^*$ for a chosen reference state and action pair. Together these pin down $\theta$. An alternative is the normalizing (anchor) action assumption used in CCP and GLADIUS, where $r(s, a_s)$ is known for one action $a_s$ in each state.

### Equivalence Theorem

All estimators on this page operate on the same soft Bellman system. Rust and Rawat (2026, Theorem A.6) establish that under the soft-control framework with the gauge normalizations, consistent estimators from different data sources converge to the same policy. If $\hat\theta_{\mathrm{NFXP}}$ (from choice data), $\hat\theta_{\mathrm{IRL}}$ (from expert demonstrations), and $\hat\theta_{\mathrm{RLHF}}$ (from pairwise preferences) each solve their respective objectives, then as sample size grows,

$$
\pi^*_{\hat\theta_{\mathrm{NFXP}}} = \pi^*_{\hat\theta_{\mathrm{IRL}}} = \pi^*_{\hat\theta_{\mathrm{RLHF}}} \;\longrightarrow\; \pi^*_{\theta^\star}.
$$

The choice of estimation method does not affect the limiting policy. This equivalence between maximum likelihood (DDC), maximum causal entropy (IRL), and Bradley-Terry preference learning (RLHF) is the central theoretical result unifying the three fields.

---

## Estimators

### Structural Estimators

Structural estimators recover utility parameters $\theta$ from observed choices, taking the direction $\theta \to \pi$.

#### NFXP

**Motivation.** Rust (1987) faced a fundamental computational challenge. Evaluating the likelihood at any candidate $\theta$ requires knowing the choice probabilities $\pi_\theta(a \mid s)$, which depend on $Q_\theta$, which is defined as the fixed point of the soft Bellman operator. Every trial value of $\theta$ in the optimizer requires solving an entire dynamic programming problem. Rust's insight was to nest this fixed-point computation inside a standard hill-climbing loop, hence the name "nested fixed point." The cost is high but the payoff is exact maximum likelihood with all its statistical guarantees. Iskhakov et al. (2016) improved the inner loop by starting with successive approximation (globally convergent but linearly fast) and switching to Newton-Kantorovich (quadratically convergent but only locally stable) near the fixed point. This polyalgorithm combines safety and speed.

**Objective.** The NFXP estimator maximizes the exact log-likelihood,

$$
\hat\theta_{\mathrm{NFXP}} = \arg\max_\theta \sum_{(s,a) \in \mathcal{D}} \log \pi^*_\theta(a \mid s),
$$

where $\pi^*_\theta$ is computed by solving $Q_\theta = \Lambda_\sigma(Q_\theta)$ at each candidate $\theta$. The gradient $\nabla_\theta \ell$ is computed analytically via the implicit function theorem through the Frechet derivative $(I - \beta P_\pi)^{-1}$. The outer loop uses BHHH optimization, which forms a positive-definite Hessian approximation from per-observation score outer products.

**Pseudocode.**

```
NFXP(D, r_theta, p, beta, sigma):
  1. Initialize theta
  2. Repeat until convergence:
     a. Inner loop — solve for Q_theta:
        Q <- 0
        Repeat (SA phase):
          Q <- Lambda_sigma(Q; theta)        # contraction mapping
        Until ||Q - Lambda_sigma(Q)|| < tol_switch
        Repeat (NK phase):
          Q <- Q - (I - beta*P_pi)^{-1} (Q - Lambda_sigma(Q))  # Newton step
        Until ||Q - Lambda_sigma(Q)|| < tol_inner
     b. Policy: pi(a|s) = exp(Q(s,a)/sigma) / sum_b exp(Q(s,b)/sigma)
     c. Log-likelihood: L = sum_{(s,a) in D} log pi(a|s)
     d. Gradient: dL/dtheta via implicit differentiation through (I - beta*P_pi)^{-1}
     e. Update theta via BHHH step
  3. Standard errors: inverse of BHHH Hessian (score outer products)
  4. Return theta, SEs
```

NFXP is the only estimator that delivers statistically efficient maximum likelihood estimates with analytical standard errors. Its cost is $O(|\mathcal{S}|^2 \times \text{inner iterations})$ per outer step, making it intractable for $|\mathcal{S}| > 10{,}000$ or $\beta > 0.995$.

#### CCP and NPL

**Motivation.** Hotz and Miller (1993) asked whether one could avoid solving the Bellman equation entirely. Their key insight was an inversion lemma. Under the EV1 assumption, the conditional choice probabilities $\pi(a \mid s)$ can be estimated nonparametrically from data, and these empirical CCPs can be directly inverted to recover value differences. The continuation value becomes a known function of the data, not an equilibrium object to be computed. This replaces thousands of Bellman iterations with a single matrix inversion. Aguirregabiria and Mira (2002) then showed that iterating this procedure in CCP space (re-estimating CCPs from updated $\theta$, re-inverting, re-estimating $\theta$) converges to the full MLE, recovering the efficiency that the one-shot Hotz-Miller estimator sacrifices. They called this nested pseudo-likelihood (NPL). The zero Jacobian property at the fixed point ensures that first-stage CCP estimation noise has zero first-order effect on $\hat\theta$, a form of Neyman orthogonality that was only recognized later.

**Objective.** The CCP takes the form

$$
\pi_\theta(a \mid s) = \frac{\exp\!\big(\hat{R}(s,a)^\top \theta + \hat{Q}_\varepsilon(s,a)\big)}{\sum_{a'} \exp\!\big(\hat{R}(s,a')^\top \theta + \hat{Q}_\varepsilon(s,a')\big)},
$$

where $\hat{R}$ and $\hat{Q}_\varepsilon$ are computed once from nonparametric CCP estimates by solving the linear systems via $(I - \beta F_\pi)^{-1}$, costing $O(|\mathcal{S}|^3)$. The estimator maximizes $\mathcal{L}^p_\mathcal{D}(\theta)$. NPL iterates re-estimate $\hat\pi$ from $\hat\theta$ at each step.

**Pseudocode.**

```
CCP(D, features, p, beta, sigma, K_npl):
  1. Estimate CCPs from data: pi_hat(a|s) = N(s,a) / N(s)
  2. For k = 1 to K_npl:
     a. Solve for R: R = (I - beta*F_pi)^{-1} * features      # one matrix inversion
     b. Solve for Q_eps: Q_eps = (I - beta*F_pi)^{-1} * e      # e(a|s) = -sigma*log(pi(a|s))
     c. Form CCP: pi_theta(a|s) = softmax(R(s,a)'*theta + Q_eps(s,a))
     d. Maximize partial likelihood over theta via L-BFGS
     e. Update CCPs: pi_hat <- pi_theta(.|.; theta_hat)         # NPL iteration
  3. Standard errors: inverse Hessian of partial log-likelihood
  4. Return theta, SEs
```

One step ($K=1$) gives a consistent but inefficient Hotz-Miller estimate. Five to ten NPL steps typically recover MLE efficiency. At the NPL fixed point, $\partial V / \partial \pi = 0$ ensures first-stage CCP noise does not affect $\hat\theta$ asymptotically.

#### SEES

**Motivation.** Both NFXP and CCP require enumerating the full state space, either to solve the Bellman equation (NFXP) or to invert the valuation matrix (CCP). When the state space is large or continuous, these tabular methods break down. Luo and Sang (2024) proposed approximating the value function with a sieve basis (Chebyshev polynomials or Fourier terms) rather than solving the inner fixed point. The idea is simple. Parameterize $V(s;\alpha) = \Psi(s)^\top \alpha$ and jointly optimize the structural parameters $\theta$ and the basis coefficients $\alpha$. The penalty term $\lambda \|\alpha\|^2$ gradually enforces Bellman consistency. There is no neural network, no SGD, no mini-batches, and no learning rates. The entire estimation is a single L-BFGS-B call.

**Objective.**

$$
(\hat\theta, \hat\alpha) = \arg\max_{\theta, \alpha} \; \ell^f(\theta) - \frac{\lambda}{2} \|\alpha\|^2,
$$

where $Q(s,a;\theta,\alpha) = r_\theta(s,a) + \beta \sum_{s'} p(s' \mid s,a) \, \Psi(s')^\top \alpha$. Standard errors for $\theta$ use the Schur complement $H_{\theta\theta} - H_{\theta\alpha} H_{\alpha\alpha}^{-1} H_{\alpha\theta}$ to marginalize out $\alpha$.

**Pseudocode.**

```
SEES(D, features, p, beta, sigma, basis_type, M, lambda):
  1. Construct sieve basis Psi(s) = (psi_1(s), ..., psi_M(s))    # Chebyshev or Fourier
  2. Define Q(s,a; theta, alpha) = r_theta(s,a) + beta * P(.|s,a)' * Psi * alpha
  3. Define pi(a|s; theta, alpha) = softmax(Q(s,.; theta, alpha) / sigma)
  4. Objective: L(theta, alpha) = sum log pi(a|s) - (lambda/2) * ||alpha||^2
  5. Optimize (theta, alpha) jointly via L-BFGS-B
  6. Standard errors: Schur complement of joint Hessian
  7. Return theta, SEs
```

Cost is $O(M)$, independent of $|\mathcal{S}|$, making SEES tractable for state spaces exceeding 100,000. The estimator achieves Cramer-Rao efficiency as the sieve dimension $M$ grows with sample size.

#### NNES

**Motivation.** Nguyen (2025) observed that sieve bases work well when the value function is smooth, but can fail when $V$ has complex nonlinear structure. Neural networks are universal approximators, but naively plugging a neural $V_w$ into the likelihood creates a problem. The Hessian $\nabla^2_{ww}$ of the neural network is intractable, making standard error computation impossible. The key theoretical breakthrough was proving that the DDC likelihood has a special orthogonality structure. The score for $\theta$ is orthogonal to errors in $V$ (Neyman orthogonality). This means $\hat\theta$ is $\sqrt{n}$-consistent even when $V_w$ converges at a slower rate. Standard errors from the partial likelihood Hessian are valid without correcting for the neural network's approximation error. The gradient $\nabla_\theta Q$ is computed via equilibrium propagation, sidestepping the intractable Hessian entirely.

**Objective.**

$$
(\hat\theta, \hat{w}) = \arg\max_{\theta, w} \; \ell^p(\theta) - \lambda \sum_{(s,a,s') \in \mathcal{D}} \big(V_w(s) - r_\theta(s,a) - \beta V_w(s')\big)^2.
$$

**Pseudocode.**

```
NNES(D, features, p, beta, sigma, hidden_dims, lambda):
  1. Initialize theta, neural network V_w
  2. For k = 1 to K_outer:
     a. Phase 1 — train V_w (Bellman residual minimization):
        For epoch = 1 to E:
          Sample mini-batch (s, a, s') from D
          Loss = ||V_w(s) - r_theta(s,a) - beta * V_w(s')||^2
          Update w via Adam
     b. Phase 2 — estimate theta (structural MLE):
        Q(s,a) = r_theta(s,a) + beta * sum_{s'} p(s'|s,a) V_w(s')
        pi(a|s) = softmax(Q/sigma)
        theta <- argmax sum log pi(a|s)    # via L-BFGS, gradient by equilibrium propagation
  3. Standard errors: inverse Hessian of partial log-likelihood (valid by Neyman orthogonality)
  4. Return theta, SEs
```

NNES achieves the semiparametric efficiency bound, $\sqrt{n}(\hat\theta - \theta_0) \to \mathcal{N}(0, \Sigma^{-1})$, making it the only neural method with theoretically valid standard errors.

#### TD-CCP

**Motivation.** Adusumilli and Eckardt (2025) wanted the scalability of neural methods and the structural transparency of CCP, without requiring an explicit transition model. Their key innovation was per-feature decomposition. Instead of approximating a monolithic value function, they learn $K+1$ separate functions via temporal difference learning, one per reward feature $R_k(s,a)$ plus the entropy correction $Q_\varepsilon(s,a)$. This is model-free because it uses observed next-state samples $(s, a, s', a')$ rather than the transition matrix. The per-feature structure provides interpretable diagnostics. If $R_3$ has high approximation error but $R_1$ converges, the third feature's continuation-value structure is the modeling challenge, not the overall algorithm. A third estimation stage applies a debiased (Neyman-orthogonal) score correction to make $\hat\theta$ robust to first-stage noise.

**Objective.** The TD regression for the continuation features is

$$
\hat\phi = \Big[E_\mathcal{D}\big\{\nu(s,a)\big[\nu(s,a) - \beta \nu(s',a')\big]^\top\big\}\Big]^{-1} E_\mathcal{D}\big\{\nu(s,a) \, \vec{r}(s,a)\big\},
$$

where $\nu(s,a)$ are basis functions and $E_\mathcal{D}$ is the empirical expectation over observed transitions. Structural parameters $\theta$ are then estimated by maximizing $\mathcal{L}^p_\mathcal{D}(\theta)$ with the CCP form.

**Pseudocode.**

```
TD-CCP(D, features, beta, sigma, basis_functions):
  1. Estimate CCPs from data: pi_hat(a|s) = N(s,a) / N(s)
  2. For each feature k = 1, ..., K:
     a. Solve TD regression for R_k:
        phi_k = [E_D{nu(s,a)[nu(s,a) - beta*nu(s',a')]'}]^{-1} E_D{nu(s,a) r_k(s,a)}
     b. R_hat_k(s,a) = nu(s,a)' * phi_k
  3. Solve TD regression for Q_eps similarly (using -sigma*log(pi) as target)
  4. Form CCP: pi_theta(a|s) = softmax(R_hat(s,a)'*theta + Q_eps_hat(s,a))
  5. Maximize partial likelihood over theta
  6. Apply Neyman-orthogonal score correction for robustness
  7. Return theta, SEs
```

TD-CCP is model-free and provides per-feature diagnostics that no other neural method offers.

### Inverse Estimators

Inverse estimators recover reward parameters $\theta$ from expert demonstrations, taking the direction $\pi \to \theta$.

#### MCE-IRL

**Motivation.** Ziebart (2010) framed the IRL problem as a constrained optimization. Many policies are consistent with the observed expert behavior, in the sense that they match the expert's average feature counts. Which one should we pick? The maximum entropy principle (Jaynes 1957) says to choose the policy that is maximally noncommittal, the one with highest entropy subject to the feature-matching constraint. Ziebart's key contribution was using causal entropy $H_c(\pi) = E_\pi\{-\sum_t \beta^t \log \pi(a_t \mid s_t)\}$, which conditions each action only on information available at decision time. Standard Shannon entropy over trajectories includes randomness from state transitions, creating a risk-seeking bias in stochastic environments. The Lagrange multipliers of the constrained problem turn out to be the reward parameters $\theta$, and the KKT conditions yield exactly the softmax Bellman equations from the Theory section. This reveals that MCE-IRL is maximum likelihood estimation under the DDC model, establishing the mathematical equivalence between IRL and structural econometrics.

**Objective.** Let $\mu_\mathcal{D} = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} \beta^t \vec{r}(s_{it}, a_{it})$ be the expert's discounted feature counts. The primal problem is

$$
\max_\pi H_c(\pi) \quad \text{subject to} \quad E_\pi\bigg\{\sum_{t=0}^T \beta^t \vec{r}(s_t, a_t)\bigg\} = \mu_\mathcal{D}.
$$

The gradient of the log partial likelihood is the feature matching residual,

$$
\nabla_\theta \ell^p(\theta) = \mu_\mathcal{D} - E_{\pi_\theta}\bigg\{\sum_{t=0}^T \beta^t \vec{r}(s_t, a_t)\bigg\}.
$$

At the optimum, expert and model feature expectations match.

**Pseudocode.**

```
MCE-IRL(D, features, p, beta, sigma):
  1. Compute expert feature counts: mu_D = (1/N) sum_i sum_t beta^t features(s_it, a_it)
  2. Initialize theta
  3. Repeat until convergence:
     a. Backward pass — soft value iteration:
        Q(s,a) = features(s,a)'*theta + beta * sum_{s'} p(s'|s,a) V(s')
        V(s) = sigma * log sum_a exp(Q(s,a)/sigma)
        pi(a|s) = exp(Q(s,a)/sigma) / sum_b exp(Q(s,b)/sigma)
     b. Forward pass — state visitation frequencies:
        D(s) = rho_0(s) + beta * sum_{s',a} D(s') pi(a|s') p(s|s',a)
     c. Expected features: mu_pi = sum_s D(s) sum_a pi(a|s) features(s,a)
     d. Gradient: g = mu_D - mu_pi
     e. Update theta via L-BFGS or Adam
  4. Bootstrap SEs: resample trajectories, re-estimate theta B times
  5. Return theta, SEs
```

MCE-IRL minimizes worst-case prediction log-loss (Ziebart 2010, Theorem 3), making the recovered policy maximally robust to distribution shift. The deep variant replaces linear reward with a neural network $r_\phi(s,a)$ but loses interpretable parameters and standard errors.

#### AIRL

**Motivation.** GAIL (Ho and Ermon 2016) showed that IRL can be cast as a GAN, with a discriminator distinguishing expert from policy state-action pairs and a generator (policy) trying to fool it. But GAIL's learned reward $\log D(s,a)$ bakes in the training dynamics. When the environment changes, the reward becomes meaningless and the policy fails. Fu, Luo, and Levine (2018) asked what structure the discriminator must have for the learned reward to transfer. Their answer was the potential-based decomposition from the Theory section. By parameterizing the discriminator logits as $f(s,a,s') = g(s,a) + \beta h(s') - h(s)$, the shaping potential $h$ absorbs all dynamics-dependent value contributions. At optimality, $g$ recovers the intrinsic reward up to a constant. This is the disentanglement theorem. The structural economists will recognize $h$ as the potential function from Ng, Harada, and Russell (1999), and $g$ as the identified component of the reward.

**Objective.** The discriminator logits decompose as

$$
f_\phi(s,a,s') = g_\phi(s,a) + \beta \, h_\phi(s') - h_\phi(s),
$$

and the discriminator classifies expert versus policy transitions,

$$
D_\phi(s,a,s') = \frac{\exp(f_\phi(s,a,s'))}{\exp(f_\phi(s,a,s')) + \pi(a \mid s)}.
$$

**Pseudocode.**

```
AIRL(D_expert, p, beta, sigma, max_rounds):
  1. Initialize reward g_phi(s,a), shaping h_phi(s), policy pi_w
  2. For round = 1 to max_rounds:
     a. Sample expert transitions (s,a,s') from D_expert
     b. Sample policy transitions (s,a,s') by rolling out pi_w
     c. Discriminator update:
        f(s,a,s') = g_phi(s,a) + beta*h_phi(s') - h_phi(s)
        D = sigmoid(f - log pi_w(a|s))
        Maximize: E_expert[log D] + E_policy[log(1 - D)]
        Update (phi) via gradient ascent
     d. Policy update:
        Reward signal: r(s,a,s') = f(s,a,s')
        Update pi_w via policy gradient (REINFORCE or PPO)
  3. Portable reward: r(s,a) = g_phi(s,a)
  4. Return g_phi, pi_w
```

Under deterministic dynamics and state-only $g_\phi(s)$, at optimality $g_\phi(s) = r(s) + \text{const}$ (Theorem 5.1). The reward transfers across environments. Lee, Sudhir, and Wang (2026) show that an economic normalizing action provides an alternative identification strategy for action-dependent rewards.

#### f-IRL

**Motivation.** All previous IRL methods require the practitioner to specify something about the reward structure, either features (MCE-IRL), a neural architecture (deep MCE-IRL), or a discriminator structure (AIRL). Ni et al. (2022) asked whether one could do IRL with zero assumptions about reward structure. Their approach directly minimizes the f-divergence between the expert's state-action occupancy measure and the policy's occupancy measure. The reward is tabular, one free parameter per state-action pair. The choice of f-divergence gives a menu of robustness properties. KL divergence recovers maximum likelihood equivalence. Total variation is robust to outlier demonstrations. Chi-squared is sensitive to variance in occupancy ratios.

**Objective.**

$$
\min_r D_f(\rho_E \| \rho_{\pi_r}),
$$

where $\rho(s,a) = E_\pi\{\sum_{t=0}^\infty \beta^t \mathbb{I}(s_t = s, a_t = a)\}$ is the discounted occupancy measure. The gradient depends on the divergence: KL gives $\log(\rho_E / \rho_\pi)$, chi-squared gives $(\rho_E / \rho_\pi) - 1$, total variation gives $\mathrm{sign}(\rho_E - \rho_\pi)$.

**Pseudocode.**

```
f-IRL(D_expert, p, beta, sigma, f_divergence, lr):
  1. Compute expert occupancy: rho_E(s,a) from D_expert
  2. Initialize tabular reward r(s,a) = 0
  3. Repeat until convergence:
     a. Solve soft Bellman under r -> pi, V, Q
     b. Compute policy occupancy rho_pi via forward propagation
     c. Compute divergence gradient:
        If KL:     g(s,a) = log(rho_E(s,a) / rho_pi(s,a))
        If chi^2:  g(s,a) = rho_E(s,a) / rho_pi(s,a) - 1
        If TV:     g(s,a) = sign(rho_E(s,a) - rho_pi(s,a))
     d. Update: r(s,a) <- r(s,a) - lr * g(s,a)
  4. Return r (tabular reward)
```

### Model-Free Neural Estimator

#### GLADIUS

**Motivation.** Kang et al. (2025) identified a gap in the existing methods. Occupancy-matching methods like GAIL and IQ-Learn minimize average Bellman error only on the expert's support, leaving the Q-function unidentified off-support. NFXP and CCP require either solving the full Bellman equation or inverting a large matrix. GLADIUS addresses both problems by separately parameterizing $Q$ and the conditional expectation of $V$ with two neural networks, then jointly training them with a max-min formulation. The key insight is that the mean squared TD error decomposes into the Bellman error (what we want to minimize) plus the conditional variance of next-period values (irreducible noise from stochastic transitions). By estimating the variance term separately with the second network, GLADIUS targets the correct Bellman error without requiring $p(s' \mid s,a)$. After training, the reward is recovered from the Bellman identity $r = Q - \beta \, EV$ and projected onto linear features to extract structural parameters.

**Objective.** The outer maximization finds $\phi_1$ to maximize the penalized partial likelihood,

$$
\max_{\phi_1} \bigg[\ell^p(\phi_1) - \lambda \, \rho_{BE}(Q_{\phi_1})\bigg],
$$

where the Bellman error $\rho_{BE}$ is separated from conditional variance via the inner minimization over $\phi_2$.

**Pseudocode.**

```
GLADIUS(D, features, beta, sigma, lambda):
  1. Initialize Q-network Q_phi1(s,a), EV-network EV_phi2(s,a)
  2. For epoch = 1 to max_epochs:
     a. Sample mini-batch (s, a, s') from D
     b. V(s') = sigma * log sum_{a'} exp(Q_phi1(s',a')/sigma)
     c. NLL loss = -sum log softmax(Q_phi1(s,.)/sigma)[a]
     d. Bellman loss = lambda * ||beta * EV_phi2(s,a) - V(s')||^2
     e. Update phi1 to minimize NLL + Bellman loss      # outer max
     f. Update phi2 to minimize ||EV_phi2(s,a) - V(s')||^2   # inner min
  3. Recover reward: r(s,a) = Q_phi1(s,a) - beta * EV_phi2(s,a)
  4. Project onto features: theta = (Phi'Phi)^{-1} Phi' r
  5. Projection R^2 measures linear fit quality
  6. Return theta, r, R^2
```

Under realizability assumptions, GLADIUS achieves global convergence with error $O(1/T) + O(1/N)$.

### Baseline

#### BC

**Motivation.** Before running any structural or inverse method, you need to know whether the data contains enough signal. Behavioral cloning is the simplest possible estimator. It uses zero MDP structure. There is no reward, no value function, no Bellman equation, and no transition model. It just counts. If a sophisticated estimator cannot beat BC, it has not learned from sequential decision-making structure. Ross, Gordon, and Bagnell (2011) proved that BC error compounds quadratically with the horizon, $O(T^2 \varepsilon)$, while IRL methods that recover the true reward achieve $O(\varepsilon)$ regardless of $T$. This gap quantifies the value of structural estimation.

**Pseudocode.**

```
BC(D):
  1. For each state s, count visits N(s) and action counts N(s,a)
  2. pi(a|s) = N(s,a) / N(s)
  3. Return pi
```

---

## Guide

This section explains when to use each estimator.

### Decision Flowchart

```{image} _static/estimator_flowchart.png
:alt: Estimator selection flowchart
:width: 100%
```

### Quick Reference

| Estimator | Direction | Reward Type | Needs Transitions | Standard Errors | Scales Beyond Tabular | Transfer |
|-----------|-----------|-------------|-------------------|-----------------|----------------------|----------|
| NFXP-NK | Forward ($\theta \to \pi$) | Linear | Yes | Analytical (MLE) | No | No |
| CCP | Forward ($\theta \to \pi$) | Linear | Yes | Hessian | No | No |
| MCE-IRL | Inverse ($\pi \to \theta$) | Linear / Neural | Yes | Bootstrap | Deep variant only | No |
| TD-CCP | Forward ($\theta \to \pi$) | Linear | Yes | Hessian | Yes (neural AVI) | No |
| NNES | Forward ($\theta \to \pi$) | Linear | Yes | Valid (orthogonality) | Yes (neural V) | No |
| SEES | Forward ($\theta \to \pi$) | Linear | Yes | Marginal Hessian | Yes ($O(1)$ in $|\mathcal{S}|$) | No |
| AIRL | Inverse ($\pi \to R$) | Linear / Tabular | Yes | No | No | Yes |
| f-IRL | Inverse ($\pi \to R$) | Tabular | Yes | No | No | No |
| BC | Imitation | None | No | No | No | No |

### When to Use Each Estimator

**NFXP-NK** is for publication-grade structural estimates on tabular problems. It is the only estimator with statistically efficient MLE and analytical standard errors. Use it for replication of canonical DDC results (Rust bus, occupational choice) and whenever you need proper confidence intervals, likelihood ratio tests, and information criteria.

**CCP** is for rapid specification search, dynamic games, and any setting where the inner-loop cost of NFXP is the bottleneck. It completely avoids the Bellman inner loop by exploiting the Hotz-Miller inversion lemma. One step gives instant consistent estimates. Five to ten NPL steps recover MLE efficiency. It is the only method that works for dynamic games where computing Nash equilibria in the inner loop is infeasible.

**SEES** is for massive state spaces where neural methods are too slow and tabular methods are impossible. The entire estimation is a single L-BFGS-B call over roughly $K + M$ parameters. There is no neural network training. Cost is $O(M)$, independent of $|\mathcal{S}|$.

**NNES** is for high-dimensional continuous states where you need both neural scalability and publication-grade inference. It is the only neural method where standard errors are theoretically valid, thanks to Neyman orthogonality.

**TD-CCP** is for continuous state variables where you need to understand which value function components drive decisions. The per-feature decomposition provides interpretable diagnostics no other neural method offers. It is model-free.

**MCE-IRL** is for recovering reward weights from expert demonstrations. It is the only inverse estimator recovering interpretable linear reward parameters with bootstrap standard errors and a provable robustness guarantee. The deep variant extends to nonlinear rewards.

**AIRL** is for sim-to-real transfer and anywhere training and deployment dynamics differ. It is the only IRL method with a proven transfer guarantee via the disentanglement theorem.

**f-IRL** is for exploratory reward recovery when you do not know what features matter. It requires zero assumptions about reward structure.

**BC** should always be run first. It tells you whether demonstrations contain enough signal, whether your IRL method is leveraging MDP structure or memorizing frequencies, and provides a calibration target for policy evaluation.

### Capability Matrix

Each row has at least one unique cell.

|  | Efficient MLE | Avoids Bellman | Inverse (demo to reward) | Valid Neural SEs | $O(1)$ in $|\mathcal{S}|$ | Transfer | Feature-Free | Per-Feature Diagnostics | Zero MDP Structure |
|--|---|---|---|---|---|---|---|---|---|
| **NFXP** | **Yes** | | | | | | | | |
| **CCP** | | **Yes** | | | | | | | |
| **MCE-IRL** | | | **Yes + interpretable** | | | | | | |
| **TD-CCP** | | | | | | | | **Yes** | |
| **NNES** | | | | **Yes** | | | | | |
| **SEES** | | | | | **Yes** | | | | |
| **AIRL** | | | | | | **Yes** | | | |
| **f-IRL** | | | | | | | **Yes** | | |
| **BC** | | | | | | | | | **Yes** |
