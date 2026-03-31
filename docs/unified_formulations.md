# Estimators

This page covers the theory behind all econirl estimators, their mathematical formulations, and a guide for choosing between them. All notation follows Rust and Rawat (2026). For code examples, see the [examples](examples/index.rst).

## Theory

### The Problem

Imagine someone making decisions over time. Each period, they see a situation (a "state" $s$), pick an action $a$, get a payoff $r(s,a)$, and move to a new situation $s'$. They care about the future, but not as much as the present, so they discount future payoffs by a factor $\beta$ between 0 and 1. Their goal is to maximize total discounted payoffs $E\{\sum_{t=0}^\infty \beta^t r(s_t, a_t)\}$.

This setup is called a Markov decision process (MDP). The solution is an optimal policy $\delta^*(s) = \arg\max_a Q^*(s,a)$, where $Q^*(s,a)$ is the value of taking action $a$ in state $s$ and then behaving optimally forever after. It satisfies

$$
Q^*(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) \, V^*(s'),
$$

where $V^*(s) = \max_a Q^*(s,a)$. This is the Bellman equation. It says the value of any action equals its immediate payoff plus the discounted value of wherever you end up.

The estimation problem runs this logic backward. We watch people make choices but we do not know their preferences. We want to figure out the payoff function $r$ (or its parameters $\theta$) that explains what we see. Economists call this structural estimation of dynamic discrete choice (DDC) models. Machine learning researchers call it inverse reinforcement learning (IRL). The math is the same.

### Why People in the Same Situation Make Different Choices

The optimal policy $\delta^*(s)$ is deterministic. Everyone in the same state should do the same thing. But in real data, people in the same situation often choose differently. A deterministic model cannot explain this.

Rust (1987) fixed this by adding unobserved "taste shocks" $\varepsilon = (\varepsilon_1, \ldots, \varepsilon_{|\mathcal{A}|})$, one for each action. The agent sees these shocks, but we as researchers do not. The payoff for action $a$ becomes $r(s,a) + \varepsilon_a$. From the agent's perspective, the choice is still deterministic given $(s, \varepsilon)$. But from our perspective, it looks random because we cannot see $\varepsilon$.

Following McFadden (1973), assume the shocks follow an extreme value type 1 (Gumbel) distribution with scale $\sigma > 0$ and mean zero. Also assume the shocks are independent across time periods. These two assumptions buy us closed-form expressions for everything.

### The Soft Bellman Equations

The Gumbel distribution has a remarkable property. When you take the expected maximum over Gumbel-distributed options, you get a simple formula. Averaging $V(s,\varepsilon) = \max_a [Q(s,a) + \varepsilon_a]$ over the shocks gives

$$
V(s) = \sigma \log \sum_{a \in \mathcal{A}(s)} \exp\!\big(Q(s,a)/\sigma\big).
$$

This is called the log-sum-exp function. It is a smooth approximation to the max. When $\sigma$ is small, it behaves like a hard maximum. When $\sigma$ is large, it spreads probability more evenly across actions.

Plugging this into the Bellman equation gives the soft Bellman operator $\Lambda_\sigma$,

$$
\Lambda_\sigma(Q)(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) \; \sigma \log \sum_{a'} \exp\!\big(Q(s',a')/\sigma\big).
$$

This operator is a contraction (it shrinks distances by a factor of $\beta$), so it has a unique fixed point $Q^* = \Lambda_\sigma(Q^*)$.

The probability that someone picks action $a$ in state $s$ is the familiar logit formula,

$$
\pi(a \mid s) = \frac{\exp\!\big(Q(s,a)/\sigma\big)}{\sum_{b} \exp\!\big(Q(s,b)/\sigma\big)}.
$$

Actions with higher $Q$-values get chosen more often. The parameter $\sigma$ controls how sensitive choices are to value differences. Small $\sigma$ means nearly deterministic choices. Large $\sigma$ means nearly random choices.

With a non-uniform base measure $\mu(a \mid s) > 0$, the formula becomes $\pi(a \mid s) = \mu(a \mid s) \exp(Q(s,a)/\sigma) / \sum_b \mu(b \mid s) \exp(Q(s,b)/\sigma)$. The uniform case $\mu = 1/|\mathcal{A}|$ is the standard model.

### The Bridge to Reinforcement Learning

The same equations come from a totally different angle. Suppose an agent picks a policy to maximize expected payoff minus a penalty for straying too far from some default behavior $\mu$,

$$
V(s) = \max_{\pi(\cdot \mid s)} \bigg\{ \sum_a \pi(a \mid s) \, Q(s,a) - \sigma \, \mathrm{KL}\!\big(\pi(\cdot \mid s) \| \mu(\cdot \mid s)\big) \bigg\}.
$$

The solution to this optimization is the same log-sum-exp value and the same logit policy. So the DDC model with taste shocks and the entropy-regularized RL model with a KL penalty are the same thing. The parameter $\sigma$ plays two roles at once. In economics, it is the spread of unobserved taste shocks. In machine learning, it is the strength of the entropy bonus that keeps the policy from becoming too extreme. Rust and Rawat (2026, Appendix A) prove this equivalence formally.

### Notation Summary

| Symbol | Definition |
|--------|-----------|
| $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$ | State, action |
| $\beta \in (0,1)$ | Discount factor |
| $\sigma > 0$ | Scale of taste shocks (or strength of entropy bonus) |
| $\mu(a \mid s) > 0$ | Default (base) policy |
| $\vec{r}(s,a) = (r_1(s,a), \ldots, r_K(s,a))$ | Feature vector |
| $r_\theta(s,a) = \vec{r}(s,a) \cdot \theta$ | Payoff function with parameters $\theta \in \mathbb{R}^K$ |
| $p(s' \mid s,a)$ | Transition probability |
| $Q(s,a)$, $V(s)$ | Action value, state value |
| $\pi(a \mid s)$ | Choice probability (CCP) or policy |
| $\mathcal{D} = \{\tau_i\}_{i=1}^N$ | Observed trajectories, $\tau_i = \{(s_{it}, a_{it})\}_{t=0}^{T_i}$ |
| $R(s,a) \in \mathbb{R}^K$ | Continuation feature vector (discounted future features) |
| $Q_r(s,a)$, $Q_\varepsilon(s,a)$ | Payoff part and taste-shock part of $Q$ |

### Splitting Q into Two Pieces

When the payoff is a weighted sum of features, $r_\theta(s,a) = \vec{r}(s,a) \cdot \theta$, the action value splits neatly into two parts: $Q(s,a) = Q_r(s,a) + Q_\varepsilon(s,a)$.

The first part depends on $\theta$. It equals $Q_r(s,a) = R(s,a)^\top \theta$, where $R$ is the continuation feature vector. Think of $R_k(s,a)$ as "how much of feature $k$ will the agent accumulate, in discounted terms, starting from $(s,a)$." It solves

$$
R(s,a) = \vec{r}(s,a) + \beta \sum_{s'} p(s' \mid s,a) \sum_{a'} \pi(a' \mid s') \, R(s',a').
$$

The second part, $Q_\varepsilon$, captures the option value of having choices in the future. It depends only on the current policy, not on $\theta$,

$$
Q_\varepsilon(s,a) = \beta \sum_{s'} p(s' \mid s,a) \sum_{a'} \pi(a' \mid s') \big[-\sigma \log \pi(a' \mid s') + Q_\varepsilon(s',a')\big].
$$

This split is the foundation of the CCP and TD-CCP estimators. It lets us separate what depends on the unknown parameters from what can be estimated directly from data.

### Likelihood Functions

We observe trajectories $\mathcal{D} = \{\tau_i\}_{i=1}^N$. The full likelihood includes both choices and transitions,

$$
\mathcal{L}^f_\mathcal{D}(\theta) = \prod_{i=1}^N \prod_{t=1}^{T_i} \pi_\theta(a_{it} \mid s_{it}) \, p_\theta(s_{it} \mid s_{it-1}, a_{it-1}).
$$

The partial likelihood uses choices only,

$$
\mathcal{L}^p_\mathcal{D}(\theta) = \prod_{i=1}^N \prod_{t=1}^{T_i} \pi_\theta(a_{it} \mid s_{it}).
$$

Most estimators maximize the partial log-likelihood $\ell^p(\theta) = \log \mathcal{L}^p_\mathcal{D}(\theta)$. The transition part drops out when transitions do not depend on $\theta$.

### Identification

Looking at data alone, we can only tell apart actions whose $Q$-values differ. Specifically,

$$
\log\!\big(\pi(a \mid s) / \pi(a' \mid s)\big) = \big(Q(s,a) - Q(s,a')\big) / \sigma.
$$

We can read off differences in $Q$ but not its level. And since $Q$ builds on $r$, the level of $r$ is not identified either.

In fact, many different reward functions produce the exact same behavior. Ng, Harada, and Russell (1999) showed that for any function $h(s)$,

$$
r_h(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) \, h(s') - h(s)
$$

gives the same policy as $r$. You can shift rewards by any "potential" $h$ and nothing changes. But if the environment changes, these equivalent rewards start to disagree. This is why getting identification right matters for counterfactual analysis.

To pin down $\theta$, Rust and Rawat (2026) use two rules. First, the average reward across actions is zero in every state: $E_{a \sim \mu}[r_\theta(s,a)] = 0$. Second, the gap between two reference $Q$-values equals a known target: $Q^*_\theta(\bar{s}, a^+) - Q^*_\theta(\bar{s}, a^-) = \Delta^*$. Together these remove the ambiguity. An alternative approach, used in CCP and GLADIUS, is to assume the reward of one "anchor" action is known in each state.

### Equivalence Theorem

All estimators on this page share the same soft Bellman foundation. Rust and Rawat (2026, Theorem A.6) show that, with proper identification restrictions, it does not matter what kind of data you start from. Maximum likelihood on choice data (NFXP), feature matching on demonstrations (MCE-IRL), and preference learning on pairwise comparisons (RLHF) all converge to the same policy as the sample grows,

$$
\pi^*_{\hat\theta_{\mathrm{NFXP}}} = \pi^*_{\hat\theta_{\mathrm{IRL}}} = \pi^*_{\hat\theta_{\mathrm{RLHF}}} \;\longrightarrow\; \pi^*_{\theta^\star}.
$$

This is the central result connecting structural econometrics, inverse RL, and RLHF.

---

## Estimators

### Structural Estimators

These estimators start from a known payoff specification $r_\theta(s,a)$ and estimate the parameters $\theta$ from observed choices.

#### NFXP

**Motivation.** To evaluate how well any candidate $\theta$ fits the data, we need the choice probabilities $\pi_\theta$. But computing $\pi_\theta$ requires solving the Bellman equation, which is itself an expensive fixed-point problem. Rust (1987) handled this by nesting the Bellman solve inside the optimizer. For every trial $\theta$, the inner loop solves for $Q_\theta$, the outer loop checks how well the resulting $\pi_\theta$ fits the data, and then tries a better $\theta$. This is expensive but gives exact maximum likelihood with proper standard errors. Iskhakov et al. (2016) sped up the inner loop with a two-phase approach: start with simple iteration (safe but slow), then switch to Newton's method (fast but needs a good starting point) once you are close to the answer.

**Objective.**

$$
\hat\theta_{\mathrm{NFXP}} = \arg\max_\theta \sum_{(s,a) \in \mathcal{D}} \log \pi^*_\theta(a \mid s),
$$

where $\pi^*_\theta$ comes from solving $Q_\theta = \Lambda_\sigma(Q_\theta)$ for each candidate $\theta$. Gradients are computed analytically through the fixed point using the implicit function theorem.

**Pseudocode.**

```
NFXP(D, r_theta, p, beta, sigma):
  1. Initialize theta
  2. Repeat until convergence:
     a. Inner loop — solve for Q_theta:
        Q <- 0
        Repeat (simple iteration):
          Q <- Lambda_sigma(Q; theta)
        Until close enough, then switch to Newton:
          Q <- Q - (I - beta*P_pi)^{-1} (Q - Lambda_sigma(Q))
        Until converged
     b. Compute policy: pi(a|s) = softmax(Q(s,.)/sigma)
     c. Compute log-likelihood: L = sum log pi(a|s) over data
     d. Compute gradient analytically
     e. Update theta (BHHH step)
  3. Standard errors from inverse Hessian
  4. Return theta, SEs
```

NFXP gives the best possible estimates (efficient MLE) with reliable standard errors. The downside is cost: $O(|\mathcal{S}|^2)$ per inner solve. It becomes impractical when $|\mathcal{S}|$ exceeds about 10,000.

#### CCP and NPL

**Motivation.** Hotz and Miller (1993) found a shortcut. Under the Gumbel shock assumption, you can estimate choice probabilities directly from data (just count how often each action is chosen in each state), and then invert them to recover value differences without ever solving the Bellman equation. This replaces thousands of iterations with a single matrix inversion. The tradeoff is that the one-shot estimator is less precise than full MLE. Aguirregabiria and Mira (2002) showed you can get the precision back by iterating: estimate $\theta$, update the choice probabilities, re-estimate $\theta$, and repeat. They called this nested pseudo-likelihood (NPL). After a few rounds, the estimates match what NFXP would give.

**Objective.**

$$
\pi_\theta(a \mid s) = \frac{\exp\!\big(\hat{R}(s,a)^\top \theta + \hat{Q}_\varepsilon(s,a)\big)}{\sum_{a'} \exp\!\big(\hat{R}(s,a')^\top \theta + \hat{Q}_\varepsilon(s,a')\big)},
$$

where $\hat{R}$ and $\hat{Q}_\varepsilon$ come from a single matrix inversion $(I - \beta F_\pi)^{-1}$ using the empirical choice probabilities.

**Pseudocode.**

```
CCP(D, features, p, beta, sigma, K_npl):
  1. Count choices: pi_hat(a|s) = N(s,a) / N(s)
  2. For k = 1 to K_npl:
     a. One matrix inversion to get R and Q_eps
     b. Plug into logit formula to get pi_theta
     c. Maximize partial likelihood over theta
     d. Update pi_hat from new theta (NPL step)
  3. Standard errors from inverse Hessian
  4. Return theta, SEs
```

One step ($K=1$) is fast and consistent. Five to ten NPL steps recover full MLE efficiency.

#### SEES

**Motivation.** NFXP and CCP both need to work with the full state space, either solving the Bellman equation or inverting a big matrix. When there are many states (large or continuous), this becomes impossible. Luo and Sang (2024) proposed a simpler idea: approximate the value function with a small set of basis functions (like polynomials), and optimize the basis coefficients alongside $\theta$. There is no neural network involved. The whole thing is one optimization call.

**Objective.**

$$
(\hat\theta, \hat\alpha) = \arg\max_{\theta, \alpha} \; \ell^f(\theta) - \frac{\lambda}{2} \|\alpha\|^2,
$$

where $V(s;\alpha) = \Psi(s)^\top \alpha$ is a polynomial or Fourier approximation and the penalty keeps the basis from overfitting.

**Pseudocode.**

```
SEES(D, features, p, beta, sigma, basis_type, M, lambda):
  1. Pick a basis: polynomials, Fourier terms, etc., with M terms
  2. Approximate V(s) = sum of basis functions weighted by alpha
  3. Compute Q from the approximated V
  4. Jointly optimize theta and alpha to maximize penalized likelihood
  5. Standard errors via Schur complement (marginalizes out alpha)
  6. Return theta, SEs
```

Cost is $O(M)$ regardless of how many states there are. Works for state spaces over 100,000.

#### NNES

**Motivation.** Basis functions work when the value function is smooth, but can struggle with sharp nonlinearities. Nguyen (2025) replaced the basis with a neural network, which can approximate any shape. The challenge is that neural networks make standard error computation very hard. The breakthrough was proving that the DDC likelihood has a special structure: errors in the neural value function have only a second-order effect on the parameter estimates (Neyman orthogonality). This means the standard errors from the likelihood Hessian are valid even though the neural network is imperfect.

**Objective.**

$$
(\hat\theta, \hat{w}) = \arg\max_{\theta, w} \; \ell^p(\theta) - \lambda \sum_{(s,a,s') \in \mathcal{D}} \big(V_w(s) - r_\theta(s,a) - \beta V_w(s')\big)^2.
$$

**Pseudocode.**

```
NNES(D, features, p, beta, sigma, hidden_dims, lambda):
  1. Initialize theta, neural network V_w
  2. Alternate:
     a. Train V_w to minimize Bellman residual (mini-batch SGD)
     b. Estimate theta by maximizing partial likelihood with V_w held fixed
  3. Standard errors from partial likelihood Hessian (valid by orthogonality)
  4. Return theta, SEs
```

NNES is the only neural method with theoretically valid standard errors.

#### TD-CCP

**Motivation.** Adusumilli and Eckardt (2025) wanted to combine the transparency of CCP with the scalability of neural methods, and to do it without knowing the transition probabilities. Their trick was to learn each piece of the $Q$ decomposition separately using temporal difference (TD) learning on the raw data. For each feature $k$, they train a separate function $R_k$ on observed transitions $(s, a, s', a')$. This is model-free because it never uses $p(s' \mid s,a)$ directly. The per-feature breakdown also helps with debugging: if one component converges but another does not, you know exactly which part of the model is causing trouble.

**Objective.**

$$
\hat\phi = \Big[E_\mathcal{D}\big\{\nu(s,a)\big[\nu(s,a) - \beta \nu(s',a')\big]^\top\big\}\Big]^{-1} E_\mathcal{D}\big\{\nu(s,a) \, \vec{r}(s,a)\big\},
$$

where $\nu(s,a)$ are basis functions and $(s,a,s',a')$ are observed transitions.

**Pseudocode.**

```
TD-CCP(D, features, beta, sigma, basis_functions):
  1. Count choices: pi_hat(a|s) = N(s,a) / N(s)
  2. For each feature k:
     Learn R_k from data using TD regression
  3. Learn Q_eps similarly
  4. Plug R_hat and Q_eps_hat into the logit formula
  5. Maximize partial likelihood over theta
  6. Apply a bias correction for robustness
  7. Return theta, SEs
```

TD-CCP is model-free and gives per-feature diagnostics.

### Inverse Estimators

These estimators start from demonstrated behavior and recover the payoff function.

#### MCE-IRL

**Motivation.** Many different payoff functions can explain the same behavior. Ziebart (2010) proposed a principled way to pick one: among all policies that match the expert's average feature usage, choose the one that is most random (highest entropy). This "maximum entropy" principle avoids over-committing to structure the data does not support. The key refinement was using causal entropy, which only counts randomness in the agent's own decisions and ignores randomness from the environment. The solution turns out to be exactly the logit model from the Theory section, so MCE-IRL is really just maximum likelihood from a different angle.

**Objective.** Let $\mu_\mathcal{D}$ be the expert's average discounted feature usage. The problem is

$$
\max_\pi H_c(\pi) \quad \text{subject to} \quad E_\pi\bigg\{\sum_{t} \beta^t \vec{r}(s_t, a_t)\bigg\} = \mu_\mathcal{D}.
$$

The gradient is simply the gap between expert and model feature averages:

$$
\nabla_\theta \ell^p(\theta) = \mu_\mathcal{D} - E_{\pi_\theta}\bigg\{\sum_{t} \beta^t \vec{r}(s_t, a_t)\bigg\}.
$$

At convergence, the two match.

**Pseudocode.**

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

MCE-IRL also provides a worst-case robustness guarantee (Ziebart 2010, Theorem 3). The deep variant uses a neural network for the payoff but loses interpretability and standard errors.

#### AIRL

**Motivation.** GAIL (Ho and Ermon 2016) framed IRL as a GAN: a discriminator tells apart expert and agent behavior, and the agent learns to fool it. But the "reward" that GAIL learns is entangled with the training environment. Move to a new environment and the learned reward stops working. Fu, Luo, and Levine (2018) fixed this by giving the discriminator a special structure. They split its output into a reward piece $g(s,a)$ and a shaping piece $\beta h(s') - h(s)$. The shaping piece absorbs everything that depends on the environment. What remains, $g$, is the true reward, and it transfers to new settings. Economists will recognize this as the potential-based shaping from Ng, Harada, and Russell (1999).

**Objective.**

$$
f_\phi(s,a,s') = g_\phi(s,a) + \beta \, h_\phi(s') - h_\phi(s), \qquad D_\phi = \frac{\exp(f_\phi)}{\exp(f_\phi) + \pi(a \mid s)}.
$$

**Pseudocode.**

```
AIRL(D_expert, p, beta, sigma, max_rounds):
  1. Initialize reward network g, shaping network h, policy pi
  2. For each round:
     a. Collect transitions from expert and from current policy
     b. Train discriminator D to tell them apart
     c. Use discriminator signal as reward to improve policy
  3. Extract portable reward: r(s,a) = g(s,a)
  4. Return g, pi
```

Under clean conditions (deterministic transitions, state-only reward), $g$ recovers the true reward up to a constant and transfers across environments (Theorem 5.1).

#### f-IRL

**Motivation.** Every other IRL method asks you to choose something about the reward: features (MCE-IRL), a network architecture (deep MCE-IRL), or a discriminator design (AIRL). Ni et al. (2022) asked: can we do IRL with no assumptions at all? Their approach assigns a free reward parameter to every state-action pair and adjusts these parameters to make the model's behavior match the expert's as closely as possible, measured by an f-divergence. You get to choose the divergence: KL for a maximum-likelihood flavor, total variation for robustness to outliers, or chi-squared for sensitivity to distribution differences.

**Objective.**

$$
\min_r D_f(\rho_E \| \rho_{\pi_r}),
$$

where $\rho(s,a)$ measures how often state-action pairs are visited under the discounted policy.

**Pseudocode.**

```
f-IRL(D_expert, p, beta, sigma, divergence, lr):
  1. Estimate how often the expert visits each (s,a)
  2. Initialize reward r(s,a) = 0 everywhere
  3. Repeat:
     a. Solve for optimal policy under current r
     b. Compute how often this policy visits each (s,a)
     c. Adjust r to close the gap (gradient depends on chosen divergence)
  4. Return r (one number per state-action pair)
```

### Model-Free Neural Estimator

#### GLADIUS

**Motivation.** Some previous methods (GAIL, IQ-Learn) only match behavior on states the expert actually visited, leaving the value function undefined elsewhere. Others (NFXP, CCP) need the transition matrix or a big matrix inversion. Kang et al. (2025) built GLADIUS to avoid both problems. It uses two neural networks: one for $Q(s,a)$ and one for the expected next-period value $EV(s,a)$. The key insight is that the standard training objective (TD error) mixes together Bellman error (what we care about) and transition noise (which is irreducible). By using the second network to estimate the noise, GLADIUS isolates the true Bellman error. It needs no transition model. After training, the reward is read off as $r = Q - \beta \cdot EV$, and structural parameters come from projecting this reward onto features.

**Objective.**

$$
\max_{\phi_1} \bigg[\ell^p(\phi_1) - \lambda \, \rho_{BE}(Q_{\phi_1})\bigg].
$$

**Pseudocode.**

```
GLADIUS(D, features, beta, sigma, lambda):
  1. Initialize Q-network and EV-network
  2. For each training epoch:
     a. Sample a mini-batch of (s, a, s') from data
     b. Compute V(s') from Q-network via log-sum-exp
     c. Loss = choice prediction error + lambda * Bellman error
     d. Update Q-network to minimize loss
     e. Update EV-network to track expected V
  3. Recover reward: r(s,a) = Q(s,a) - beta * EV(s,a)
  4. Project reward onto features: theta = least-squares fit
  5. R-squared tells you how linear the learned reward is
  6. Return theta, r, R-squared
```

### Baseline

#### BC

**Motivation.** Before trying anything fancy, check whether the data has a signal at all. Behavioral cloning just counts: how often was each action chosen in each state? No model, no optimization, no value function. If a sophisticated estimator cannot beat this, it is not learning anything useful. Ross et al. (2011) showed that BC errors grow as $O(T^2 \varepsilon)$ with horizon length $T$, while methods that recover the true reward achieve $O(\varepsilon)$ regardless of $T$. That gap is why structural estimation matters.

**Pseudocode.**

```
BC(D):
  1. For each state s, count how often each action a was chosen
  2. pi(a|s) = count(s,a) / count(s)
  3. Return pi
```

---

## Guide

This section helps you choose the right estimator.

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

**NFXP-NK** is for when you need the best possible estimates with reliable standard errors on a manageable state space. Think published papers, confidence intervals, hypothesis tests. Use it for classic problems like bus engine replacement or occupational choice.

**CCP** is for when NFXP is too slow. It skips the Bellman solve entirely. One step gives quick estimates, five to ten NPL steps match NFXP quality. It is the only option for dynamic games.

**SEES** is for big state spaces without neural networks. One optimization call, no training loops. Scales past 100,000 states.

**NNES** is for big state spaces when you need standard errors you can trust. It is the only neural method with theoretically valid inference.

**TD-CCP** is for continuous states when you want to see which features are easy or hard to predict. It works without knowing the transition probabilities.

**MCE-IRL** is for learning reward weights from demonstrations with standard errors. It is the bridge between IRL and structural econometrics.

**AIRL** is for when the learned reward needs to work in a different environment than the one it was trained in.

**f-IRL** is for exploring what the reward looks like when you have no idea which features matter.

**BC** should always be your first step. If nothing beats it, the data may not have enough structure for model-based methods.

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
