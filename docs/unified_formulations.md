# Estimators

This page presents the theory, mathematical formulations, and selection guide for all econirl estimators using common notation from Rust and Rawat (2026). For worked code examples, see the [examples](examples/index.rst).

## Theory

All estimators operate on a single soft Bellman system. The agent chooses action $a \in \mathcal{A}(s)$ in state $s \in \mathcal{S}$ to maximize discounted utility subject to idiosyncratic taste shocks.

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
| $R(s,a) \in \mathbb{R}^K$ | Discounted continuation feature vector (per-feature expected value) |
| $Q_r(s,a)$, $Q_\varepsilon(s,a)$ | Reward component and entropy correction of $Q$ |

### Soft Bellman Equations

The choice-specific value $Q$ is the unique fixed point of the soft Bellman operator $\Lambda_\sigma$, defined by

$$
\Lambda_\sigma(Q)(s,a) = r_\theta(s,a) + \beta \sum_{s'} p(s' \mid s,a) \; \sigma \log \sum_{a' \in \mathcal{A}(s')} \exp\!\big(Q(s',a')/\sigma\big).
$$

The expected value function $V$ is the log-sum-exp aggregation over actions,

$$
V(s) = \sigma \log \sum_{a \in \mathcal{A}(s)} \exp\!\big(Q(s,a)/\sigma\big).
$$

The conditional choice probability (softmax policy) is

$$
\pi(a \mid s) = \frac{\mu(a \mid s) \exp\!\big(Q(s,a)/\sigma\big)}{\sum_{b \in \mathcal{A}(s)} \mu(b \mid s) \exp\!\big(Q(s,b)/\sigma\big)}.
$$

When the base policy is uniform, $\mu(a \mid s) = 1/|\mathcal{A}(s)|$, this reduces to the standard multinomial logit. The soft Bellman operator $\Lambda_\sigma$ is a $\beta$-contraction, so the fixed point $Q^* = \Lambda_\sigma(Q^*)$ exists and is unique.

### Variational Identity

The value function solves a KL-penalized optimization at each state. By the Fenchel-Moreau (Donsker-Varadhan) duality,

$$
V(s) = \max_{\pi(\cdot \mid s) \in \Delta(\mathcal{A})} \bigg\{ \sum_a \pi(a \mid s) \, Q(s,a) - \sigma \, \mathrm{KL}\!\big(\pi(\cdot \mid s) \| \mu(\cdot \mid s)\big) \bigg\},
$$

and the maximizer is the softmax policy $\pi(a \mid s)$ above. This establishes that the DDC multinomial logit model and entropy-regularized reinforcement learning solve the same optimization problem.

### DDC Micro-Foundation

The soft Bellman system arises from a dynamic discrete choice model with additive taste shocks. When $\varepsilon_a$ are i.i.d. mean-centered extreme value type 1 draws with scale $\sigma$, the agent chooses $a^* \in \arg\max_a \{Q^*_\theta(s,a) + \sigma \log \mu(a \mid s) + \varepsilon_a\}$. Integrating over $\varepsilon$ recovers the log-sum-exp value and the softmax CCP. This provides the structural micro-foundation for all estimators on this page.

### Q Decomposition

When the reward is linear, $r_\theta(s,a) = \vec{r}(s,a) \cdot \theta$, the choice-specific value decomposes as $Q(s,a) = Q_r(s,a) + Q_\varepsilon(s,a)$. The reward component satisfies $Q_r(s,a) = R(s,a)^\top \theta$ where the continuation feature matrix $R$ solves

$$
R(s,a) = \vec{r}(s,a) + \beta \sum_{s'} p(s' \mid s,a) \sum_{a'} \pi(a' \mid s') \, R(s',a').
$$

The entropy correction $Q_\varepsilon$ solves the analogous system with $-\sigma \log \pi(a' \mid s')$ replacing the feature vector,

$$
Q_\varepsilon(s,a) = \beta \sum_{s'} p(s' \mid s,a) \sum_{a'} \pi(a' \mid s') \big[-\sigma \log \pi(a' \mid s') + Q_\varepsilon(s',a')\big].
$$

### Likelihood Functions

The full likelihood conditions on both choices and transitions,

$$
\mathcal{L}^f_\mathcal{D}(\theta) = \prod_{i=1}^N \prod_{t=1}^{T_i} \pi_\theta(a_{it} \mid s_{it}) \, p_\theta(s_{it} \mid s_{it-1}, a_{it-1}).
$$

The partial likelihood conditions on choices only,

$$
\mathcal{L}^p_\mathcal{D}(\theta) = \prod_{i=1}^N \prod_{t=1}^{T_i} \pi_\theta(a_{it} \mid s_{it}).
$$

Most estimators maximize the log partial likelihood $\ell^p(\theta) = \log \mathcal{L}^p_\mathcal{D}(\theta)$ because the transition component does not depend on $\theta$ when transitions are estimated nonparametrically.

### Identification

Rewards in dynamic discrete choice and IRL models are not uniquely identified from observed behavior without additional restrictions. From the softmax CCP, only Q-value differences are recoverable from choice data,

$$
\log\!\big(\pi(a \mid s) / \pi(a' \mid s)\big) = \big(Q(s,a) - Q(s,a')\big) / \sigma.
$$

The level of $Q$ is not identified, so neither is the level of $r$. Ng, Harada, and Russell (1999) showed that the set of observationally equivalent rewards forms an equivalence class under potential-based shaping. For any function $h \colon \mathcal{S} \to \mathbb{R}$,

$$
r_h(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) \, h(s') - h(s)
$$

yields $\pi_{r_h} = \pi_r$. All rewards in this class generate the same policy under the training dynamics. However, under alternative transitions $\lambda(s' \mid s,a) \neq p(s' \mid s,a)$, the counterfactual policies diverge because $\pi_{r_h} = f(\beta, r_h, \lambda) \neq f(\beta, r, \lambda)$.

Rust and Rawat (2026) adopt two normalizations that together pin down $\theta$. The state-potential gauge removes the additive state constant,

$$
E_{a \sim \mu(\cdot \mid s)}\big[r_\theta(s,a)\big] = 0 \quad \text{for all } s \in \mathcal{S}.
$$

The reference Q-gap fixes the scale,

$$
Q^*_\theta(\bar{s}, a^+) - Q^*_\theta(\bar{s}, a^-) = \Delta^*,
$$

for a chosen reference state $\bar{s}$ and action pair $(a^+, a^-)$ where $\Delta^*$ is known from calibration or simulation. Under mild conditions, the gap function $\lambda \mapsto \Delta(\lambda)$ is continuous and strictly increasing, so there is a unique scale $\lambda^*$ satisfying $\Delta(\lambda^*) = \Delta^*$.

An alternative identification strategy is the normalizing (anchor) action assumption used in the CCP and GLADIUS estimators. For each state $s$, there exists an action $a_s$ with known reward $r(s, a_s)$. This provides $|\mathcal{S}|$ restrictions, making the remaining $|\mathcal{S}|(|\mathcal{A}|-1)$ unknown rewards exactly identified from the same number of CCP parameters.

### Equivalence

All estimators on this page operate on the same soft Bellman system. Rust and Rawat (2026, Theorem A.6) establish that under the soft-control framework with the gauge normalizations above, consistent estimators from different data sources converge to the same policy. If $\hat\theta_{\mathrm{NFXP}}$ (from choice data), $\hat\theta_{\mathrm{IRL}}$ (from expert demonstrations), and $\hat\theta_{\mathrm{RLHF}}$ (from pairwise preferences) each solve their respective objectives, then as sample size grows,

$$
\pi^*_{\hat\theta_{\mathrm{NFXP}}} = \pi^*_{\hat\theta_{\mathrm{IRL}}} = \pi^*_{\hat\theta_{\mathrm{RLHF}}} \;\longrightarrow\; \pi^*_{\theta^\star}.
$$

The choice of estimation method does not affect the limiting policy, provided all methods use the same soft-control framework and apply consistent identification restrictions. This equivalence between maximum likelihood (DDC), maximum causal entropy (IRL), and Bradley-Terry preference learning (RLHF) is the central theoretical result unifying the three fields.

---

## Estimators

### Structural Estimators

Structural estimators recover utility parameters $\theta$ from observed choices, taking the direction $\theta \to \pi$.

#### NFXP

Rust (1987) introduced nested fixed point estimation for dynamic discrete choice. Iskhakov, Jorgensen, Rust, and Schjerning (2016) added the SA-to-NK polyalgorithm for the inner loop.

The NFXP estimator maximizes the exact log-likelihood,

$$
\hat\theta_{\mathrm{NFXP}} = \arg\max_\theta \sum_{(s,a) \in \mathcal{D}} \log \pi^*_\theta(a \mid s),
$$

where $\pi^*_\theta$ is computed by solving the soft Bellman fixed point $Q_\theta = \Lambda_\sigma(Q_\theta)$ at each candidate $\theta$. The outer loop uses BHHH optimization, which forms a positive-definite Hessian approximation from per-observation score outer products. The inner loop starts with successive approximation (globally convergent) and switches to Newton-Kantorovich (quadratically convergent) near the fixed point. The gradient $\nabla_\theta \ell$ is computed analytically via the implicit function theorem through the Frechet derivative $(I - \beta P_\pi)^{-1}$, avoiding finite differences or backpropagation.

NFXP is the only estimator that delivers statistically efficient maximum likelihood estimates with analytical standard errors. Its cost is $O(|\mathcal{S}|^2 \times \text{inner iterations})$ per outer step, making it intractable for $|\mathcal{S}| > 10{,}000$ or $\beta > 0.995$.

#### CCP and NPL

Hotz and Miller (1993) showed that observed choice probabilities directly invert to value differences without solving the Bellman equation. Aguirregabiria and Mira (2002) introduced the nested pseudo-likelihood (NPL) iteration that recovers MLE efficiency.

The CCP estimator substitutes first-stage nonparametric estimates into the partial likelihood. The CCP takes the form

$$
\pi_\theta(a \mid s) = \frac{\exp\!\big(\hat{R}(s,a)^\top \theta + \hat{Q}_\varepsilon(s,a)\big)}{\sum_{a'} \exp\!\big(\hat{R}(s,a')^\top \theta + \hat{Q}_\varepsilon(s,a')\big)},
$$

where $\hat{R}$ and $\hat{Q}_\varepsilon$ are computed once from nonparametric CCP estimates $\hat\pi(a \mid s) = N(s,a)/N(s)$ by solving the linear systems for $R$ and $Q_\varepsilon$ via a single matrix inversion $(I - \beta F_\pi)^{-1}$, which costs $O(|\mathcal{S}|^3)$. The estimator then maximizes $\mathcal{L}^p_\mathcal{D}(\theta)$.

NPL iterates this procedure. At step $k$, it re-estimates the CCPs from $\hat\theta_{k-1}$ and resolves the value function. One step ($K=1$) gives a consistent but inefficient Hotz-Miller estimate. Five to ten steps typically recover MLE efficiency. At the NPL fixed point, $\partial V / \partial \pi = 0$, the zero Jacobian property. This Neyman orthogonality means first-stage CCP noise does not affect $\hat\theta$ asymptotically.

#### SEES

Luo and Sang (2024) introduced a sieve-based estimator that approximates $V(s)$ with a basis expansion rather than solving the Bellman inner loop.

The value function is parameterized as $V(s;\alpha) = \Psi(s)^\top \alpha$ where $\Psi(s) = (\psi_1(s), \ldots, \psi_M(s))$ is a vector of basis functions (Chebyshev polynomials or Fourier terms). The estimator jointly optimizes structural parameters $\theta$ and basis coefficients $\alpha$ via penalized maximum likelihood,

$$
(\hat\theta, \hat\alpha) = \arg\max_{\theta, \alpha} \; \ell^f(\theta) - \frac{\lambda}{2} \|\alpha\|^2,
$$

where $Q(s,a;\theta,\alpha) = r_\theta(s,a) + \beta \sum_{s'} p(s' \mid s,a) \, \Psi(s')^\top \alpha$ and the penalty enforces Bellman consistency. Standard errors for $\theta$ use the Schur complement $H_{\theta\theta} - H_{\theta\alpha} H_{\alpha\alpha}^{-1} H_{\alpha\theta}$ to marginalize out the nuisance parameter $\alpha$.

SEES requires no neural network training. The basis matrix is deterministic and the entire estimation is a single L-BFGS-B call over roughly $K + M$ parameters. Cost is $O(M)$, independent of $|\mathcal{S}|$, making it tractable for state spaces exceeding 100,000 where neural and tabular methods fail. The estimator achieves Cramer-Rao efficiency as the sieve dimension $M$ grows with sample size.

#### NNES

Nguyen (2025) replaced the sieve basis with a neural network while preserving valid standard errors through Neyman orthogonality.

The value function $V_w(s)$ is parameterized by a neural network with weights $w$. Like SEES, NNES jointly optimizes structural parameters and the value approximation,

$$
(\hat\theta, \hat{w}) = \arg\max_{\theta, w} \; \ell^p(\theta) - \lambda \sum_{(s,a,s') \in \mathcal{D}} \big(V_w(s) - r_\theta(s,a) - \beta V_w(s')\big)^2,
$$

using NPL-style iteration. The gradient $\nabla_\theta Q$ is computed via equilibrium propagation, avoiding the intractable Hessian $\nabla^2_{ww}$ of the neural network.

The key theoretical result is that the DDC likelihood score is orthogonal to $V$-approximation error (Neyman orthogonality). Specifically, $\partial \ell(\theta_0, \pi^*, V^*) / \partial \pi = 0$ at the true parameters. This means $\hat\theta$ is $\sqrt{n}$-consistent even when $V_w$ converges at the slower rate $o_p(n^{-1/4})$. The estimator achieves the semiparametric efficiency bound, $\sqrt{n}(\hat\theta - \theta_0) \to \mathcal{N}(0, \Sigma^{-1})$, making it the only neural method with theoretically valid standard errors.

#### TD-CCP

Adusumilli and Eckardt (2025) combined temporal difference learning with the CCP estimator to avoid both the Bellman inner loop and explicit transition estimation.

The key innovation is per-feature decomposition. Instead of a monolithic value function, TD-CCP learns $K+1$ separate functions, one for each reward feature $R_k(s,a)$ plus the entropy correction $Q_\varepsilon(s,a)$, via temporal difference regression on observed transitions. With basis functions $\nu(s,a)$ and parameters $\phi$, the TD estimate is

$$
\hat\phi = \Big[E_\mathcal{D}\big\{\nu(s,a)\big[\nu(s,a) - \beta \nu(s',a')\big]^\top\big\}\Big]^{-1} E_\mathcal{D}\big\{\nu(s,a) \, \vec{r}(s,a)\big\},
$$

where $E_\mathcal{D}$ denotes the empirical expectation over observed transitions $(s,a,s',a')$. This is model-free because it uses observed next-state samples rather than the transition matrix $p(s' \mid s,a)$.

The structural parameters $\theta$ are then estimated by maximizing the partial likelihood $\mathcal{L}^p_\mathcal{D}(\theta)$ using the CCP form with $\hat{R}(s,a) = \nu(s,a)^\top \hat\phi$ substituted for the true continuation features. A third stage applies a Neyman-orthogonal score correction to make $\hat\theta$ robust to first-stage estimation noise in $\hat{R}$ and $\hat{Q}_\varepsilon$. The per-feature decomposition provides interpretable diagnostics. If $R_3$ has high approximation error but $R_1$ converges, the third feature's continuation-value structure is the modeling challenge.

### Inverse Estimators

Inverse estimators recover reward parameters $\theta$ from expert demonstrations, taking the direction $\pi \to \theta$.

#### MCE-IRL

Ziebart (2010) derived the optimal policy from maximum causal entropy subject to feature matching. Let $\mu_\mathcal{D} = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} \beta^t \vec{r}(s_{it}, a_{it})$ denote the expert's discounted feature counts.

The MCE-IRL primal problem is

$$
\max_\pi H_c(\pi) \quad \text{subject to} \quad E_\pi\bigg\{\sum_{t=0}^T \beta^t \vec{r}(s_t, a_t)\bigg\} = \mu_\mathcal{D},
$$

where $H_c(\pi) = E_\pi\{-\sum_t \beta^t \log \pi(a_t \mid s_t)\}$ is causal entropy, which conditions each action only on information available at decision time. The Lagrange multipliers of this constrained problem are the reward parameters $\theta$, and the KKT conditions yield exactly the softmax policy from the common framework.

The gradient of the log partial likelihood is the feature matching residual,

$$
\nabla_\theta \ell^p(\theta) = \mu_\mathcal{D} - E_{\pi_\theta}\bigg\{\sum_{t=0}^T \beta^t \vec{r}(s_t, a_t)\bigg\}.
$$

At the optimum, expert and model feature expectations match, confirming that maximum causal entropy IRL is maximum likelihood estimation under the softmax Bellman model. The algorithm iterates three steps. A backward pass solves the soft Bellman equations for $Q_\theta$ and $V_\theta$. A forward pass computes state-action visitation frequencies under $\pi_\theta$. A gradient step updates $\theta \leftarrow \theta + \alpha(\mu_\mathcal{D} - E_{\pi_\theta}[\cdot])$.

MCE-IRL minimizes worst-case prediction log-loss (Ziebart 2010, Theorem 3), making the recovered policy maximally robust to distribution shift while matching observed features. The deep variant replaces the linear reward with a neural network $r_\phi(s,a)$ but loses interpretable parameters and standard errors.

#### AIRL

Fu, Luo, and Levine (2018) introduced adversarial IRL with a structured discriminator that disentangles reward from dynamics-dependent shaping.

The discriminator logits decompose as

$$
f_\phi(s,a,s') = g_\phi(s,a) + \beta \, h_\phi(s') - h_\phi(s),
$$

where $g_\phi$ is the reward approximator and $h_\phi$ is a learned shaping potential. The discriminator classifies expert versus policy transitions,

$$
D_\phi(s,a,s') = \frac{\exp\!\big(f_\phi(s,a,s')\big)}{\exp\!\big(f_\phi(s,a,s')\big) + \pi(a \mid s)}.
$$

Training alternates between updating the discriminator to distinguish expert from generated $(s,a,s')$ triples and updating the policy $\pi$ via RL with reward signal $f_\phi$. After convergence, the portable reward is $r(s,a) = g_\phi(s,a)$.

Under deterministic dynamics and a true state-only reward with $g_\phi$ restricted to depend only on $s$, Fu et al. prove that at the optimum $g_\phi(s) = r(s) + \text{const}$ and $h_\phi(s) = V(s) + \text{const}$ (Theorem 5.1). The learned reward is disentangled from dynamics and transfers across environments. Without the state-only restriction or under stochastic dynamics, the decomposition is approximate. Lee, Sudhir, and Wang (2026) show that an economic normalizing action (exit option with known payoff zero) provides an alternative identification strategy for action-dependent rewards.

#### f-IRL

Ni, Sikchi, Wang, and Bhatt (2022) formulated IRL as f-divergence minimization between state-action occupancy measures, requiring no assumptions about reward structure.

Let $\rho_E(s,a) = E_{\pi_E}\{\sum_{t=0}^\infty \beta^t \mathbb{I}(s_t = s, a_t = a)\}$ be the expert's discounted occupancy measure and $\rho_\pi$ the current policy's occupancy measure. The objective is

$$
\min_r D_f(\rho_E \| \rho_{\pi_r}),
$$

where $D_f$ is an f-divergence. The tabular reward gradient depends on the choice of divergence. For KL divergence the gradient is $\log(\rho_E / \rho_\pi)$. For chi-squared divergence it is $(\rho_E / \rho_\pi) - 1$. For total variation it is $\mathrm{sign}(\rho_E - \rho_\pi)$.

The algorithm estimates $\rho_E$ empirically from demonstrations, then iterates between solving the soft Bellman system under the current reward to get $\pi_r$ and $\rho_{\pi_r}$, computing the divergence gradient, and updating the tabular reward $r(s,a)$. The choice of f-divergence gives a menu of robustness properties. KL recovers maximum likelihood equivalence. Total variation is robust to outlier demonstrations. Chi-squared is sensitive to variance in occupancy ratios.

### Model-Free Neural Estimator

#### GLADIUS

Kang et al. (2025) introduced GLADIUS (Gradient-based Learning with Ascent-Descent for Inverse Utility learning from Samples), a model-free estimator using two neural networks.

GLADIUS parameterizes $Q_{\phi_1}(s,a)$ and $EV_{\phi_2}(s,a) = E_p\{V(s') \mid s,a\}$ separately. The mean squared TD error decomposes as

$$
\rho_{TD}(Q) = \rho_{BE}(Q) + \beta^2 E_p\big\{[V(s') - EV(s,a)]^2\big\},
$$

where $\rho_{BE}$ is the mean squared Bellman error and the second term is conditional variance of next-period values. GLADIUS uses a max-min formulation. The outer maximization finds $\phi_1$ to maximize the penalized partial likelihood,

$$
\max_{\phi_1} \bigg[\ell^p(\phi_1) - \lambda \, \rho_{BE}(Q_{\phi_1})\bigg],
$$

while the inner minimization finds $\phi_2$ to estimate the conditional variance, separating Bellman error from transition noise. This targets the correct Bellman error without requiring $p(s' \mid s,a)$.

After training, the reward is recovered as $r(s,a) = Q_{\hat\phi_1}(s,a) - \beta \, EV_{\hat\phi_2}(s,a)$ via the soft Bellman identity. Structural parameters are extracted by projecting the neural reward onto features, $\hat\theta = (\Phi^\top \Phi)^{-1} \Phi^\top \hat{r}$, where $\Phi$ is the feature matrix. The projection $R^2$ measures how much of the neural reward is explained by the linear specification. Like the CCP estimator, GLADIUS uses a normalizing action for identification. Under realizability assumptions, the estimator achieves global convergence with error $O(1/T) + O(1/N)$.

### Baseline

#### BC

Pomerleau (1991) introduced behavioral cloning. Ross, Gordon, and Bagnell (2011) characterized its error compounding.

The BC estimator is the empirical frequency,

$$
\hat\pi(a \mid s) = \frac{N(s,a)}{N(s)}.
$$

BC uses zero MDP structure. There is no reward, no value function, no Bellman equation, and no transition model. It establishes the floor that any structural or inverse method must beat. If an estimator cannot outperform BC, it has not learned from sequential decision-making structure.

Under distribution shift, BC error compounds quadratically with the horizon, $\text{Error} = O(T^2 \varepsilon)$, where $\varepsilon$ is per-step error (Ross et al. 2011). IRL methods that recover the true reward achieve $O(\varepsilon)$ regardless of $T$, demonstrating the value of structural estimation.

---

## Guide

This section explains why each estimator exists, what theorem makes it unique, and when to use it. Every estimator occupies a point in the capability space that no other method covers.

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

### Decision Flowchart

```
Do you have a parametric utility model u(s,a;θ)?
├── YES (structural estimation — recover θ)
│   ├── Tabular state space (|S| < 10K)?
│   │   ├── Need MLE efficiency + analytical SEs? → NFXP-NK
│   │   └── Need speed / model selection / games? → CCP (NPL)
│   └── Large / continuous state space?
│       ├── Need valid standard errors? → NNES
│       ├── Need per-feature diagnostics? → TD-CCP
│       └── Need fastest possible estimation? → SEES
│
└── NO (inverse RL — recover reward from demonstrations)
    ├── Know what features matter?
    │   ├── Need interpretable linear weights + SEs? → MCE-IRL
    │   ├── Need nonlinear reward? → MCE-IRL (Deep)
    │   └── Need reward that transfers across environments? → AIRL
    ├── Don't know what features matter? → f-IRL
    └── Just need a baseline? → BC
```

### When to Use Each Estimator

**NFXP-NK** is the structural MLE gold standard (Rust 1987, Iskhakov et al. 2016). It maximizes the exact log-likelihood where choice probabilities come from solving the Bellman equation to machine precision at each outer step. The SA-to-NK polyalgorithm switches from successive approximation to Newton-Kantorovich near the fixed point. Analytical gradients via the implicit function theorem differentiate through the Bellman fixed point without finite differences. NFXP is irreplaceable for publication-grade structural estimates on tabular problems with proper confidence intervals, likelihood ratio tests, and information criteria.

**CCP** is fast structural estimation via the Hotz-Miller Inversion Lemma (Hotz and Miller 1993, Aguirregabiria and Mira 2002). Under additive separability and IID private shocks, observed choice probabilities uniquely invert to value differences. Value recovery is a single matrix inversion $O(|\mathcal{S}|^3)$ rather than thousands of Bellman iterations. The Aguirregabiria-Mira NPL iteration recovers MLE efficiency when iterated five to ten steps. CCP is irreplaceable for rapid specification search and dynamic games where computing Nash equilibria in the inner loop is infeasible.

**MCE-IRL** bridges economics and machine learning (Ziebart 2010). The distribution maximizing causal entropy subject to feature matching has a recursive closed-form solution that is exactly the softmax Bellman equation. The gradient is the feature matching residual. MCE-IRL minimizes worst-case prediction log-loss, making the recovered policy maximally robust to distribution shift. It is the only inverse estimator recovering interpretable linear reward parameters with bootstrap standard errors and a provable robustness guarantee. The deep variant (Wulfmeier et al. 2016) extends to nonlinear rewards via neural networks.

**TD-CCP** provides neural approximate value iteration with per-feature decomposition (Adusumilli and Eckardt 2025). Instead of a monolithic value function, TD-CCP learns $K+1$ separate neural networks, one per utility feature plus an entropy network. The feature decomposition provides interpretable per-component diagnostics no other neural method offers. If one component has high loss while another converges, you know which feature's continuation value structure is the modeling challenge. TD-CCP is irreplaceable for continuous state variables where discretization introduces massive approximation error.

**NNES** is the neural V-network with valid inference (Nguyen 2025). The DDC likelihood score is orthogonal to $V$-approximation error (Neyman orthogonality), so $\hat\theta$ is $\sqrt{n}$-consistent even when the neural value function converges at a slower rate. The estimator achieves the semiparametric efficiency bound with no bias correction. NNES is irreplaceable as the only neural method where standard errors are theoretically valid.

**SEES** is the fastest scalable estimator (Luo and Sang 2024). The value function is approximated by a sieve basis (Fourier or Chebyshev) and the entire estimation is a single L-BFGS-B call over roughly $K + M$ parameters. There is no neural network training, no SGD, no mini-batches, and no learning rates. Cost is $O(M)$, independent of $|\mathcal{S}|$, scaling to state spaces exceeding 100,000 where neural methods hit memory limits.

**AIRL** provides adversarial reward recovery with transfer guarantees (Fu, Luo, and Levine 2018). The discriminator structure forces the learned reward to be state-only by canceling potential-based shaping at optimality. Under deterministic dynamics and state-only rewards, the disentanglement theorem proves that the recovered reward transfers across environments. AIRL is irreplaceable for sim-to-real transfer and anywhere training and deployment dynamics differ.

**f-IRL** performs feature-free distribution matching (Ni et al. 2022). It minimizes the f-divergence between state-action occupancy measures, requiring zero assumptions about reward structure. No features to design, no neural architecture to choose, no discriminator to train. The choice of f-divergence gives a menu of robustness properties. f-IRL is irreplaceable for exploratory reward recovery when you do not know what features matter.

**BC** is the honest baseline (Pomerleau 1991, Ross et al. 2011). It uses zero MDP structure, just the empirical frequency $\hat\pi(a \mid s) = N(s,a)/N(s)$. BC establishes the floor that any method must beat. Error compounds as $O(T^2 \varepsilon)$ under distribution shift, while IRL methods achieve $O(\varepsilon)$ regardless of horizon.

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
