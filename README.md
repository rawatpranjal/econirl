# econirl

Benchmarking dynamic discrete choice and inverse RL algorithms on a variety of MDPs — comparing reward recovery, imitation, and generalization.

## Install

```bash
uv pip install -e .
```

## Try It

```python
from econirl.evaluation.benchmark import BenchmarkDGP, run_single, get_default_estimator_specs

# 5-state bus engine replacement MDP (Rust 1987)
dgp = BenchmarkDGP(n_states=5, discount_factor=0.95)
specs = get_default_estimator_specs()

# Run all 17 estimators with benchmark-tuned defaults
for spec in specs:
    result = run_single(dgp, spec, n_agents=100, n_periods=50, seed=42)
    print(f"{result.estimator:12s}  {result.pct_optimal:6.1f}%  {result.time_seconds:5.1f}s")
```

![5-State Bus Engine Replacement MDP](docs/mdp_data_generation.gif)

### Results

| Estimator   | Category    | Recovers Params | Recovers Reward | % Optimal | % Transfer | Time   |
|-------------|-------------|:---------------:|:---------------:|----------:|-----------:|-------:|
| **Structural Estimators** | | | | | | |
| NFXP        | Structural  | Yes | Yes |  99.7% |  99.8% |  13.9s |
| CCP         | Structural  | Yes | Yes |  99.7% |  99.8% |  18.6s |
| SEES        | Structural  | Yes | Yes |  99.6% |  99.6% |  28.6s |
| NNES        | Structural  | Yes | Yes |  99.6% |  99.1% |  13.7s |
| **Entropy-Based IRL** | | | | | | |
| MCE IRL     | IRL         | Yes | Yes |  99.7% |  99.7% |  20.6s |
| MaxEnt IRL  | IRL         | No  | Yes |  98.2% |  97.8% |   9.1s |
| Deep MaxEnt | IRL         | No  | Yes |  98.3% |  98.2% |  52.3s |
| BIRL        | IRL         | No  | Yes |  99.5% |  99.5% | 237.8s |
| **Margin-Based IRL** | | | | | | |
| Max Margin  | IRL         | Yes | Yes |  99.3% |  99.3% |  64.8s |
| Max Margin IRL | IRL      | No  | Yes |  31.1% |  34.2% |   0.3s |
| **Distribution Matching** | | | | | | |
| f-IRL       | IRL         | No  | Yes |  99.1% |  99.1% |  44.9s |
| **Neural Estimators** | | | | | | |
| TD-CCP      | Neural      | Yes | Yes |  99.8% |  99.7% |  16.3s |
| GLADIUS     | Neural      | Yes | Yes |  99.6% |  88.7% |   4.2s |
| **Adversarial Methods** | | | | | | |
| GAIL        | Adversarial | No  | No  |  54.3% |  50.9% | 112.9s |
| AIRL        | Adversarial | No  | Yes |  99.4% |  99.5% | 123.0s |
| GCL         | Adversarial | No  | Yes |  92.7% |  95.3% | 166.5s |
| **Baseline** | | | | | | |
| BC          | Baseline    | No  | No  |  99.5% |  99.5% |   0.1s |

5-state MDP, 100 agents x 50 periods, seed=42. **% Optimal** = value achieved vs true optimal on training dynamics (baseline-normalized). **% Transfer** = same metric on held-out transition dynamics (same rewards, different wear rates). **Recovers Params** = recovers interpretable structural parameters. **Recovers Reward** = recovers a reward function (enables transfer to new dynamics).

![Internal Validity — Policy Execution on Training Dynamics](docs/internal_validity.gif)

![External Validity — Policy Execution on Transfer Dynamics](docs/external_validity.gif)

![Estimated vs True Rewards](docs/reward_heatmaps.png)

## Algorithms

### Structural Estimators

Assume the econometrician knows the model and recover flow utility parameters by maximum likelihood.

| Algorithm | Paper | Method |
|-----------|-------|--------|
| NFXP      | [Rust (1987)](https://doi.org/10.2307/1911259) | Full-solution MLE via nested fixed point |
| CCP       | [Hotz & Miller (1993)](https://doi.org/10.2307/2298122) | Two-step conditional choice probability with NPL iterations |
| SEES      | [Luo & Sang (2024)](https://arxiv.org/abs/2404.12843) | Sieve basis V(s) approximation + penalized joint MLE |
| NNES      | [Nguyen (2025)](https://arxiv.org/abs/2501.14375) | Neural V(s) network (Bellman residual) + structural MLE |

### Entropy-Based IRL

Recover reward functions from demonstrations using maximum entropy or Bayesian principles.

| Algorithm   | Paper | Method |
|-------------|-------|--------|
| MCE IRL     | [Ziebart (2010)](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf) | Maximum causal entropy IRL with soft value iteration |
| MaxEnt IRL  | [Ziebart et al. (2008)](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf) | Maximum entropy IRL with state visitation frequencies |
| Deep MaxEnt | [Wulfmeier et al. (2016)](https://arxiv.org/abs/1507.04888) | Neural reward network + MaxEnt feature matching |
| BIRL        | [Ramachandran & Amir (2007)](https://www.ijcai.org/Proceedings/07/Papers/416.pdf) | Bayesian MCMC (Metropolis-Hastings) over reward parameters |

### Margin-Based IRL

Recover rewards by maximizing the margin between expert and non-expert behavior.

| Algorithm      | Paper | Method |
|----------------|-------|--------|
| Max Margin     | [Ratliff et al. (2006)](https://doi.org/10.1145/1143844.1143936) | Structured max-margin planning |
| Max Margin IRL | [Abbeel & Ng (2004)](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf) | Apprenticeship learning via margin maximization |

### Distribution Matching

Match state-marginal distributions rather than feature expectations.

| Algorithm | Paper | Method |
|-----------|-------|--------|
| f-IRL     | [Ni et al. (2022)](https://arxiv.org/abs/2011.04709) | State-marginal matching via f-divergences (KL, chi-squared, TV) |

### Neural Estimators

Approximate value functions with neural networks for scalability to large state spaces.

| Algorithm | Paper | Method |
|-----------|-------|--------|
| TD-CCP    | [Adusumilli & Eckardt (2022)](https://arxiv.org/abs/1912.09509) | TD-learning + CCP with neural approximate value iteration |
| GLADIUS   | [Kang, Yoganarasimhan & Jain (2025)](https://arxiv.org/abs/2502.14131) | Dual Q + EV networks with Bellman consistency penalty |

### Adversarial Methods

Learn reward or policy via a discriminator that distinguishes expert from generated behavior.

| Algorithm | Paper | Method |
|-----------|-------|--------|
| GAIL      | [Ho & Ermon (2016)](https://arxiv.org/abs/1611.03852) | Generative adversarial imitation learning |
| AIRL      | [Fu et al. (2018)](https://arxiv.org/abs/1710.11248) | Adversarial inverse RL with disentangled reward |
| GCL       | [Finn et al. (2016)](https://arxiv.org/abs/1603.00448) | Guided cost learning with importance sampling |

### Baseline

| Algorithm | Paper | Method |
|-----------|-------|--------|
| BC        | — | Supervised: empirical P(a\|s) from demonstrations |

## Appendix: Pseudocode

### Notation

| Symbol | Definition |
|--------|-----------|
| $\mathcal{S}, \mathcal{A}$ | State space, action space |
| $\theta$ | Structural / reward parameters |
| $\beta$ | Discount factor |
| $\phi(s,a)$ | Feature vector |
| $u(s,a) = \phi(s,a)^\top\theta$ | Flow utility |
| $P(s' \mid s,a)$ | Transition probability |
| $V(s)$, $Q(s,a)$ | Value function, action-value function |
| $\pi(a \mid s)$ | Choice probability (policy) |
| $\mathcal{D}$ | Demonstrations: observed $(s_t, a_t)$ trajectories |
| $\bar\phi_\mathcal{D}$ | Empirical feature mean: $\frac{1}{T}\sum_t \phi(s_t, a_t)$ |
| $\mu(s)$ | State visitation frequency |

**Softmax Bellman operator** (used by all algorithms):

$$Q(s,a) = u(s,a) + \beta \sum_{s'} P(s' \mid s,a)\, V(s'), \qquad V(s) = \log\sum_a e^{Q(s,a)}, \qquad \pi(a \mid s) = e^{Q(s,a) - V(s)}$$

---

### Structural Estimators

**NFXP** — Nested Fixed Point (Rust, 1987)

```
1. Outer loop: optimize theta via MLE
2.   Inner loop: solve V = T(V; theta) to fixed point via contraction mapping
3.   Policy: pi(a|s) = exp(Q(s,a) - V(s))
4.   Log-likelihood: L(theta) = sum_{(s,a) in D} log pi(a|s)
5.   Update theta via BFGS
6. Return theta_MLE
```

**CCP** — Conditional Choice Probability / NPL (Hotz & Miller, 1993)

```
1. Estimate CCPs from data: pi_hat(a|s) = count(s,a) / count(s)
2. Social surplus: e(a|s) = euler_constant - log(pi_hat(a|s))
3. Choice-weighted transitions: P^pi(s'|s) = sum_a pi_hat(a|s) P(s'|s,a)
4. Invert: V_bar = (I - beta*P^pi)^{-1} sum_a pi_hat(a|s)[u(s,a;theta) + e(a|s)]
5. Q(s,a) = u(s,a;theta) + beta * P(.|s,a)' * V_bar
6. Optimize theta via MLE on log-softmax(Q)
7. NPL: re-estimate pi_hat from theta, repeat 2-6 until convergence
```

**SEES** — Sieve Estimator (Luo & Sang, 2024)

```
1. Choose sieve basis {b_1(s), ..., b_J(s)}  (e.g., Chebyshev polynomials)
2. Approximate: V(s; alpha) = sum_j alpha_j * b_j(s)
3. Q(s,a) = u(s,a;theta) + beta * P * V(.; alpha)
4. Penalized MLE: max_{theta, alpha}  L(theta, alpha) - lambda * ||V - T(V;theta)||^2
5. Optimize (theta, alpha) jointly via L-BFGS
```

**NNES** — Neural Network Estimator (Nguyen, 2025)

```
1. Parameterize: V_w(s) = NeuralNet_w(s)
2. Q(s,a) = u(s,a;theta) + beta * sum_s' P(s'|s,a) V_w(s')
3. Penalized MLE: max_{theta, w}  L(theta, w) - lambda * ||V_w - T(V_w;theta)||^2
4. Optimize (theta, w) jointly via Adam
```

---

### Entropy-Based IRL

**MCE IRL** — Maximum Causal Entropy (Ziebart, 2010)

```
1. Initialize theta
2. Repeat:
   a. Backward: soft value iteration V, Q from r(s,a) = phi(s,a)'*theta
   b. Policy: pi(a|s) = exp(Q(s,a) - V(s))
   c. Forward: state visitation mu(s) via propagation under pi
   d. Expected features: phi_bar = sum_s mu(s) sum_a pi(a|s) phi(s,a)
   e. Gradient: dL/dtheta = phi_D - phi_bar
   f. Update theta via Adam
3. Return theta
```

**MaxEnt IRL** — Maximum Entropy (Ziebart et al., 2008)

```
1. Initialize theta
2. Repeat:
   a. Reward: r(s,a) = phi(s,a)' * theta
   b. Soft value iteration -> pi(a|s)
   c. State visitation mu(s) via forward propagation under pi
   d. Gradient: dL/dtheta = phi_D - sum_s mu(s) sum_a pi(a|s) phi(s,a)
   e. Update theta
3. Return theta (reward weights)
```

**Deep MaxEnt** — Deep Maximum Entropy (Wulfmeier et al., 2016)

```
1. Initialize neural network f_w
2. Repeat:
   a. Reward: r(s,a) = f_w(phi(s,a))
   b. Soft value iteration -> pi(a|s)
   c. State visitation: mu_D (empirical), mu_pi (under pi)
   d. Reward gradient: dL/dr = mu_D - mu_pi
   e. Backprop: dL/dw = (dL/dr) * (dr/dw)
   f. Update w via Adam
3. Return f_w (learned reward network)
```

**BIRL** — Bayesian IRL (Ramachandran & Amir, 2007)

```
1. Initialize reward R, prior P(R), step size delta
2. For m = 1..M (MCMC):
   a. Propose R' by perturbing one component of R by +/- delta
   b. Solve MDP under R' -> Q*(s,a; R')
   c. Likelihood: P(D|R') = prod_{(s,a)} exp(alpha*Q*(s,a;R')) / Z(s)
   d. Accept R' with prob min(1, P(D|R')P(R') / P(D|R)P(R))
3. Return posterior mean: R_hat = (1/M) sum_m R^(m)
```

---

### Margin-Based IRL

**Max Margin** — Structured Max-Margin (Ratliff et al., 2006)

```
1. For each (s, a*) in D, for each alternative a != a*:
     Constraint: theta'*phi(s,a*) >= theta'*phi(s,a) + loss(a,a*) - xi
2. QP: min ||theta||^2 + C * sum(xi)  s.t. margin constraints, xi >= 0
3. Return theta
```

**Max Margin IRL** — Apprenticeship Learning (Abbeel & Ng, 2004)

```
1. Expert feature expectations: mu_E = E_D[sum_t beta^t phi(s_t,a_t)]
2. Initialize random policy pi_0, compute mu_0
3. Repeat until ||mu_E - closest||_2 < epsilon:
   a. theta = mu_E - argmin_{mu in conv(mu_0,...,mu_i)} ||mu_E - mu||
   b. Solve MDP under theta -> pi_{i+1}
   c. mu_{i+1} = E_{pi_{i+1}}[sum_t beta^t phi(s_t,a_t)]
4. Return theta (reward direction)
```

---

### Distribution Matching

**f-IRL** — f-Divergence IRL (Ni et al., 2022)

```
1. Initialize reward network r_theta(s), discriminator D_w(s)
2. Repeat:
   a. Train D_w to classify expert vs policy states
   b. Density ratio: rho_E(s)/rho_theta(s) = D_w(s) / (1 - D_w(s))
   c. h_f(ratio) for chosen f-divergence (KL, chi^2, TV)
   d. Gradient: dL/dtheta = (1/T) cov(sum_t h_f(ratio_t), sum_t dr_theta/dtheta)
   e. Update theta, recompute policy via soft value iteration
3. Return r_theta
```

---

### Neural Estimators

**TD-CCP** — TD-Learning + CCP (Adusumilli & Eckardt, 2022)

```
1. Train V_w(s) = NeuralNet_w(s) via temporal difference on D:
     L_TD = sum_{(s,a,s')} (V_w(s) - [r(s,a) + beta*V_w(s')])^2
2. Q(s,a) = phi(s,a)'*theta + beta * sum_s' P(s'|s,a) V_w(s')
3. MLE: theta = argmax sum_{(s,a) in D} log softmax(Q(s,.))_a
```

**GLADIUS** — Dual Network IRL (Kang et al., 2025)

```
1. Initialize Q-network Q_w(s,a), EV-network EV_psi(s,a)
2. Joint loss:
     L = -sum_{(s,a)} log softmax(Q_w(s,.))_a                        (NLL)
       + lambda * sum_{(s,a,s')} ||EV_psi(s,a) - beta*V(s')||^2      (Bellman)
   where V(s) = log sum_a exp(Q_w(s,a))
3. Train via mini-batch SGD
4. Rewards: r(s,a) = Q_w(s,a) - beta * EV_psi(s,a)
5. Structural params: theta = (Phi'Phi)^{-1} Phi' r
```

---

### Adversarial Methods

**GAIL** — Generative Adversarial Imitation (Ho & Ermon, 2016)

```
1. Initialize policy pi_w, discriminator D_psi
2. Repeat:
   a. Sample trajectories from pi_w
   b. D_psi: max E_D[log D_psi(s,a)] + E_{pi_w}[log(1 - D_psi(s,a))]
   c. Reward: r(s,a) = -log(1 - D_psi(s,a))
   d. Update pi_w via REINFORCE with reward r
3. Return pi_w (policy only — no transferable reward)
```

**AIRL** — Adversarial Inverse RL (Fu et al., 2018)

```
1. Initialize policy pi_w, reward g_psi(s,a), shaping h_xi(s)
2. Structured discriminator:
     f(s,a,s') = g_psi(s,a) + beta*h_xi(s') - h_xi(s)
     D(s,a,s') = sigmoid(f(s,a,s') - log pi_w(a|s))
3. Repeat:
   a. Update (psi, xi): max E_D[log D] + E_{pi_w}[log(1-D)]
   b. Update pi_w via REINFORCE with reward f
4. Return g_psi (disentangled reward function)
```

**GCL** — Guided Cost Learning (Finn et al., 2016)

```
1. Initialize cost network c_psi(tau), policy pi_w
2. Repeat:
   a. Sample demo tau_D, policy trajectories tau_pi
   b. Partition function via importance sampling:
        Z ≈ (1/M) sum_j exp(-c_psi(tau_j)) / q(tau_j)
   c. Update cost: min_psi E_D[c_psi(tau)] + log Z
   d. Update pi_w to minimize c_psi via REINFORCE
3. Return c_psi (negative reward)
```

---

### Baseline

**BC** — Behavioral Cloning

```
1. pi(a|s) = count(s, a in D) / count(s in D)
2. Return pi (frequency estimate — no reward recovered)
```

## License

MIT
