# f-IRL: Inverse Reinforcement Learning via State Marginal Matching

**Authors:** Tianwei Ni, Harshit Sikchi, Yufei Wang, Tejus Gupta, Lisa Lee, Benjamin Eysenbach
**Published:** CoRL 2020 (arXiv: 2011.04709)
**PDF:** ni_2022.pdf (in this directory)
**Source:** https://arxiv.org/abs/2011.04709

---

## 1. How f-Divergences Are Used to Match State Marginals

### Definition
For probability distributions P and Q, the f-divergence is:

    D_f(P||Q) := integral_Omega f(dP/dQ) dQ

Applied to state marginal matching between expert (rho_E) and policy (rho_theta):

    L_f(theta) = D_f(rho_E(s) || rho_theta(s)) = integral_S f(rho_E(s)/rho_theta(s)) rho_theta(s) ds

### Common f-Divergence Instantiations

| Divergence    | f(u)                                    | h_f(u) = f(u) - f'(u)*u |
|---------------|----------------------------------------|--------------------------|
| Forward KL    | u log u                                | -u                       |
| Reverse KL    | -log u                                 | 1 - log u                |
| Jensen-Shannon| u log u - (1+u) log((1+u)/2)          | -log(1+u)                |

Key property for Reverse KL: h_RKL(rho_E(s)/rho_theta(s)) = 1 - log rho_E(s) + log rho_theta(s),
so the normalizing factor of rho_E(s) cancels -- allows "unnormalized density specification."

---

## 2. The Reward Update Procedure

### Main Theorem (Theorem 4.1): Analytic Gradient of f-Divergence

    grad_theta L_f(theta) = (1/(alpha*T)) * cov_{tau ~ rho_theta(tau)} (
        sum_{t=1}^T h_f(rho_E(s_t)/rho_theta(s_t)),
        sum_{t=1}^T grad_theta r_theta(s_t)
    )

Where:
- alpha = entropy temperature parameter
- T = horizon length
- cov = covariance under agent's trajectory distribution rho_theta(tau)
- h_f = divergence-specific function (see table above)

### Complete Algorithm (Algorithm 1: f-IRL)

    Input: Expert state density rho_E(s) or observations s_E, f-divergence choice
    Output: Learned reward r_theta, Policy pi_theta

    Initialize r_theta and density model/discriminator

    For i = 1 to Iterations:

        1. pi_theta <- MaxEntRL(r_theta)
           Collect agent trajectories tau_theta

        2. IF given rho_E(s):
               Fit density model rho_hat_theta(s) to state samples from tau_theta
           ELSE IF given s_E:
               Fit discriminator D_omega via binary cross-entropy (Eq. 4)

        3. Compute gradient estimate:
           grad_hat_theta L_f(theta) using Theorem 4.1

        4. Update reward:
           theta <- theta - lambda * grad_hat_theta L_f(theta)

    Return r_theta, pi_theta

### Reward Parameterization
- Uses **state-only reward**: r_theta(s) parameterized as an MLP neural network
- Avoids reward ambiguity issues inherent to state-action-next-state rewards
- Enables "disentangled" reward recovery robust across different dynamics

---

## 3. Computing Expert vs Policy State Marginal Distributions

### MaxEnt RL Trajectory Distribution

    rho_theta(tau) = (1/Z) * p(tau) * exp(r_theta(tau)/alpha)

where:
    p(tau) = rho_0(s_0) * product_{t=0}^{T-1} p(s_{t+1}|s_t,a_t)
    r_theta(tau) = sum_{t=1}^T r_theta(s_t)
    Z = integral p(tau) exp(r_theta(tau)/alpha) d_tau

### State Marginal from Trajectories

    rho_theta(s) proportional_to integral p(tau) exp(r_theta(tau)/alpha) eta_tau(s) d_tau

where eta_tau(s) = sum_{t=1}^T 1(s_t = s) is the state visitation count.

### Gradient of State Marginal w.r.t. Reward

    d rho_theta(s)/d theta = (1/(alpha*Z)) integral p(tau) exp(r_theta(tau)/alpha) eta_tau(s) sum_{t=1}^T d r_theta(s_t)/d theta d_tau
                            - (T/alpha) rho_theta(s) integral rho_theta(s*) d r_theta(s*)/d theta ds*

### Density Ratio Estimation (from expert observations)

When expert trajectories are provided, fit a discriminator D_omega(s) via binary cross-entropy:

    max_omega E_{s~s_E}[log D_omega(s)] + E_{s~rho_theta(s)}[log(1-D_omega(s))]

At optimality:
    D_omega*(s) = rho_E(s) / (rho_E(s) + rho_theta(s))

Density ratio:
    rho_E(s)/rho_theta(s) approx D_omega(s) / (1-D_omega(s))

### From Direct Density Specification
Fit density model rho_hat_theta(s) to agent state samples and directly estimate ratio.

---

## 4. Practical Modification for High Dimensions

For high-dimensional states with distribution mismatch, use biased gradient:

    grad_tilde_theta L_f(theta) := (1/(alpha*T)) * cov_{tau ~ (rho_theta(tau) + rho_E(tau))/2} (
        sum_{t=1}^T h_f(rho_E(s_t)/rho_theta(s_t)),
        sum_{t=1}^T grad_theta r_theta(s_t)
    )

This becomes unbiased at convergence when agent and expert distributions match.

---

## 5. Downstream Application: Reward Shaping

Learned rewards serve as potential-based reward shaping:

    r(s,a,s') = r_task(s,a,s') + lambda * (gamma * r_prior(s') - r_prior(s))

where r_prior is recovered by f-IRL, preserving optimal policy invariance.
