# Bayesian Inverse Reinforcement Learning
## Ramachandran & Amir (2007), IJCAI-07

PDF: `ramachandran_amir_2007.pdf` (from https://www.ijcai.org/Proceedings/07/Papers/416.pdf)

---

## 1. Setup

MDP M = (S, A, T, gamma, R) where:
- S: finite set of N states
- A = {a_1, ..., a_k}: set of k actions
- T: S x A x S -> [0,1] transition probabilities
- gamma in [0,1): discount factor
- R: S -> R, reward function (state-only), bounded by R_max

The expert X operates in MDP M = (S, A, T, gamma) with unknown reward R.
Observations: O_X = {(s_1, a_1), (s_2, a_2), ..., (s_k, a_k)}

Key assumptions about the expert:
1. X maximizes total accumulated reward according to R (e.g., epsilon-greedy)
2. X executes a stationary policy (invariant w.r.t. time)

---

## 2. Likelihood Function

The likelihood of a single state-action pair, given reward R:

```
Pr_X((s_i, a_i) | R) = (1/Z_i) * exp(alpha_X * Q*(s_i, a_i, R))
```

where:
- Q*(s, a, R) is the optimal Q-function for reward R
- alpha_X is a confidence parameter representing how reliably X chooses good actions
  (higher alpha = more confident the expert picks optimal actions)
- Z_i is the normalizing constant: Z_i = sum_a exp(alpha_X * Q*(s, a, R))

This is a **Boltzmann (softmax) rational** model of the expert's behavior.

Because the expert's policy is stationary, the full likelihood factorizes:

```
Pr_X(O_X | R) = prod_i Pr_X((s_i, a_i) | R)
              = (1/Z) * exp(alpha_X * E(O_X, R))
```

where the "energy" function is:

```
E(O_X, R) = sum_i Q*(s_i, a_i, R)
```

This is a **Boltzmann distribution** with energy E(O_X, R) and temperature 1/alpha_X.

---

## 3. Prior Distributions over Rewards

Rewards are assumed i.i.d. across states. Several priors are considered:

### 3a. Uniform Prior
```
P_R(R) = 1  for all R in R^n  (improper)
```
Or bounded: R(s) in [-R_max, R_max] for each s in S.

### 3b. Gaussian Prior
```
P_Gaussian(R(s) = r) = (1/sqrt(2*pi*sigma)) * exp(-r^2 / (2*sigma^2)),  for all s in S
```
Encourages parsimonious (small) rewards.

### 3c. Laplacian Prior
```
P_Laplace(R(s) = r) = (1/(2*sigma)) * exp(-|r|/(2*sigma)),  for all s in S
```
Promotes sparsity. **Key result (Theorem 2):** The original IRL algorithm of Ng & Russell (2000) is equivalent to the MAP estimator for BIRL with a Laplacian prior.

### 3d. Beta Prior (for planning problems)
```
P_Beta(R(s) = r) = 1 / ((r/R_max)^(1/2) * (1 - r/R_max)^(1/2)),  for all s in S
```
Has modes at high and low ends of reward space. Good for goal-based MDPs where most states have low reward but a few goal states have high reward.

### 3e. Ising Prior (for structured state spaces)
```
P_R(R) = (1/Z) * exp(-J * sum_{(s',s) in N} R(s)*R(s') - H * sum_s R(s))
```
where N is the set of neighboring state pairs, J is the coupling parameter, and H is the magnetization parameter. Useful when neighboring states should have correlated rewards.

---

## 4. Posterior Distribution

By Bayes' theorem:

```
Pr_X(R | O_X) = Pr_X(O_X | R) * P_R(R) / Pr(O_X)
              = (1/Z') * exp(alpha_X * E(O_X, R)) * P_R(R)
```

where Z' is the normalizing constant (intractable to compute directly).

---

## 5. Point Estimates from the Posterior

### 5a. Posterior Mean (for reward learning)
- Minimizes the expected **squared error loss**: L_SE(R, R_hat) = ||R - R_hat||_2
- The posterior mean R_hat = E[R | O_X] is the optimal estimator under squared loss (Berger, 1993).
- Computed via sample mean from MCMC samples.

### 5b. Posterior Median (for reward learning)
- Minimizes the expected **linear loss**: L_linear(R, R_hat) = ||R - R_hat||_1

### 5c. MAP Estimate
- Maximum a posteriori: argmax_R Pr(R | O_X)
- Less representative when posterior is multimodal (common in IRL).

### 5d. Optimal Policy for Apprenticeship Learning (Theorem 3)
The policy loss L^p_policy(R, pi) = ||V*(R) - V^pi(R)||_p is minimized for all p
by pi*_M, the optimal policy for the **mean reward function** E_P[R].
This means: compute posterior mean R_hat, then solve the MDP with R_hat to get the best policy.

---

## 6. MCMC Sampling Algorithm: PolicyWalk

The paper uses a modified MCMC algorithm called **PolicyWalk** (a variant of GridWalk
from Vempala 2005) to sample from the posterior.

### Algorithm: PolicyWalk(Distribution P, MDP M, Step Size delta)

```
1. Pick a random reward vector R in R^{|S|}/delta
   (i.e., on a grid of length delta in |S|-dimensional space)

2. pi := PolicyIteration(M, R)
   (compute optimal policy for initial R)

3. Repeat:
   (a) Pick a reward vector R_tilde uniformly at random from the
       neighbours of R in R^{|S|}/delta
       (i.e., change one component of R by +delta or -delta)

   (b) Compute Q^pi(s, a, R_tilde) for all (s, a) in S x A
       (Q-values under current policy pi for the NEW reward R_tilde)

   (c) If there exists (s, a) in (S, A) such that
       Q^pi(s, pi(s), R_tilde) < Q^pi(s, a, R_tilde):
       [The current policy pi is no longer optimal for R_tilde]

       i.   pi_tilde := PolicyIteration(M, R_tilde, pi)
            (recompute optimal policy starting from old policy pi)
       ii.  Set R := R_tilde and pi := pi_tilde
            with probability min{1, P(R_tilde, pi_tilde) / P(R, pi)}

       Else:
       [Policy pi is still optimal for R_tilde]

       i.   Set R := R_tilde
            with probability min{1, P(R_tilde, pi) / P(R, pi)}

4. Return R
```

### Key Efficiency Insight
- When pi is known, Q^pi(s, a, R) can be computed as a LINEAR function of R:
  V^pi(R) = (I - gamma * T^pi)^{-1} * R   (Equation 4)
  So step 3b is efficient.
- A change in optimal policy is detected cheaply in step 3c.
- When the policy DOES change, only a few steps of policy iteration are needed
  (starting from the old policy).
- The Metropolis-Hastings acceptance ratio P(R_tilde)/P(R) only requires the
  RATIO of posterior densities, so the intractable normalizing constant Z' cancels.

### Acceptance Probability (Metropolis-Hastings)
The acceptance probability for moving from R to R_tilde is:

```
min{1, P(R_tilde | O_X) / P(R | O_X)}
= min{1, exp(alpha_X * (E(O_X, R_tilde) - E(O_X, R))) * P_R(R_tilde) / P_R(R)}
```

For uniform prior, this simplifies to:
```
min{1, exp(alpha_X * (sum_i Q*(s_i, a_i, R_tilde) - sum_i Q*(s_i, a_i, R)))}
```

---

## 7. Convergence Guarantee (Theorem 4)

For uniform prior with R_max = O(1/N):
- The Markov chain induced by PolicyWalk on the posterior mixes rapidly
- Converges to within epsilon of the true posterior in **O(N^2 * log(1/epsilon))** steps
- This follows from the posterior being a pseudo-log-concave function (Lemma 1,
  from Applegate and Kannan 1993)

The proof shows:
- f(R) = alpha_X * E(O_X, R) satisfies the Lipschitz condition with alpha = O(1)
- f satisfies approximate log-concavity with beta = 2*alpha_X*N*R_max/(1-gamma)
- For R_max = O(1/N), beta = O(1), giving polynomial mixing time

Note: R_max = O(1/N) is not restrictive because rewards can be rescaled by a constant
factor k without changing the optimal policy (all value/Q functions scale by k too).

---

## 8. Computing the Posterior Mean in Practice

1. Run PolicyWalk for a burn-in period (discard initial samples).
2. Collect M samples R^(1), R^(2), ..., R^(M) from the chain.
3. Compute the sample mean:
   ```
   R_hat = (1/M) * sum_{j=1}^{M} R^(j)
   ```
4. For apprenticeship learning: solve MDP(S, A, T, gamma, R_hat) to get optimal policy pi*.

---

## 9. Key Takeaways for Implementation

- **Likelihood is Boltzmann/softmax** over Q-values, controlled by alpha_X
- **Prior choice matters**: uniform for agnostic, Gaussian/Laplacian for sparse rewards, Beta for goal-based
- **PolicyWalk is the workhorse**: grid-based MCMC that efficiently tracks policy changes
- **Posterior mean** is the optimal point estimate for squared-error reward learning
- **For apprenticeship learning**: just compute the optimal policy for the mean reward
- **Original IRL (Ng & Russell 2000) = MAP with Laplacian prior** (Theorem 2)
- **Polynomial mixing time** guaranteed for uniform prior (Theorem 4)
