# AIRL

| Category | Citation | Reward | Transitions | SEs | Scales | Transfer |
|----------|----------|--------|-------------|-----|--------|----------|
| Inverse | Fu, Luo, and Levine (2018) | Linear or Tabular | No (adversarial) | No | No | Yes |

## What this estimator does

Maximum entropy IRL recovers a reward function, but the recovered reward entangles the true reward with the value function through potential-based shaping. If the environment dynamics change, the shaping term evaluates differently and the learned reward stops working. Fu, Luo, and Levine (2018) fix this by giving the discriminator a special structure. They split the discriminator logit into a reward piece $g_\theta(s)$ and a shaping piece $\beta h_\phi(s') - h_\phi(s)$. The shaping piece absorbs everything that depends on the environment dynamics. What remains, $g_\theta$, is the true reward and it transfers to new settings.

The disentanglement theorem guarantees that if the reward is state-only, the MDP is decomposable (all states reachable), and adversarial training converges to the Nash equilibrium, then $g_\theta(s) = r^*(s) + c$ for a constant $c$. The decomposability condition rules out absorbing states, and the state-only restriction excludes action-dependent payoffs. Both limitations matter for DDC models, which is why Lee, Sudhir, and Wang (2026) extend AIRL with anchor constraints in AAIRL.

## How it works

The algorithm alternates between training the discriminator to distinguish expert from policy transitions and updating the policy to fool the discriminator. The discriminator has the structured form

$$
f_{\theta,\phi}(s,a,s') = g_\theta(s) + \beta h_\phi(s') - h_\phi(s), \quad D_{\theta,\phi} = \frac{\exp(f_{\theta,\phi})}{\exp(f_{\theta,\phi}) + \pi(a|s)}.
$$

The reward signal for the policy update is the log-odds of the discriminator. At convergence $D = 1/2$ everywhere and the shaping potential $h_\phi$ converges to the soft value function. AIRL does not produce analytical standard errors. Bootstrap is available but computationally expensive because each replicate requires a full adversarial training run.

## When to use it

AIRL is the right choice when the learned reward needs to work in a different environment than the one it was trained in. Transfer experiments on PointMaze and continuous control tasks show that AIRL transfers while GAIL and MaxEnt IRL do not. On tabular problems where transitions are known, structural methods like NFXP and MCE-IRL are faster, more stable, and produce analytical standard errors. AIRL's practical value is in continuous-state settings where tabular methods cannot operate.

## References

- Fu, J., Luo, K., and Levine, S. (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. *ICLR 2018*.
- Ng, A. Y., Harada, D., and Russell, S. (1999). Policy Invariance Under Reward Transformations. *ICML 1999*.

The full derivation, algorithm, and simulation results are in the [AIRL primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/airl.pdf).
