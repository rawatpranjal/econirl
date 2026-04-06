# AIRL-Het

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Lee, Sudhir, and Wang (2026) | Linear | No (adversarial) | No | No |

## What this estimator does

Standard AIRL recovers a transferable reward by decomposing the discriminator into reward and shaping components, but the disentanglement theorem requires state-only rewards and rules out absorbing states. Dynamic discrete choice models have action-dependent payoffs and exit options, violating both assumptions. Lee, Sudhir, and Wang (2026) extend AIRL with anchor constraints that resolve these limitations.

Anchor identification works by fixing the reward of one action (exit) to zero everywhere and the value of one state (absorbing) to zero. These two constraints pin down the reward uniquely at convergence. Without anchors, IRL rewards are identified only up to $|\mathcal{S}|$ free variables corresponding to the value function. With anchors, the decomposition $g_\theta(s,a) = r^*(s,a)$ and $h_\phi(s) = V^*(s)$ holds exactly. The estimator also supports latent heterogeneity via an EM algorithm that discovers consumer segments, each with its own reward and policy.

## How it works

The algorithm follows the same adversarial structure as AIRL. The discriminator logit is

$$
f_{\theta,\phi}(s,a,s') = g_\theta(s,a) + \beta h_\phi(s') - h_\phi(s),
$$

with the anchor constraints $g_\theta(s, a_{\text{exit}}) = 0$ for all states and $h_\phi(s_{\text{abs}}) = 0$ enforced after each discriminator update. The policy generator can be tabular (soft value iteration with conservative mixing) or neural (PPO). At convergence the reward network $g_\theta$ captures the true action-dependent reward. Standard errors are not available analytically due to the adversarial training loop.

## When to use it

AAIRL is designed for structural estimation on platforms with serialized content or subscription decisions, where exit actions and absorbing states are natural features of the environment. It is the only estimator in the package that handles action-dependent rewards with transfer guarantees. On tabular problems with known transitions, NFXP remains more accurate and faster. A known issue is adversarial reward scale inflation, where the discriminator loss is invariant to uniform scaling of the reward and the optimizer drifts upward. The relative ordering of feature weights is preserved but absolute values may be inflated.

## References

- Lee, S., Sudhir, K., and Wang, Y. (2026). Modeling Serialized Content Consumption: Adversarial IRL for DDC. Working paper.
- Fu, J., Luo, K., and Levine, S. (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. *ICLR 2018*.

The full derivation, algorithm, and simulation results are in the [AAIRL primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/aairl.pdf).
