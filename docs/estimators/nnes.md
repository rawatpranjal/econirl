# NNES

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Nguyen (2025) | Linear | Yes | Valid (orthogonality) | Yes (neural V) |

## Background

Basis functions work when the value function is smooth, but can struggle with sharp nonlinearities. Nguyen (2025) replaced the basis with a neural network, which can approximate any shape. The challenge is that neural networks make standard error computation very hard. The breakthrough was proving that the DDC likelihood has a special structure: errors in the neural value function have only a second-order effect on the parameter estimates (Neyman orthogonality). This means the standard errors from the likelihood Hessian are valid even though the neural network is imperfect.

## Key Equations

$$
(\hat\theta, \hat{w}) = \arg\max_{\theta, w} \; \ell^p(\theta) - \lambda \sum_{(s,a,s') \in \mathcal{D}} \big(V_w(s) - r_\theta(s,a) - \beta V_w(s')\big)^2.
$$

## Pseudocode

```
NNES(D, features, p, beta, sigma, hidden_dims, lambda):
  1. Initialize theta, neural network V_w
  2. Alternate:
     a. Train V_w to minimize Bellman residual (mini-batch SGD)
     b. Estimate theta by maximizing partial likelihood with V_w held fixed
  3. Standard errors from partial likelihood Hessian (valid by orthogonality)
  4. Return theta, SEs
```

## Strengths and Limitations

NNES is the only neural method with theoretically valid standard errors. Neyman orthogonality insulates the structural parameter estimates from neural network approximation noise, so the Hessian-based confidence intervals are reliable even when the value function approximation is imperfect.

The limitation is sensitivity to initialization. Achieving an exact zero Bellman residual in a deep learning context is rare, and poor starting values for $\theta$ or the network weights can lead to slow convergence or local optima. The alternating optimization between the neural network and the structural parameters also requires careful tuning of learning rates and the penalty weight $\lambda$.

NNES is the right choice for high-dimensional or continuous state spaces when you need publication-grade inference with valid standard errors.

## References

- Nguyen, H. (2025). Neural Network Estimation of Structural Models.
