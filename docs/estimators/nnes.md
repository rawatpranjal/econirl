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

## Diagnostics and Guarantees

Identification requires the same conditions as NFXP and CCP. The utility features must vary across actions, and the transition matrix must be known. The additional requirement is that the neural value function approximation must be flexible enough to represent the true value function. Underfitting in Phase 1 produces biased structural parameters, though the Neyman orthogonality property (described below) makes the estimates robust to small approximation errors.

The estimator alternates between two phases across a fixed number of outer iterations (default 3). Phase 1 trains the V-network via supervised regression on the NPL target computed from the current CCPs, running for 500 epochs of mini-batch SGD with a batch size of 512. Phase 2 maximizes the CCP pseudo-likelihood over structural parameters using L-BFGS-B with a gradient tolerance of 1e-6 and a maximum of 200 iterations. After Phase 2, the CCPs are updated from the estimated parameters for the next outer iteration. There is no explicit convergence criterion across outer iterations.

Standard errors from the pseudo-likelihood Hessian are valid by the Neyman orthogonality property (Nguyen 2025, Propositions 3 and 4). The zero Jacobian property of the NPL mapping ensures that first-order errors in the neural V approximation drop out of the Phase 2 score, so the Hessian-based confidence intervals are reliable even when the value function fit is imperfect. This guarantee applies only to the NPL-based variant (NNESEstimator). The legacy NFXP-based variant (NNESNFXPEstimator) does not have this orthogonality property, and its standard errors may be unreliable.

The main practical limitation is sensitivity to initialization. Poor starting values for theta or the network weights can lead to slow convergence or local optima. The estimator bootstraps initial parameters from a Hotz-Miller inversion when possible, and uses gradient clipping of 1.0 to stabilize V-network training. The alternating optimization between the neural network and the structural parameters requires careful tuning of learning rates and the number of training epochs.

The default configuration uses a V-network with 2 hidden layers of 32 units each, a learning rate of 1e-3 for Adam, 500 training epochs per outer iteration, and 3 outer iterations. The L-BFGS-B optimizer for Phase 2 uses a gradient tolerance of 1e-6 with a maximum of 200 iterations. These defaults are tuned for tabular problems with moderate state spaces.
