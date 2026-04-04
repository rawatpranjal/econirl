## TEMPORAL-DIFFERENCE ESTIMATION OF DYNAMIC DISCRETE CHOICE MODELS

## KARUN ADUSUMILLI † AND DITA ECKARDT *

Abstract. We study the use of Temporal-Difference learning for estimating the structural parameters in dynamic discrete choice models. Our algorithms are based on the conditional choice probability approach but use functional approximations to estimate various terms in the pseudo-log-likelihood function. We suggest two approaches: The first-linear semi-gradient-provides approximations to the recursive terms using basis functions. The second-Approximate Value Iteration-builds a sequence of approximations to the recursive terms by solving non-parametric estimation problems. Our approaches are fast and naturally allow for continuous and/or high-dimensional state spaces. Furthermore, they do not require specification of transition densities. In dynamic games, they avoid integrating over other players' actions, further heightening the computational advantage. Our proposals can be paired with popular existing methods such as pseudo-maximum-likelihood, and we propose locally robust corrections for the latter to achieve parametric rates of convergence. Monte Carlo simulations confirm the properties of our algorithms in practice.

## 1. Introduction

Recent years have seen a number of important developments in the field of Reinforcement Learning (RL) for computation of value functions. The goal of this paper is to study the use of a popular RL technique, Temporal-Difference (TD) learning, for estimation and inference in Dynamic Discrete Choice (DDC) models.

DDC models are frequently used to describe the inter-temporal choices of forward-looking individuals in a variety of contexts. In these models, agents maximize their expected future payoff through repeated choice amongst a set of discrete alternatives. Based on a revealed preference argument, structural estimation proceeds by using

Key words and phrases. Dynamic discrete choice models, Dynamic discrete games, Temporal-Difference learning, Reinforcement Learning.

This version : August 28, 2025.

† Department of Economics, University of Pennsylvania; akarun@sas.upenn.edu.

* Department of Economics, University of Warwick; dita.eckardt@warwick.ac.uk.

We would like to thank the editor and three anonymous referees for valuable comments that substantially improved the paper. Thanks also to Xiaohong Chen, Frank Diebold, Aviv Nevo, Whitney Newey and Frank Schorfheide for helpful discussions.

microdata on choices and outcomes to recover the underlying model parameters. 1 A key challenge in this literature is the complexity of estimation. Uncovering the structural parameters originally required an explicit solution to a dynamic programming problem in addition to the optimization of an estimation criterion. A key advance has been Hotz and Miller's (1993) Conditional Choice Probability (CCP) algorithm which avoids the repeated solution of the inter-temporal optimization problem by taking advantage of a mapping between value function differences and conditional choice probabilities.

Unfortunately, the standard CCP algorithm of Hotz and Miller (1993) is computationally infeasible when the underlying states are continuous and/or the state space is high-dimensional. Such state spaces are common in applications. One approach to tackle continuous state spaces is through state space discretization, e.g., Kalouptsidi (2014) and Almagro and Domínguez-Iino (2025) use aggregation and clustering methods to do this. However, it is not always clear how to perform such a discretization in practice, and moreover, it introduces bias into the parameter estimates. An alternative is to employ functional approximations for the value functions. For instance, Barwick and Pathak (2015) and Kalouptsidi (2018) use estimated transition densities and numerical/analytical integration to approximate the value functions using linear sieves and LASSO, respectively. However, the theoretical properties of these methods when using machine learning methods (such as LASSO) are as yet unknown, and they still require estimation of transition densities, which is not straightforward, along with numerical integration, which can be computationally expensive. 2

The aim of this paper is to develop tractable algorithms for CCP estimation when the state variables are continuous and/or the state space is large. Such algorithms should possess three properties: First, they should be fast to compute even under high-dimensional state spaces. Second, they should avoid state space discretization, and instead rely on functional approximation of value functions. Third, they should avoid estimation of transition densities which are difficult to parameterize and estimate under continuous states. If the DDC model in question satisfies either finite dependence or a terminal state property there already exist algorithms possessing these properties, see, e.g., Ackerberg et al. (2014) and Chernozhukov et al. (2022). Our interest here is in developing general purpose algorithms that do not require these assumptions.

1 See Aguirregabiria and Mira (2010) for a survey of the literature on the estimation of DDC models. 2 Yet another alternative is to use forward Monte Carlo simulations (Bajari et al., 2007, Hotz et al., 1994), but this again becomes very involved as the number of continuous state variables or players increases. The use of a finite number of Monte Carlo simulations also adds bias to the estimates.

We suggest two methods, based on TD learning, that satisfy all the above properties. The methods involve two different techniques for estimating recursive terms (which are akin to value functions) that arise in CCP estimation. The first approach, the linear semi-gradient method, provides functional approximations to the recursive terms using basis functions. This simply involves inverting a matrix whose dimension is the number of basis functions, so the computational cost is generally trivial. Furthermore, it only requires the observed sequences of current and future state-action pairs as input and estimation of transition densities is not needed. The second approach, Approximate Value Iteration (AVI), builds a sequence of approximations to the value terms by solving a non-parametric estimation problem in each step. Almost any machine learning (ML) method for prediction can be used for the approximation, including (but not limited to) LASSO, Random Forests and Neural Networks. To our knowledge, the AVI method is the first estimator for general DDC models that can be applied with any ML method that achieves suitable rates of convergence. Hence, it naturally allows for very highdimensional state spaces. Again, no estimation of transition densities is required. We derive the non-parametric rates of convergence for estimation of the value terms under both methods. Using the estimates of these functions, estimation of the structural parameters can proceed with standard methods such as pseudo-maximum-likelihood estimation (PMLE, Aguirregabiria and Mira (2002)) or minimum distance estimation.

The focus of this paper is on the estimation of structural parameters. To this end, our procedures avoid modeling state transitions. Performing counterfactual analysis may still require estimating the transition density, but we argue that our techniques remain useful, even for this purpose, for two reasons: First, counterfactuals often involve transition densities which are different from the ones that enter the estimation of the structural parameters, see e.g., Kalouptsidi (2018). Our methods thus avoid estimation of the original transition densities. Second, with continuous states, decoupling the estimation of structural parameters and transition densities may be beneficial for robustness and efficiency. For instance, it is common to employ AR (e.g., Aguirregabiria and Mira, 2007; Kalouptsidi, 2014) or VAR (Barwick and Pathak, 2015) specifications for transition densities. However, these specifications involve a number of choices (e.g., dimension of VARs, distribution of error terms etc.), which the structural parameter estimates may not be robust to. Importantly, even when non-parametric estimates of transition densities are available, plugging them into the second-stage PMLE criterion would seriously degrade

the rate of convergence of structural parameters. One would need to adjust the PMLE to account for the non-parametric first stage, but the form of this adjustment is not known. By contrast, our proposals use non-parametric estimates of value functions, and as described below, we derive the necessary adjustments to account for this. To perform counterfactual analysis, we suggest combining our estimates of the structural parameters - which do not rely on non-parametric estimates of transition densities and are robust to mis-specification - with non-parametric estimates of the transition densities.

The previous discussion highlights that in continuous state spaces, estimation of structural parameters is inherently a problem of semi-parametric estimation. In fact, even under discrete states, estimation of transition densities affects the variance of the structural parameter estimates, see Aguirregabiria and Mira (2002). If the state variables are continuous, existing two-step CCP methods such as the PMLE are no longer √ n -consistent. We therefore derive a locally robust estimator by adding a correction term to the PMLE criterion function that accounts for the non-parametric estimation of value function terms using either of our TD methods. This construction is novel and does not directly follow from existing results, e.g., in Chernozhukov et al. (2022). 3 The resulting estimator converges at parametric rates under continuous states and unrestricted transition densities.

Our TD estimators are thus consistent, converge at parametric rates, and provide a feasible estimation method when the states are continuous and/or the state space is large. The latter is particularly important for the estimation of dynamic discrete games. Existing methods for the estimation of dynamic games (Bajari et al., 2007; Aguirregabiria and Mira, 2007; Pesendorfer and Schmidt-Dengler, 2008) require integrating out other players' actions, which can get quite cumbersome with many players, or under continuous states. By contrast, our procedure works directly with the joint empirical distribution of the states and their sample successors. Thus the 'integrating out' is done implicitly within the sample expectations.

3 Independently of our work and around the release of our initial draft, Chernozhukov et al. (2019) derived orthogonal moment conditions for weighted averages of value functions of the form θ 0 = E [ w ( x ) V ( x )], where V ( · ) is an estimated value function and w ( · ) is a known weight function. Their approach to deriving the orthogonal moment is different from ours as it is based on the methods by Ichimura and Newey (2022), but similarly to us, it results in a debiasing function involving backward projections, where the current state is regressed on a future one. Subsequent versions of Chernozhukov et al. (2019), released after the first revision of our paper, extend their framework to estimate parameters of the form θ = E [ m ( x, V )], where m ( x, · ) is a nonlinear functional of V . In this generalized setting, their construction of the locally robust correction term involves iterating the local-Riesz representer α ( · ) for E [ m ( x, V )] backwards in what the authors call a 'dynamic dual representation'. This approach closely parallels the methodology of our paper.

Finally, we also incorporate permanent unobserved heterogeneity into our methods by combining the TD estimation with an Expectation-Maximization (EM) algorithm.

A range of Monte Carlo studies confirm the workings of our algorithms. First, we present simulations based on the dynamic firm entry problem described in Aguirregabiria and Magesan (2018). The model has seven structural parameters and five continuous state variables. Existing methods often struggle at this dimensionality; certainly, state space discretization would not work too well. We provide simulations for the linear semigradient and the AVI method with Random Forests, with and without locally robust corrections. Our estimators perform very well in this setting and they outperform CCP estimators that employ discretization, leading to a 10-fold reduction in average mean squared error across the structural parameters. They also perform similar to or outperform alternative methods such as the 2-step Euler-Equation (EE) approach of Aguirregabiria and Magesan (2018), even though the latter only applies to a more restricted class of models. Our linear semi-gradient method is even three times faster to compute.

Second, we test our algorithms for dynamic discrete games based on a firm entry game similar to that outlined in Aguirregabiria and Mira (2007). We use the linear semigradient method here and, as before, our estimates are closely centered around the true parameters. Since this approach requires the selection of a set of basis functions for the functional approximations, we present results for different sets (a second, third and fourth order polynomial) in this model. Our findings suggest that the choice of basis functions has only a small effect on the performance of the estimator. Moreover, a simple cross-validation procedure may be used to select the preferred set of functions.

1.1. Related literature. Rust (1987) is the seminal work in the literature of DDC models. Motivated by computational considerations, Hotz and Miller (1993) propose the CCP algorithm. The CCP idea has subsequently been refined by Hotz et al. (1994) who suggest a simulation-based method, and Aguirregabiria and Mira (2002) who develop a pseudolikelihood estimator. Arcidiacono and Miller (2011) introduce and exploit the property of finite dependence to speed up CCP estimation. Despite these advances, the estimation of DDC models remains constrained by its computational complexity, particularly in the large class of models where finite dependence does not hold. Estimation of dynamic discrete games is particularly affected by these issues as the strategic interaction of agents means that the state space increases exponentially with the number of players. It is also uncommon for finite dependence to hold in dynamic games.

The standard CCP algorithm is a two-step method, and is known to suffer from severe bias in finite samples. Aguirregabiria and Mira (2002; 2007) address this issue by presenting a recursive CCP estimator, the nested pseudo-likelihood (NPL), that is equivalent to the nested-fixed-point estimator (NFXP, Rust, 1987). Both are in turn equivalent to partial-MLE, which employs a plug-in estimate of the transition density in the MLE criterion, but they are not fully efficient (they are not equivalent to full-MLE). In fact, with continuous states, estimation of the transition density introduces bias that is the dominant term in determining the rate of convergence. This motivates the construction of our locally robust estimator which gets rid of this bias and restores parametric rates. In fact, our proposal can be more efficient than these methods even with a parametric model for the transition density, see Section 4.2.2.

Ackerberg et al. (2014) and Chernozhukov et al. (2022) consider semi-parametric estimation using ML methods when either finite dependence or a 'terminal action' property holds (Hotz and Miller, 1993). 4 Chernozhukov et al. (2022) also derive locally robust corrections for this setting. Under finite dependence the PMLE criterion can be written as a function of choice probabilities only (transition densities are not required); the authors employ non-parametric estimates for choice probabilities and correct for this estimation in the second stage. Computation and estimation is thus relatively simpler under finite dependence. By contrast, our methods are applicable to the more general and difficult setting where finite dependence may not apply. Nevertheless, the computational speed of our linear semi-gradient procedure is comparable to methods that exploit finite dependence. For dynamic games, Semenova (2018) allows for high-dimensional state spaces, but the approach it is based on, due to Bajari et al. (2007), is not efficient, e.g., it may only partially identify parameters even if the model is fully identified. On the other hand, it allows for continuous actions, unlike our method.

In making use of TD learning, our methods relate to the literature on RL, particularly batch RL. Batch RL describes learning about how to map states into actions to maximize an expected payoff, using a fixed set of data (a so-called batch); see Lange et al. (2012) for a survey. 5 A key step in RL, including batch RL, is the estimation of value functions. TD learning methods, first formulated by Sutton (1988), are the most commonly used set of algorithms for this purpose. We study non-parametric estimation

4 In a different application of ML methods in this context, Norets (2012) suggests combining Neural Networks with a Bayesian MCMC approach.

5 See Sutton and Barto (2018) for a detailed treatment of RL in genreal.

of value functions using two TD methods: semi-gradients and AVI. Our analysis builds on the techniques developed by Tsitsiklis and Van Roy (1997) for linear semi-gradients, and Munos and Szepesvári (2008) for AVI. While Tsitsiklis and Van Roy (1997) focus on online learning (i.e., where collection of data and estimation of value functions is conducted simultaneously), we translate their methods to batch learning. 6 With regards to Munos and Szepesvári (2008), we differ in employing assumptions that are more common to econometrics and our characterization of the rates is also different (compare Theorem 2 in their paper with our Theorem 3).

TD methods are distinct from other value function approximation methods developed in economics, e.g., parametric policy iteration (Benítez-Silva et al., 2000), simulation and interpolation (Keane and Wolpin 1994), and sieve value function iteration (Arcidiacono et al., 2013). The last of these is similar in spirit to AVI with linear functional approximations. However, our semi-gradient method provides a linear approximation in a single step without any need for iterations, and we analyze AVI under generic machine learning methods. Our approximation results, and the technical arguments leading to them, are thus different from Arcidiacono et al. (2013); in fact, their setting is different too as the authors focus on estimating the 'optimal' value function, while the recursive terms in our setting are more akin to a value function under a fixed policy.

## 2. Setup

We start with an infinite horizon single-agent DDC model in discrete time, where observations consist of i = 1 , . . . , n agents. We assume that the agents are homogeneous, relegating extensions to unobserved heterogeneity to Online Appendix C. In each period, an agent chooses among A mutually exclusive actions, denoted by a . Choosing a when the current state is x gives the agent an instantaneous utility of z ( a, x ) ⊺ θ ∗ + e , where z ( a, x ) is a known vector-valued function of a, x and e is an idiosyncratic error term. We denote the realized state of an agent i at time t by x it , and her corresponding action and error by a it and e it . We assume that e it is an iid draw from some known distribution g e ( · ). Let ( a ′ , x ′ ) denote the one-period ahead random variables following the actions and states ( a, x ), where x ′ ∼ K ( ·| a, x ), with K ( ·| a, x ) denoting the transition density given a, x (more precisely, it is the Markov kernel). We do not make any parametric assumptions about K ( ·| a, x ). The utility from future periods is discounted by β .

6 See also Chen and Qi (2022) for related results on Q -learning under series approximations.

Agent i chooses actions a i = ( a i 1 , a i 2 , . . . ) to sequentially maximize the discounted sum of payoffs

<!-- formula-not-decoded -->

The econometrician observes a panel consisting of state-action pairs for all individuals, ( x i , a i ) = { ( x i 1 , a i 1 ) , . . . , ( x iT , a iT ) } , for T periods (note, however, that the agent maximizes an infinite horizon objective, not a fixed T one). Typically T /lessmuch n in applications, so we work within an asymptotic regime where n →∞ but T is fixed. Using this data, the econometrician aims to recover the structural parameters θ ∗ .

In this paper, we study the CCP approach for estimating θ ∗ (Hotz and Miller, 1993). CCP methods are based on the conditional choice probabilities of choosing action a given state x . We denote these by P t ( a | x ) for a given period t but henceforth drop the subscript t with the idea that it can be made a part of the state variable x , if needed (we should also add that some of our theoretical results are based on assuming stationarity, i.e., P t ( a | x ) is independent of t ). Denote e ( a, x ) as expected value of the idiosyncratic error term e given that action a was chosen. Hotz and Miller (1993) show that if the distribution of e follows a Generalized Extreme Value (GEV) distribution, it is possible to express e ( a, x ) as a function of the choice probabilities P ( a | x ), i.e., e ( a, x ) = G ( P ( a | x )). We assume that e follows a Type I Extreme Value distribution, which is perhaps the most common choice in applications. In this case e ( a, x ) = γ -ln P ( a | x ), where γ is the Euler constant.

Using the standard CCP approach, under the given distributional assumptions, the parameters are obtained as the maximizers of the pseudo-log-likelihood function

<!-- formula-not-decoded -->

where for any ( a, x ), h ( · ) and g ( · ) solve the following recursive expressions: 7

<!-- formula-not-decoded -->

Here, E [ ·| a, x ] denotes the expectation over the distribution of ( a ′ , x ′ ) conditional on ( a, x ); it is a function of K ( ·| a, x ) , P ( ·| x ). Both h ( a, x ) and g ( a, x ) have a 'value-function' form, which turns out to be useful for our approach.

7 Note that h ( a it , x it ) = E [∑ ∞ τ = t β ( τ -t ) z ( a iτ , x iτ ) ∣ ∣ a it , x it ] , i.e., we can interpret h ( a it , x it ) ⊺ θ as the expected discounted utility (excluding the error term) given the current state a it , x it . A similar interpretation holds for g ( · ). See Aguirregabiria and Mira (2010) for a further description.

Clearly, h ( · ) and g ( · ) are functions of K ( ·|· ) and P ( ·|· ). Since the latter are unknown, current literature generally proceeds by first estimating these as ( ˆ K, ˆ P ). Typically, ˆ K is obtained by MLE based on a parametric form of K ( x ′ | a, x ; θ f ), while ˆ P is estimated nonparametrically using either a blocking scheme or kernel regression. Then, given ( ˆ K, ˆ P ), h ( · ) and g ( · ) are estimated by solving the recursive equations (2.2). In the next section, we propose an alternative algorithm for maximizing Q ( θ ) that directly estimates h ( · ) and g ( · ) in a single step without requiring any knowledge about or estimation of K ( ·|· ).

Notation. We assume that the distribution of ( a it , x it , a it +1 , x it +1 ) is time stationary. This greatly simplifies our notation. It is not necessary for our results on the approximation properties of our TD methods, see Appendix A, but we do require it for deriving a locally robust estimator. Since the transition density and choice probabilities are time independent (the latter due to Blackwell's theorem), the stationarity assumption is equivalent to supposing that { a it , x it , a it +1 , x it +1 } i are random draws from the ergodic, i.e., long-run distribution of ( a, x, a ′ , x ′ ). 8 Let P denote such a distribution over ( a, x, a ′ , x ′ ), and E [ · ] the corresponding expectation over P . Define E n [ · ] as the expectation over the empirical distribution, P n , of ( a, x, a ′ , x ′ ). In particular, E n [ f ( a, x, a ′ , x ′ )] := ( n ( T -1)) -1 ∑ n i =1 ∑ T -1 t =1 f ( a it , x it , a it +1 , x it +1 ), i.e., we always drop the last time period in the summation index even if f ( · ) does not depend on a ′ , x ′ .

Let H denote the space of all square integrable functions over the domain A×X of ( a, x ). Define the pseudo-norm ‖·‖ 2 over H as ‖ f ‖ 2 := E [ | f ( a, x ) | 2 ] 1 / 2 for all f ∈ H . We use |·| to denote the usual Euclidean norm on a Euclidean space.

## 3. Temporal-difference estimation

This section presents our TD estimation of h ( · ) and g ( · ). Note that h ( · ) is a vector of the same dimension as θ ∗ . Our methods provide functional approximations separately for each component h ( j ) of h . To simplify notation, we drop the superscript j indexing the elements of h ( · ) and proceed as if the latter, and therefore θ ∗ , is a scalar. However, all our results hold for general h ( · ), as long as each of its elements is treated separately.

8 This is a slightly stronger requirement than the one imposed by Aguirregabiria and Mira (2002), who assume only that { a it , x it , a it +1 } i are i.i.d. draws from a distribution ˘ P satisfying ˘ P ( x it = x ) &gt; 0 for all x ∈ support( X ). In the discrete case, ˘ P and the ergodic distribution P are mutually absolutely continuous, with d ˘ P /d P ≤ C &lt; ∞ . As a result, replacing P with ˘ P would only introduce a constant multiplicative factor in our results, without altering the convergence rates. Moreover, as Aguirregabiria and Mira (2002) note, the i.i.d. assumption is often used as a convenient approximation for timeseries dynamics. But the equivalence between time-series and cross-sectional analysis holds only under the ergodic distribution, which guarantees that cross-sectional expectations coincide with long-run time averages.

For any candidate function, f ( a, x ), for h ( a, x ), denote the TD error by

<!-- formula-not-decoded -->

and the dynamic programming operator by

<!-- formula-not-decoded -->

Clearly, h ( a, x ) is the unique fixed point of Γ z [ · ]. TD estimation involves approximating h ( a, x ) using a functional class F , where each element h ( · ; ω ) of F is indexed by a finitedimensional vector ω . The aim is to ostensibly minimize the mean-squared TD error

<!-- formula-not-decoded -->

However, this minimization problem is neither computationally feasible nor is it proven to converge when the true h / ∈ F . Instead, two approaches are commonly used.

The first approach, the semi-gradient method, involves updating ω as

<!-- formula-not-decoded -->

for some small value of α . As the name suggests, the above is not a complete gradient as the derivative does not take into account how ω affects the 'target', i.e., the future value h ( a ′ , x ′ ; ω ). Nevertheless, for linear functional classes F , it is possible to explicitly characterize the limit point of the updates, ω ∗ , and compute it directly. Section 3.1 describes this in greater detail. In the RL literature, it is common to employ semigradients with Neural Networks as the functional class F , but it appears difficult to extend our theoretical analysis to this setting (we can, however, use Neural Networks with our AVI procedure described below).

The second approach, Approximate Value Iteration (AVI; Munos and Szepesvári, 2008), employs the idea of 'target networks'. Here, the parameters in the future value of h are fixed at the current ω , and the functional parameters iteratively updated as

<!-- formula-not-decoded -->

Clearly, the semi-gradient method and AVI are closely related: if one were to solve the problem in (3.2) using gradient descent, the updates within each iteration would look similar to (3.1) except for fixing the value of ω in h ( a ′ , x ′ ; ω ) at the past values. After the updates converge, i.e., at the end of the iteration, h ( a ′ , x ′ ; ω ) is revised with the new

ω . The semi-gradient approach can thus be considered a one-step variant of AVI. Section 3.2 describes AVI in more detail. We characterize the theoretical properties of AVI under general functional classes F including Neural Networks, Random Forests, LASSO etc.

The approximation to g follows similarly after replacing δ z ( · ; f ) , Γ z [ · ] by

<!-- formula-not-decoded -->

3.1. Semi-gradients. Let φ ( a, x ) consist of a set of basis functions over the domain ( a, x ). Then the linear approximation class is F ≡ { φ ( a, x ) ⊺ ω : ω ∈ R k φ } , where k φ = dim( φ ). Denote the projection operator onto F by P φ :

<!-- formula-not-decoded -->

For linear basis functions, it can be shown, e.g., Tsitsiklis and Van Roy (1997), that the sequence of functional approximations h ( a, x ; ω j ) := φ ( a, x ) ⊺ ω j converges to h ∗ := φ ( a, x ) ⊺ ω ∗ , defined as the fixed point of the projected dynamic programming operator P φ Γ z [ · ] (i.e., P φ Γ z [ h ∗ ] = h ∗ ). Based on this characterization, we show in Lemma 1 (Online Appendix B.2) that h ∗ ( a, x ) = φ ( a, x ) ⊺ ω ∗ , where

<!-- formula-not-decoded -->

Lemma 2 in Online Appendix B.2 assures that E [ φ ( a, x ) ( φ ( a, x ) -βφ ( a ′ , x ′ )) ⊺ ] is indeed non-singular as long as β &lt; 1 and E [ φ ( a, x ) φ ( a, x ) ⊺ ] is non-singular. While ω ∗ cannot be computed directly, we can obtain an estimator, ˆ ω , by replacing E [ · ] with the sample expectation E n [ · ]:

<!-- formula-not-decoded -->

Using ˆ ω , we estimate h ( · ) as ˆ h ( a, x ) = φ ( a, x ) ⊺ ˆ ω .

We now turn to the estimation of g ( · ). As with h ( · ), we approximate g ( · ) using basis functions r ( a, x ), which may generally be different from φ ( a, x ). Let P r denote the projection operator onto the space { r ( a, x ) ⊺ ξ : ξ ∈ R k r } , where k r = dim( r ). The limit of the semi-gradient iterations is g ∗ ( a, x ) := r ( a, x ) ⊺ ξ ∗ , defined as the fixed point of P r Γ e [ · ]. We thus obtain the following characterization of ξ ∗ in analogy with (3.3):

<!-- formula-not-decoded -->

In the above, e ( a, x ) = γ -ln P ( a | x ) is a function of unknown choice probabilities. Denote η ( a, x ) := P ( a | x ). Suppose that we have access to a non-parametric estimator ˆ η of η , e.g., through series or kernel regression. We can then use this estimate to obtain e ( a, x ; ˆ η ) := γ -ln ˆ η ( a, x ). This in turn enables us to estimate ξ ∗ using ˆ ξ , computed as

<!-- formula-not-decoded -->

Using the above, we estimate g ( · ) as ˆ g ( a, x ) = r ( a, x ) ⊺ ˆ ξ . Algorithm 3 in Online Appendix D describes the estimation steps for both ˆ ω and ˆ ξ .

Interestingly, estimation of ξ ∗ is unaffected to a first order by the estimation of ˆ η , even though the latter converges to the true η at non-parametric rates (see Section 4 for a formal statement). This is because of an orthogonality property for the estimation of ξ :

<!-- formula-not-decoded -->

where ∂ η · denotes the Fréchet derivative with respect to η . To show (3.7), expand

<!-- formula-not-decoded -->

where the first equality follows from the Markov property. Consider the functional M (˜ η ) := E [ ln ˜ η ( a ′ , x ′ ) | x ′ ] at different candidate values ˜ η ( · , · ). At the true conditional choice probability, η , M (˜ η ) becomes the conditional entropy of P ( a | x ′ ) and attains its maximum. Hence, ∂ η E [ ln η ( a ′ , x ′ ) | x ′ ] = 0 and (3.7) follows from (3.8). Consequently, ˆ ξ is a locally robust estimator for ξ .

Computation of ˆ ω and ˆ ξ is very cheap as it only involves solving linear equations of dimension dim( φ ) and dim( r ), respectively. Using ˆ h ( a, x ) and ˆ g ( a, x ), we can in turn estimate θ ∗ in many different ways. For instance, we can use the PMLE estimator

<!-- formula-not-decoded -->

However, such plug-in estimates are sub-optimal. In Section 4.2, we suggest a locally robust version of (3.9).

Suppose that the underlying states and actions are discrete, and that our algorithm uses the set of all discrete elements of x, a as basis functions. We show in Online Appendix

B.1 that the resulting estimate of h ( a, x ) is identical to that obtained from the standard CCP estimators, if the choice and transition probabilities were estimated using cell values.

A limitation of the linear semi-gradient method is that it requires one to choose a series basis and also does not allow for high-dimensional state spaces ( i.e., dim( x ) ∝ n ). The AVI method, described below, does not share this limitation.

3.2. Approximate Value Iteration (AVI). For a feasible estimation procedure using AVI, we can replace E [ · ] by E n [ · ] in (3.2). The procedure builds a sequence of approximations { ˆ h j ; j = 1 , . . . , J } for h , where

<!-- formula-not-decoded -->

The process can be started with an arbitrary initialization, e.g., ˆ h 1 ( a, x ) = z ( a, x ). The maximum number of iterations, J , is only limited by computational feasibility. 9

The minimization problem (3.10) is equivalent to a prediction problem using the functional class F , where the outcomes are z ( a, x ) + β ˆ h j ( a ′ , x ′ ). Hence, the estimation target for ˆ h j +1 is the conditional expectation E [ z ( a, x )+ β ˆ h j ( a ′ , x ′ ) | a, x ] ≡ Γ z [ ˆ h j ]( a, x ), i.e., each ˆ h j +1 is a non-parametric approximation to Γ z [ ˆ h j ], and in this manner AVI builds a series of approximate value function iterations.

The interpretation of (3.10) as a prediction problem enables us to employ any machine learning method devised for prediction, including (but not limited to) LASSO, Random Forests and Neural Networks. Our theoretical results show that it is possible to estimate h at suitably fast rates under very weak assumptions on the non-parametric estimation rates of machine learning methods.

The estimation procedure for g ( · ) is similar: we construct a sequence of approximations { ˆ g j , j = 1 , . . . , J } for g as

<!-- formula-not-decoded -->

As in Section 3.1, it will be shown that the estimation error of η is first-order ignorable for the estimation of g . Using ˆ h ( a, x ) and ˆ g ( a, x ), we can, as before, estimate θ ∗ in many different ways, including the PMLE estimator (3.9).

Compared to the semi-gradient approach, AVI is computationally more expensive as it requires solving J prediction problems (in Section 4.1 we show that in the worst case

9 In practice, we suggest monitoring ε 2 j := E n [ ‖ ˆ h j +1 -ˆ h j ‖ 2 ] / E n [ ‖ ˆ h j -E n [ ˆ h j ] ‖ 2 ] , the L 2 distance between successive iterations scaled by the variance of ˆ h j . We could keep increasing J until ε J goes below a pre-determined threshold, say 0.01.

J ≈ ln n , but this can be substantially reduced through good initializations). However, semi-gradient methods require differentiable classes of functions (e.g., Random Forests are not allowed) and it appears difficult to characterize their theoretical properties beyond the case of linear basis functions.

A note on implementing (3.10): Since z ( a, x ) is known, we recommend running a nonparametric regression of only ˆ h j ( a ′ , x ′ ) on ( a, x ) at each step. We can then multiply the resulting non-parametric estimator by β and add back z ( a ′ , x ′ ) to obtain the next estimate ˆ h j +1 ( · ). A similar comment applies to (3.11). Algorithm 1 describes the estimation steps.

## Algorithm 1 AVI using Random Forest

Require: Non-parametric estimate ˆ η ; initial values ˆ h 1 , ˆ g 1 ; J (# iterations)

- 1: for j = 1 , 2 , . . . , J -1: do
- 2: Predict β ˆ h j ( a ′ , x ′ ) and βe ( a ′ , x ′ , ˆ η )+ β ˆ g j ( a ′ , x ′ ) using ( a, x ) with Random Forest, obtain prediction functions ˜ h j +1 ( · ) and ˆ g j +1 ( · )
- 3: ˆ h j +1 ( · ) ← ˜ h j +1 ( · ) + z ( · )
- 4: end for
- 5: Return ˆ h ( a, x ) = ˜ h J ( a, x ) + z ( a, x ) and ˆ g ( a, x ) = ˆ g J ( a, x )

Notes: We recommend estimating η with a logit model using a 2 nd or 3 rd order polynomial in x , and setting ˆ h 1 = (1 -β ) -1 E n [ z ( a, x )], ˆ g 1 = β (1 -β ) -1 E n [ e ( a ′ , x ′ ; ˆ η )] and J = 20 as defaults (or else, employ the procedure set out in Footnote 9 for J ). The Random Forest tuning parameters ntree and mtry can be kept at default values, but we suggest checking whether the results change meaningfully if mtry varies by ± 1 (if they do, a cross-validation function can be used to determine mtry , e.g., rfcv in R ).

3.3. Tuning parameters. Both the semi-gradient and AVI methods require choosing tuning parameters. For AVI this is straightforward: as each iteration is a non-parametric estimation problem, the tuning parameters can be chosen in the usual manner, e.g., through cross-validation. In the case of linear semi-gradient methods, the tuning parameters are the dimensions k φ = dim( φ ) and k r = dim( r ) of the basis functions. In analogy with AVI, we propose selecting both through a procedure akin to cross-validation. The value of ω is estimated using a training sample and its performance evaluated on a hold-out or test sample, where the performance is measured in terms of the empirical mean-squared TD error E n, test [ δ 2 z ( a, x, a ′ , x ′ ; ˆ h )] on the test dataset. The values of k φ , k r are chosen to minimize the mean squared TD error (see Section 6.2.1 for an example).

3.4. Unobserved heterogeneity. In Online Appendix C, we incorporate permanent unobserved heterogeneity by pairing our TD methods with the sequential ExpectationMaximization (EM) algorithm (Arcidiacono and Jones, 2003). This algorithm can handle

discrete heterogeneity in both individual utilities and transition densities. Monte Carlo evidence suggests that the algorithm works well in practice (see Online Appendix E.2).

## 4. Theoretical Properties of TD estimators

4.1. Estimation of non-parametric terms. We characterize rates of convergence for estimation of h ( · ) and g ( · ) under both semi-gradients and AVI.

4.1.1. Linear semi-gradients. We impose the following assumptions for estimation of h ( · ).

Assumption 1. (i) The basis vector φ ( a, x ) is linearly independent (i.e., φ ( a, x ) ⊺ ω = 0 for all ( a, x ) if and only if ω = 0 ). Additionally, the eigenvalues of E [ φ ( a, x ) φ ( a, x ) ⊺ ] are uniformly bounded away from zero for all k φ := dim ( φ ) .

- (ii) | φ ( a, x ) | ∞ ≤ M for some M &lt; ∞ .
- (iii) There exists C &lt; ∞ and α &gt; 0 such that ‖ h -P φ [ h ] ‖ 2 ≤ Ck -α φ .
- (iv) The domain of ( a, x ) is a compact set, and | z ( a, x ) | ∞ ≤ L for some L &lt; ∞ .
- (v) k φ →∞ and k 2 φ /n → 0 as n →∞ .

Assumption 1(i) rules out multi-collinearity in the basis functions. This is easily satisfied. Assumption 1(ii) ensures that the basis functions are bounded. This is again a mild requirement and is easily satisfied if either the domain of ( a, x ) is compact, or the basis functions are chosen appropriately (e.g., a Fourier basis). Assumption 1(iii) is a standard condition on the rate of approximation of h ( a, x ) using a basis approximation. The value of α is related to the smoothness of h ( · ). Newey (1997) shows that for splines and power series, α = r/d , where r is the number of continuous derivatives of h ( a, · ) and d is the dimension of x . Similar results can also be derived for other approximating functions such as Fourier series, wavelets and Bernstein polynomials. The smoothness properties of h ( a, · ) are discussed in Online Appendix B.3.2, where we provide primitive conditions on z ( a, x ) , K ( x ′ | a, x ) that ensure existence of r continuous derivatives of h ( a, · ) for each a ∈ A . Assumption 1(iv) requires z ( a, x ) to be bounded. Finally, Assumption 1(v) specifies the rate at which the dimension of the basis functions is allowed to grow. The rate requirements are mild, and are the same as those employed for standard series estimation. For the theoretical properties, the exact rate of k φ is not relevant up to a first order since we propose estimators of θ ∗ that are locally robust to estimation of h ( · ).

We then have the following theorem on the estimation of h ( a, x ):

Theorem 1. Under Assumption 1, the following holds:

- (i) Both ω ∗ and ˆ ω exist, the latter with probability approaching one.

<!-- formula-not-decoded -->

(iii) The L 2 error for the difference between h ( a, x ) and φ ( a, x ) ⊺ ˆ ω is bounded as

<!-- formula-not-decoded -->

We prove Theorem 1 in Appendix A.1 by adapting the results of Tsitsiklis and Van Roy (1997). Part (i) ensures that the population and empirical TD fixed points exist. Parts (ii) and (iii) imply that the approximation bias and MSE of linear semi-gradients are analogous to those of standard series estimation apart from a (1 -β ) -1 factor.

For the estimation of ˆ ξ we make use of cross-fitting as a technical device to obtain easyto-verify assumptions on the estimation of η . This entails the following: we randomly partition the data into two folds. We estimate ˆ ξ separately for each fold using ˆ η estimated from the opposite fold. The final estimate of ξ ∗ is the weighted average of ˆ ξ from both the folds. For specific estimation methods, e.g., series estimation, it is possible to derive our theoretical results without cross-fitting; the latter may then be unnecessary in practice.

We impose the following assumptions for the estimation of g ( a, x ).

Assumption 2. (i) The basis vector r ( a, x ) is linearly independent, and the eigenvalues of E [ r ( a, x ) r ( a, x ) ⊺ ] are uniformly bounded away from zero for all k r := dim ( r ) .

- (ii) | r ( a, x ) | ∞ ≤ M for some M &lt; ∞ .
- (iii) There exists C &lt; ∞ and α &gt; 0 such that ‖ g -P r [ g ] ‖ 2 ≤ Ck -α r .
- (iv) The domain of ( a, x ) is a compact set, and | e ( a, x ) | ∞ ≤ L &lt; ∞ .
- (v) k r →∞ and k 2 r /n → 0 as n →∞ .

(vi) ˆ ξ is estimated from a cross-fitting procedure described above. The conditional choice probability function satisfies η ( a, x ) &gt; δ &gt; 0 , where δ is independent of a, x . Additionally, ‖ η -ˆ η ‖ ∞ = o p (1) and ‖ η -ˆ η ‖ 2 2 = o p ( n -1 / 2 ) .

Assumption 2 is a direct analogue of Assumption 1, except for the last part which provides regularity conditions when η ( · ) is estimated. These conditions are typical for locally robust estimates and only require the non-parametric function η ( a, x ) to be estimable at faster than n -1 / 4 rates. This is easily verified for most non-parametric estimation methods such as kernel or series regression. Under these assumptions, we have the following analogue of Theorem 1, which we prove in Appendix A.2 .

Theorem 2. Under Assumption 2, the following holds:

- (i) Both ξ ∗ and ˆ ξ exist, the latter with probability approaching one.

<!-- formula-not-decoded -->

(iii) The L 2 error for the difference between g ( a, x ) and r ( a, x ) ⊺ ˆ ξ is bounded as

<!-- formula-not-decoded -->

4.1.2. Approximate Value Iteration. We can expand the estimation error ∥ ∥ ∥ h -ˆ h J ∥ ∥ ∥ 2 in terms of the non-parametric estimation errors ∥ ∥ ∥ Γ z [ ˆ h j -1 ] -h j ∥ ∥ ∥ 2 for j = 1 , . . . , J . In particular, since Γ z [ h ] = h and Γ z [ · ] is a β -contraction, we have

<!-- formula-not-decoded -->

Iterating the above gives

<!-- formula-not-decoded -->

Equation (4.1) is a special case of error propagation (Munos and Szepesvári, 2008).

Recall that ˆ h 1 is an arbitrary initialization. It is thus straightforward to provide conditions under which ∥ ∥ ∥ h -ˆ h 1 ∥ ∥ ∥ 2 is bounded by some constant M 1 . As for the second term in (4.1), recall from the discussion in Section 3.2 that the minimization problem (3.10) corresponds to non-parametric estimation of Γ z [ ˆ h j -1 ]. Most machine learning methods come with guarantees on the non-parametric estimation rate ∥ ∥ ∥ Γ z [ ˆ h j -1 ] -ˆ h j ∥ ∥ ∥ 2 .

We now describe our assumptions for AVI. Let X denote the d -dimensional space of x , and define W γ, ∞ M ( X ) as the Hölder ball with smoothness parameter γ :

<!-- formula-not-decoded -->

Assumption 3. There exist M 0 , M &lt; ∞ such that:

- (i) The domain, X , of x is compact, | h | ∞ ≤ M 0 and h ( a, · ) ∈ W γ, ∞ M ( X ) for each a .
- (ii) ∣ ∣ ∣ ˆ h 1 ∣ ∣ ∣ ∞ ≤ M 0 and ∥ ∥ ∥ h -ˆ h 1 ∥ ∥ ∥ 2 ≤ M 1 for some M 1 &lt; ∞ .
- (iii) | Γ z [ f ] | ∞ ≤ M 0 and Γ z [ f ]( a, · ) ∈ W γ, ∞ M ( X ) for all a ∈ A and { f : | f | ∞ ≤ M 0 } .

(iv) The candidate class of functions F is such that | f | ∞ ≤ M 0 for all f ∈ F . Additionally, consider the non-parametric estimation problem (with i.i.d. observations i = 1 , . . . , n and T fixed): ˆ f = arg min ˜ f ∈F ∑ n i =1 ∑ T -1 t =1 ( y it -˜ f ( a it , x it )) 2 , where y it is compactly supported and E [ y it | a it , x it ] = f ( a it , x it ) for some f ∈ W γ, ∞ M ( X ) . Then, uniformly over all

f ∈ W γ, ∞ M ( X ) , E [∥ ∥ ∥ f -ˆ f ∥ ∥ ∥ 2 ] ≤ Cn -c for constants C &lt; ∞ , c &gt; 0 independent of n , but C may depend on M,M 0 , γ and c on γ .

Assumption 3(i) is not needed to obtain a convergence rate for the AVI estimator, but we state it here as it is useful for subsequent results. The assumption of γ -Hölder continuity is taken from Farrell et al. (2021). Assumption 3(ii) is a mild condition on the initialization ˆ h 1 . Assumption 3(iii), which is novel to this paper, is a crucial smoothness condition requiring the operator Γ z [ · ]( a, · ) to map all bounded f onto W γ, ∞ M ( X ). In Online Appendix B.3.2, we show that both requirements in Assumption 3(iii) are satisfied if z ( a, · ) and K ( x ′ | a, · ) are γ -Hölder continuous.

Assumption 3(iv) is a high-level condition on the machine learning (ML) method F . The requirement of bounded f implies that the ML method cannot diverge in the l ∞ sense, see Farrell et al. (2021) for a discussion of this in the context of multi-layer perceptrons (MLPs). The second part of Assumption 3(iv) implies that the ML method is able to non-parametrically approximate all functions in W γ, ∞ M ( X ) at the rate of at least n -c . Most ML methods are proven to satisfy this. Consider, for instance, the class F of MLPs of width W and depth L ; MLPs and, more generally, Neural Networks are widely used in RL. Farrell et al. (2021) show that for W /equivasymptotic n d 2( γ + d ) ln 2 n and L /equivasymptotic ln n ,

<!-- formula-not-decoded -->

Thus, Assumption 3(iv) is satisfied for MLPs. See Biau (2012) for related results on Random Forests. Note that Assumption 3(iv) is also the only way in which the dimension of x enters our estimation. Through a suitable choice of the ML method, e.g., Random Forests or LASSO, we can allow dim( x ) to be proportional to, or even bigger than n .

Assumptions 3(iii) and 3(iv) imply that one can estimate Γ[ f ] for any | f | ∞ ≤ M 0 at the n -c rate, i.e., sup j E [∥ ∥ ∥ Γ z [ ˆ h j -1 ] -ˆ h j ∥ ∥ ∥ 2 ] ≤ Cn -c . Combined with (4.1), this proves:

Theorem 3. Suppose Assumptions 3(ii) to 3(iv) hold. Then, for all n large enough,

<!-- formula-not-decoded -->

See Online Appendix B.3.3 for a formal proof of Theorem 3. The first term in the expression for E [∥ ∥ ∥ h -ˆ h J ∥ ∥ ∥ 2 ] from Theorem 3 is the statistical rate of estimation of h . The second term is the numerical error, which is seen to decline exponentially with the number of iterations J . Setting J /equivasymptotic ln n will ensure the numerical error is smaller than

the statistical rate of convergence. The number of iterations can be further reduced using a good initialization, ˆ h 1 , that makes M 1 small. For instance, initializing using the linear semi-gradient estimator, which is fast to compute, ensures M 1 = o p (1). Incidentally, Theorem 3 justifies the use of Neural Networks for batch RL; to the best of our knowledge this appears to be new even in the RL literature.

Turning to estimation of ˆ g , we again assume cross-fitting is employed as in Theorem 2, i.e., ˆ η is computed from one half of the data, and ˆ g is computed using AVI on the other half, taking ˆ η as given. Define Γ e, ˜ η [ f ]( a, x ) := β E [ e ( a ′ , x ′ ; ˜ η ) + f ( a ′ , x ′ ) | a, x ], where ˜ η is any candidate function for η .

Assumption 4. (i) Let ˜ η ( a, x ) ∈ [0 , 1] be any function such that inf a,x ˜ η ( a, x ) &gt; δ &gt; 0 . Then, there exist M 1 , M, C &lt; ∞ , that may depend on δ but are otherwise independent of ˜ η ( · ) , such that Assumptions 3(i) 3(iv) hold after replacing ( h, ˆ h 1 , Γ z [ · ]) with ( g, ˆ g 1 , Γ e, ˜ η [ · ]) .

(ii) ˆ g is estimated from a cross-fitting procedure. The true conditional choice probability function satisfies inf a,x η ( a, x ) &gt; δ &gt; 0 . Additionally, ‖ η -ˆ η ‖ 2 2 = o p ( n -1 / 2 ) and with probability approaching one, inf a,x ˆ η ( a, x ) &gt; δ &gt; 0 .

Assumption 4(i) requires analogues of Assumption 3 to hold. In Online Appendix B.3.2, we show that analogues of Assumptions 3(i) and 3(iii) are satisfied as long as K ( x ′ | a, · ) is γ -Hölder continuous (the other assumptions simply place restrictions on the initial value and the ML method used). Assumption 4(ii) is similar to Assumption 2(vi).

Theorem 4. Suppose Assumption 4 holds. Then, with probability approaching one,

<!-- formula-not-decoded -->

See Online Appendix B.3.4 for a formal proof of Theorem 4.

4.2. Estimation of structural parameters. Estimation of h ( a, x ) and g ( a, x ) is inherently non-parametric because these functions depend on two non-parametric terms: the choice probabilities η ( a, x ), and the transition densities K ( x ′ | a, x ). The TD estimators implicitly take both into account. Under the PMLE criterion, the estimates for K ( x ′ | a, x ) and θ ∗ are not orthogonal to each other and this extends to the lack of orthogonality between the estimates ˆ h , ˆ g and θ ∗ . 10 We allow θ ∗ to be vector-valued for the remainder of this section.

10 For discrete states, this lack of orthogonality implies an additional variance term for the structural parameter estimates, though the rates of convergence are still parametric. With continuous states,

We can recover √ n -consistent estimation by adjusting the PMLE criterion to account for the first-stage estimation of h and g . Denote (˜ a , ˜ x ) := ( a, x, a ′ , x ′ ) and m ( a, x ; θ, h, g ) := ∂ θ ln π ( a, x ; θ, h, g ), where

<!-- formula-not-decoded -->

The PMLE estimator with plug-in estimates solves E n [ m ( a, x ; θ, ˆ h, ˆ g )] = 0, but this is not robust to estimation of h, g . Let V ( a, x ; θ, h, g ) := h ( a, x ) ⊺ θ + g ( a, x ) denote the continuation value given ( a, x ). Also, define λ ( a, x ; θ ) as the fixed point of the 'backward' dynamic programming operator

<!-- formula-not-decoded -->

where ( a -′ , x -′ ) denotes the past actions and states preceding ( a, x ), and

<!-- formula-not-decoded -->

In Online Appendix B.4, we show that the locally robust moment corresponding to m ( a, x ; θ, h, g ) is given by

<!-- formula-not-decoded -->

Crucially, the correction term is not required to be a function of θ (though if we replaced θ ∗ in (4.4) with θ , that would be a valid correction term too).

The construction of the locally robust moment (4.4) is new. But it is infeasible since θ ∗ , λ ( · ) , h ( · ) , g ( · ) and η ( · ) are unknown. However, we can replace these quantities with consistent estimates. We have already described how to estimate η ( · ) , h ( · ) , g ( · ). Recall that ˜ θ denotes the plug-in estimator of θ ∗ using (3.9); note that ˜ θ consistently estimates θ ∗ but is not efficient. An estimator, ˆ λ ( · ), of λ ( · ) can then be obtained by applying either of our TD estimation methods on (4.2), with ˜ θ, ˆ h, ˆ g, ˆ η replacing θ ∗ , h, g, η . For instance, using AVI, we could obtain iterative approximations { ˆ λ ( j ) , j = 1 , . . . , J } for λ ( · ) using

<!-- formula-not-decoded -->

however, the PMLE estimator with plug-in values of ˆ h and ˆ g will converge at slower than parametric rates.

Plugging in ˆ λ ( · ) , ˆ h, ˆ g, ˆ η into (4.4), we obtain the feasible locally robust moment

<!-- formula-not-decoded -->

Using the above, we can obtain a locally robust estimator, ˆ θ , as the solution to E n [ ζ (˜ a , ˜ x ; θ, ˆ h, ˆ g, ˆ η, ˆ λ, ˜ θ )] = 0. We recommend obtaining this using cross-fitting, see Section 4.2.1 for details. Compared to the plug-in estimate (3.9), our locally robust estimator requires computation of λ ( · ), but when linear semi-gradients are used to estimate h, g , we can even derive a closedform expression for λ ( · ), see Online Appendix B.5. Solving E n [ ζ (˜ a , ˜ x ; θ, ˆ h, ˆ g, ˆ η, ˆ λ, ˜ θ )] = 0 is also computationally easy; the correction term is a constant, and ∇ θ ζ (˜ a , ˜ x ; θ, ˆ h, ˆ g, ˆ η, ˆ λ, ˜ θ ) = ∇ θ m ( a, x ; θ, ˆ h, ˆ g ) is negative definite (as the PMLE criterion is concave), so solving this is no harder than solving the original moment condition without a correction term.

4.2.1. √ n -consistent estimation. We focus on the general construction of the locally robust estimator, ˆ θ , using (4.6). As mentioned in the previous sub-section, we advocate cross-fitting to obtain this estimator. Algorithm 2 describes the estimation steps.

## Algorithm 2 Structural parameter estimation

Require: Non-parametric estimate ˆ η ; initial values ˆ h 1 , ˆ g 1 ; J (# iterations)

- 1: Split the data into two equal folds N 1 , N 2
- 2: for each N k , k = { 1 , 2 } : do
- 3: Run Algorithm 1 to obtain ˆ h ( k ) , ˆ g ( k )
- 4: Obtain preliminary estimates ˜ θ ( k ) := arg max θ ( k ) ˆ Q ( θ ( k ) ) as in (3.9)
- 5: Run Algorithm 5 (Online Appendix D) with ˆ h ( k ) , ˆ g ( k ) and ˜ θ ( k ) as inputs to obtain ˆ λ ( k )
- 6: Using plug-in quantities ˜ θ ( -k ) , ˆ h ( -k ) , ˆ g ( -k ) , ˆ η ( -k ) , ˆ λ ( -k ) from the other fold N -k , obtain ˆ θ ( k ) by solving E ( k ) n [ ζ (˜ a , ˜ x ; θ, ˆ h ( -k ) , ˆ g ( -k ) , ˆ η ( -k ) , ˆ λ ( -k ) , ˜ θ ( -k ) )] = 0, as in (4.6), where E ( k ) n [ · ] denotes the empirical expectation using only observations from N k
- 7: end for
- 8: Obtain the final estimate ˆ θ = ( ˆ θ (1) + ˆ θ (2) ) / 2

Notes: We recommend estimating η with a logit model using a 2 nd or 3 rd order polynomial in x , and setting ˆ h 1 = (1 -β ) -1 E n [ z ( a, x )], ˆ g 1 = β (1 -β ) -1 E n [ e ( a ′ , x ′ ; ˆ η )] and J = 20 as defaults (or else, employ the procedure set out in Footnote 9 for J ). The Random Forest tuning parameters ntree and mtry can be kept at default values, but we suggest checking whether the results change meaningfully if mtry varies by ± 1 (if they do, a cross-validation function can be used to determine mtry , e.g., rfcv in R ).

Following the analysis of Chernozhukov et al. (2022), it can be shown that this estimator has the same limiting distribution as the one based on (4.4). In particular, it achieves parametric rates of convergence. We state the regularity conditions below:

Assumption 5. (i) θ ∗ ∈ Θ , a compact set, and E [ m ( a, x ; θ, h, g )] = 0 ⇐⇒ θ = θ ∗ .

(ii) There exists a neighborhood, N , of θ ∗ such that uniformly over θ ∈ N and for ‖ ˜ h -h ‖ , ‖ ˜ g -g ‖ sufficiently small, ∥ ∥ ∥ ∇ θ m ( a, x ; θ, ˜ h, ˜ g ) -∇ θ m ( a, x ; θ ∗ , ˜ h, ˜ g ) ∥ ∥ ∥ ≤ d ( a, x ) ‖ θ -θ ∗ ‖ , where E [ d ( a, x )] &lt; ∞ . Furthermore, G := E [ ∇ θ m ( a, x ; θ ∗ , h, g )] is invertible.

(iii) ∥ ∥ ∥ ˆ h -h ∥ ∥ ∥ 2 = o p ( n -1 / 4 ) , ‖ ˆ g -g ‖ 2 = o p ( n -1 / 4 ) and ‖ ˆ η -η ‖ 2 = o p ( n -1 / 4 ) . Furthermore, h, g are continuous, ∥ ∥ ∥ ˆ h ∥ ∥ ∥ ∞ , ‖ ˆ g ‖ ∞ ≤ M &lt; ∞ and there exists δ &gt; 0 such that inf a,x η ( a, x ) &gt; δ and inf a,x ˆ η ( a, x ) &gt; δ with probability approaching one.

<!-- formula-not-decoded -->

Assumption 5(i) implies θ ∗ is identified. When h, g are exactly known, m ( · ) is just the derivative of the pseudo-log-likelihood (2.1); the latter is always concave. Assumption 5(i) is satisfied if E [ ∇ θ m ( a, x ; θ, h, g )] is strictly positive-definite at θ ∗ . This is essentially equivalent to the requirement for identification in the discrete state space regime, which previous work e.g., Aguirregabiria and Mira (2002), has also assumed. 11 For instance, when the action space is binary ( a ∈ { 0 , 1 } ), direct computation shows that a sufficient condition for Assumption 5(i) is:

<!-- formula-not-decoded -->

where η ( a, x ) := π ( a, x ; θ ∗ , h, g ) is the true conditional choice probability, and ' /follows 0' indicates that the matrix in question is strictly positive-definite. In fact, under our assumptions (continuity of h, g and compactness of x ), η ( a, x ) &gt; δ &gt; 0 independently of a, x , so we can further rewrite the sufficient condition as Ω := E [ { h (1 , x ) -h (0 , x ) } { h (1 , x ) -h (0 , x ) } ⊺ ] /follows 0. This holds as long as h (1 , x ) -h (0 , x ) is not linearly dependent at (almost surely) every x . For β = 0, it is equivalent to linear independence of z (1 , x ) -z (0 , x ). 12

Assumption 5(ii) is a mild regularity condition that is similar to Assumption 4 in Chernozhukov et al. (2022). The first part of Assumption 5(iii) follows from Theorems 1-4 under suitable conditions on the degree of smoothness of h, g . For instance, it is satisfied for AVI with Neural Networks if γ ≥ d . The second part of Assumption 5(iii) is mild, and is satisfied as long as ˆ h, ˆ g, ˆ η are continuous (given that the support of x

11 When the error distribution is unspecified, Buchholz et al. (2021) show that identification of θ ∗ is feasible only when β is sufficiently smaller than 1. Their findings do not directly apply to our setting as we assume the errors follow a Type I Extreme Value distribution; however, we do also require β &lt; 1. 12 More generally, under the ergodic distribution, Ω = (1 -β 2 ) -1 { Ψ 0 + ∑ ∞ j =0 β 2 j (Ψ j + Ψ ⊺ j ) } , where Ψ j := E [ { z (1 , x t ) -z (0 , x t ) } { z (1 , x t + j ) -z (0 , x t + j ) } ⊺ ]. One can then posit various conditions on β and { Ψ j } j such that Ω /follows 0. For instance, if Ψ j +Ψ ⊺ j is positive semi-definite and Ψ 0 /follows 0, we have Ω /follows 0 for any β &lt; 1. However, β = 1 is never possible. We leave open the discussion of alternative conditions.

is compact). In fact, we also directly impose this restriction in the context of the AVI estimator. Importantly Assumption 5(iii) only requires L 2 -convergence of ˆ h, ˆ g, ˆ η , and not uniform convergence. This is due to the use of a locally robust moment together with cross-fitting; see Chernozhukov et al. (2022) for a discussion of how they enable very mild assumptions on the convergence rates of ML estimators. 13

Assumption 5(iv) requires λ ( · , · ; θ ∗ ) to be estimable at faster than n -1 / 4 rates as well. If h, g are known, it is straightforward to derive n -1 / 4 rates as in Theorems 1-4. For plugin estimation, we would need additional assumptions. For instance, we could employ three-way sample splitting as in Chernozhukov et al. (2018) where the first third of the sample is used to compute ˆ h, ˆ g, ˆ η, ˜ θ , and these estimates are then plugged into the second third of the sample to estimate λ . Lemma 8 in Online Appendix B.6 then shows that Assumption 5(iv) holds under the previous assumptions and some mild conditions on the AVI estimator (4.5) as long as ˆ h ( a, · ) , ˆ g ( a, · ) ∈ W γ, ∞ M ( X ) for some M &lt; ∞ at each a ∈ A and γ is sufficiently large. It is possible to verify Assumption 5(iv) without three-way sample splitting as well, but this requires much stronger regularity conditions.

We are now ready to state the main result of this section.

Theorem 5. Suppose that either Assumptions 1, 2 &amp; 5 (for linear semi-gradients) or 3-5 (for AVI) hold. Then the estimator, ˆ θ of θ ∗ , based on (4.6) is √ n -consistent, and satisfies

<!-- formula-not-decoded -->

where V = ( G ⊺ Ω -1 G ) -1 , with Ω := E [ ζ (˜ a , ˜ x ; θ ∗ , h, g, η, λ, θ ∗ ) ζ (˜ a , ˜ x ; θ ∗ , h, g, η, λ, θ ∗ ) ⊺ ] .

The proof of the above theorem follows by verifying the regularity conditions of Chernozhukov et al. (2022, Theorem 9), see Appendix A.3 for the details. For inference on ˆ θ , the covariance matrix V can be estimated as ˆ V = ( ˆ V 1 + ˆ V 2 ) / 2, where ˆ V 1 = ( ˆ G ⊺ 1 ˆ Ω -1 1 ˆ G 1 ) -1 with (a similar expression holds for ˆ V 2 )

<!-- formula-not-decoded -->

In Online Appendix B.10, we show that ˆ V is consistent for V under our stated assumptions.

13 The intuitive reason for this is that cross-fitting ensures only the prediction properties of the nonparametric estimator are relevant.

4.2.2. On the relative efficiency of TD estimation. The TD estimator from (4.6) is robust to non-parametric estimation of transition densities. When the transition density has a parametric form, it is less efficient than the full-MLE estimator that jointly estimates the structural and transition density parameters. However full-MLE is seldom, if ever, used. Standard approaches such as NFXP and NPL are equivalent to partial-MLE, which employs a plug-in estimate of the transition density. For this reason, neither NFXP nor NLP are fully efficient: if we make the model for the transition density richer, while still keeping it parametric, the performance of NFXP and NLP will start to degrade and become worse than TD estimation. In the non-parametric regime, these methods lose √ n -consistency. On the other hand, when the transition density is fully known, NFXP and NPL are equivalent to full-MLE, and therefore more efficient than TD estimation. Between these two extremes, whether or not TD estimation is more efficient than NFXP will depend on the statistical complexity of the model used for the transition density.

An interesting open question is whether our estimator attains the semi-parametric efficiency bound when the transition density is unknown. The GMM formulation of the problem in (B.9)-(B.10) suggests this may be the case, but we leave this as a conjecture.

## 5. Estimation of dynamic discrete games

Our setup for dynamic games is based on Aguirregabiria and Mira (2010). We assume a single Markov-Perfect-Equilibrium setup where multiple players i = 1 , 2 , . . . , n play against each other in M different markets. Each player chooses among A mutually exclusive actions to maximize an infinite horizon objective. We observe the state of play for T time periods, where both T and the number of players n are fixed, while M →∞ . Utility of the players in any time period is affected by the actions of all the others, and a set of states x that are observed by all players. The per-period utility is denoted by z i ( a i , a -i , x ) ⊺ θ ∗ + e i for each player i , for some finite-dimensional parameter θ ∗ , where a i denotes player i 's action, a -i denotes the actions of all other players and e i is an idiosyncratic error term. As in Section 3, we take θ ∗ to be scalar to simplify the notation; all our results continue to hold for vector-valued θ ∗ , as long as each dimension is treated separately. Evolution of the states in the next period is determined by the transition density K ( x ′ | a, x ) where a := ( a 1 , . . . , a n ) denotes the actions of all the players. We denote by x tm the state at market m in time period t, by a tm the vector of actions by all players at time t in market m , and by a itm the action of player i at time t in market m .

We also let P i ( a i | x t ) denote the choice probability of player i taking action a i when the state is x t , and define e i ( a i , x ) := γ -ln P i ( a i | x ).

As in the single-agent case, the parameter θ ∗ can be obtained as solutions to the pseudo-log-likelihood function:

<!-- formula-not-decoded -->

where h i ( . ) and g i ( . ) are now player-specific, and given by

<!-- formula-not-decoded -->

In contrast to (2.2), the expectation averages over the actions of the other players as well.

Previous literature estimates θ ∗ using a two-step procedure: In the first step, the conditional choice probabilities P i ( a i | x t ) are calculated non-parametrically. These, along with estimates of K ( . ) are then used to recursively solve for h i ( . ) and g i ( . ) using equation (5.2). This step requires integrating over the actions of all the other players. Finally, given the estimated values of h i ( . ) and g i ( . ), the parameter θ ∗ is estimated through either pseudo-maximum-likelihood (PML; Aguirregabiria and Mira, 2007), minimum distance estimation (MDE; Pesendorfer and Schmidt-Dengler, 2008) or iterative versions of these (Bugni and Bunting, 2021). By contrast, our algorithm is a straightforward extension of those suggested in earlier sections for single-agent models. Let ˆ η i ( a i , x ) denote a nonparametric estimate of the choice probabilities for player i and denote e ( a i , x ; ˆ η i ) = γ -ln ˆ η i ( a i , x ). We apply our TD methods on the recursion (5.2), separately for each player. The linear semi-gradient estimates are given by ˆ h i ( a i , x ) = φ ( a i , x ) ⊺ ˆ ω i and ˆ g i ( a i , x ) = r ( a i , x ) ⊺ ˆ ξ i , where

<!-- formula-not-decoded -->

and for any function f ( · ), we define

<!-- formula-not-decoded -->

Similarly, the AVI iterations for h i ( · ) , g i ( · ) are given by

<!-- formula-not-decoded -->

If the players are symmetric ( z i ( a i , a -i , x ) does not depend on player i ) we can obtain computationally faster and more precise estimates by pooling across players.

Importantly, neither of the estimation strategies (5.3) nor (5.5) require partialling out other players' actions, leading to a tremendous reduction of computation. The nonparametric estimates ˆ h i , ˆ g i can be plugged into the PMLE criterion (5.1) to obtain an estimate for θ ∗ as

<!-- formula-not-decoded -->

It is straightforward to construct a locally robust estimator for θ ∗ in analogy with that for single-agent models. We describe this in Online Appendix B.7. The convergence properties of the locally robust estimators for games are also similar to those for singleagent models; a formal statement is provided in Online Appendix B.8.

The PMLE criterion (5.6) with plug-in estimates for h i ( . ) and g i ( . ) is not efficient even with discrete states, as discussed by Aguirregabiria and Mira (2007). However the values of h i ( . ) and g i ( . ) can be plugged into other, more efficient objectives, such as the MDE criterion with an efficient weighting matrix; Bugni and Bunting (2021) show that the latter is more efficient than even iterated PMLE estimation. With continuous states, however, one would need to employ locally robust corrections even for MDE to recover parametric rates of convergence for estimation of θ ∗ . The locally robust correction term can be constructed in a similar way as that for the PMLE criterion.

## 6. Simulations

In this section, we run two Monte Carlo simulations to test our methods, and compare them to alternative approaches. Our first simulation is based on the firm entry problem in Aguirregabiria and Magesan (2018). In the second set of Monte Carlo simulations, we

test our estimation method for dynamic discrete games. The latter simulations are based on the dynamic firm entry game used in Aguirregabiria and Mira (2007). 14

Online Appendices E.2 and E.3 report additional simulations based on the famous Rust (1987) bus engine replacement problem. Using this model, we provide results for a case with permanent unobserved heterogeneity and also compare our methods to the estimator proposed by Chernozhukov et al. (2018) for DDC models with finite dependence.

6.1. Firm entry problem. Consider the following dynamic firm entry problem described in Aguirregabiria and Magesan (2018). A firm decides whether to enter ( a t = 1) or not enter ( a t = 0) in a market for t = 1 , ..., T time periods. The payoff when entering is given by Π t = V P t -FC t -EC t + ε t , where V P t , FC t and EC t denote the firm's variable profit, fixed cost and entry cost, and ε t is a transitory shock that follows a logistic distribution. Variable profit is given by V P t = ( θ V P 0 + θ V P 1 z 1 t + θ V P 2 z 2 t ) exp( ω t ), where ω t denotes the firm's productivity shock, and z 1 t , z 2 t are exogenous state variables affecting the price-cost margin in the market. The fixed cost is given by FC t = θ FC 0 + θ FC 1 z 3 t , and the entry cost is given by EC t = ( θ EC 0 + θ EC 1 z 4 t )(1 -a t -1 ), where z 3 t , z 4 t are further exogenous state variables, and a t -1 denotes the entry decision in period t -1 which is an endogenous state variable. The payoff of not entering is normalized to zero. The parameters θ ∗ ≡ { θ V P 0 , θ V P 1 , θ V P 2 , θ FC 0 , θ FC 1 , θ EC 0 , θ EC 1 } are the structural parameters of interest. The exogenous state variables z jt , j ∈ { 1 , 2 , 3 , 4 } , and ω t are continuous and follow AR(1) processes, where z jt = γ j 0 + γ j 1 z jt -1 + e jt , and ω t = γ ω 0 + γ ω 1 ω t -1 + e ωt . The error terms e jt , e ωt follow normal N (0 , 1) distributions. The discount factor β is 0 . 95.

To carry out the simulations, we choose values for the structural parameters θ ∗ ( θ V P 0 = 0 . 5, θ V P 1 = 1 . 0, θ V P 2 = -1 . 0, θ FC 0 = 1 . 5, θ FC 1 = 1 . 0, θ EC 0 = 1 . 0, θ EC 1 = 1 . 0) and for the autoregressive processes of z jt and ω t ( γ j 0 = 0 . 0, γ j 1 = 0 . 6, γ ω 0 = 0 . 2, γ ω 1 = 0 . 6), and discretize the exogenous state variables to obtain a transition matrix with a 6-point support following Tauchen (1986). The resulting dimension of the state space is 2 × 6 5 = 15 , 552 . The discretization of the support is for simulations only; our methods treat these variables as continuous and do not require any prior knowledge of how they evolve (the knowledge of AR(1) dynamics is also not used). We iterate on the value function to obtain the vector of choice probabilities for each combination of the states, and use these to derive the ergodic, i.e., steady-state distribution of the state variables. Using this distribution, we generate data for 3000 firms, with T = 2 time periods.

14 R code for all simulations is made available as part of the replication package.

6.1.1. Simulation results - firm entry problem. Table 1 shows the results based on 1000 simulations using the linear semi-gradient and AVI methods. For the linear semi-gradient method, we parameterize h ( a, x ) and g ( a, x ) using a first order polynomial in the state variables. 15 For AVI, we approximate h ( a, x ) and g ( a, x ) using a Random Forest, and iterate the AVI procedure 20 times for each round of the simulations. For both the linear semi-gradient and the AVI methods, we estimate the choice probabilities η that enter e ( a ′ , x ′ ; η ) using a logit model where the explanatory variables are the state variables, their squares and interactions up to the second order.

We present results generated with and without the locally robust correction. For the results without correction, we obtain estimates for θ ∗ using (3.9). To generate the locally robust estimates, we use moment equation (B.12) for the linear semi-gradient method, and moment equation (4.6) for the AVI method where we employ a Random Forest to derive an estimate for the λ ( a, x, ˜ θ ) term contained in the locally robust moment. As before, the AVI method for estimation of λ ( · ) is iterated 20 times. We also use the sample splitting method described in Section 4.2.1 for the locally robust estimators, and we obtain the final ˆ θ as weighted average of the θ ∗ estimates from the two samples.

Both the linear semi-gradient and AVI estimates are closely centered around the true values, but the latter is clearly preferable in terms of mean squared error (MSE). While the locally robust estimator should in theory be preferable, we find that it produces results which are similar and if anything have slightly higher MSE than the non-robust versions. In fact, we find that there is very little bias to begin with, and the distribution of the estimates under the non-robust versions are already very close to normal, see Online Appendix E.1 for the plots of the finite sample distributions. The lower bias may be due to the specific nature of the example, which falls under a special class of DDC models called 'dynamic-logit models' (see Section 6.1.2). On the flip side, the locally robust methods are associated with higher variance due to cross-fitting. So, overall, there appears to be no gain from using the locally robust method in this example. Presumably, the variability of locally robust estimators can be lowered by using more folds in the cross-fitting procedure (we use two folds in all our examples); this would, however, come at the expense of slower computation times.

15 For the ω 's relating to parameters θ V P 0 , θ V P 1 , θ V P 2 , θ FC 0 , θ FC 1 , and for ξ , the terms include a constant, the exogenous state variables, the player's binary choice a t and the interactions of a t with all terms in the exogenous states. Given the set-up of the model, we also include the interactions z 1 t exp( ω t ) and z 2 t exp( ω t ) as state variables. In addition to the terms included above, the ω 's relating to parameters θ EC 0 and θ EC 1 also contain the terms (1 -a t -1 ) and (1 -a t -1 ) z 4 t , respectively. The total number of terms included is 16 (17 for θ EC 0 and θ EC 1 ).

6.1.2. Comparison with existing methods. Table 1 compares our methods to the two-step Euler Equation (EE) approach of Aguirregabiria and Magesan (2018). As given in Aguirregabiria and Magesan (2018, eq. 30), the EE estimator is not universally applicable; it can only be employed on the restricted class of 'dynamic-logit models' where the only endogenous variable is the past action and all the other variables are exogenous. 16 Our estimation strategy, unlike EE, does not exploit this special feature of the model (which is satisfied by the simulation study but is otherwise restrictive). Nevertheless, the linear semi-gradient method without locally robust corrections is three times faster than EE, albeit at the expense of a somewhat higher MSE.

On the other hand, the MSE of AVI is slightly lower than that of EE, but it is also much slower. However, we think raw computation times do not paint the full picture here. The reason AVI is slower is because we employ Random Forest (RF). The computational time can be made an order of magnitude smaller using other ML techniques such as series estimation, Ridge, LASSO or MARS, but that is not necessarily a reason to choose these over RF. An analogy can be drawn here with prediction: despite being slower, RF is often used in moderate and high-dimensional prediction problems as its predictive performance is superior, and more importantly, it is less sensitive to how the state variables are transformed. By contrast, both linear semi-gradient and EE approaches require choosing a specification; we need to choose the family of basis functions for the former, and the level and type of discretization for the latter. In practice, one typically runs specification checks to ensure robustness, but this takes up significantly more computational time, and in truly high-dimensional scenarios (e.g., when dim( x ) ∝ n ), finding the right specification (e.g., the level of discretization) may not even be feasible. A major advantage of RF, then, is that it does not require a specification and is also remarkably robust to tuning parameter choices (Hastie et al., 2009, p.590). In many practical applications, we think this advantage trumps the additional computational time that it involves.

Our locally robust corrections also make computations more time consuming, but are needed to achieve √ n -consistency under continuous states. The EE estimator is only √ n -consistent if the states are discrete, but for continuous states, discretization bias would imply a loss of √ n -consistency. A fair comparison of computational times would thus require comparing the locally robust estimator with a locally robust version of EE, but constructing the latter is beyond the scope of this paper.

16 While we presume that an EE estimator can also be derived for more general models, the construction and computation of the EE mapping with endogenous state variables is much more involved.

Table 1. Simulations: Firm entry problem

|                         |         | Linear semi-gradient   | Linear semi-gradient   | Linear semi-gradient   | Linear semi-gradient   | AVI                | AVI                | AVI              | AVI            | 2-step EE        | 2-step EE   |
|-------------------------|---------|------------------------|------------------------|------------------------|------------------------|--------------------|--------------------|------------------|----------------|------------------|-------------|
|                         |         | not locally robust     | not locally robust     | locally robust         | locally robust         | not locally robust | not locally robust | locally robust   | locally robust |                  |             |
|                         | DGP (1) | TDL (2)                | MSE (3)                | TDL (4)                | MSE (5)                | TDL (6)            | MSE (7)            | TDL (8)          | MSE (9)        | EE (10)          | MSE (11)    |
| θ V P 0                 | 0.5     | 0.5028 (0.0760)        | 0.0058                 | 0.5087 (0.0821)        | 0.0068                 | 0.4877 (0.0582)    | 0.0035             | 0.4844 (0.0711)  | 0.0053         | 0.5052 (0.0555)  | 0.0031      |
| θ V P 1                 | 1.0     | 0.9831 (0.0689)        | 0.0050                 | 1.0049 (0.0762)        | 0.0058                 | 1.0045 (0.0581)    | 0.0034             | 1.0118 (0.0704)  | 0.0051         | 1.0105 (0.0572)  | 0.0034      |
| θ V P 2                 | -1.0    | -0.9839 (0.0725)       | 0.0055                 | -1.0061 (0.0805)       | 0.0065                 | -1.0059 (0.0602)   | 0.0037             | -1.0136 (0.0719) | 0.0053         | -1.0119 (0.0574) | 0.0034      |
| θ FC 0                  | 1.5     | 1.5066 (0.1482)        | 0.0220                 | 1.5254 (0.1583)        | 0.0257                 | 1.5379 (0.1231)    | 0.0166             | 1.5433 (0.1460)  | 0.0232         | 1.5136 (0.1218)  | 0.0150      |
| θ FC 1                  | 1.0     | 0.9746 (0.1228)        | 0.0157                 | 0.9916 (0.1342)        | 0.0180                 | 1.0090 (0.1001)    | 0.0101             | 1.0041 (0.1206)  | 0.0145         | 1.0044 (0.0939)  | 0.0088      |
| θ EC 0                  | 1.0     | 0.9973 (0.1003)        | 0.0101                 | 1.0132 (0.1076)        | 0.0117                 | 0.9864 (0.1007)    | 0.0103             | 0.9982 (0.1203)  | 0.0145         | 1.0030 (0.1018)  | 0.0104      |
| θ EC 1                  | 1.0     | 0.9948 (0.1613)        | 0.0260                 | 1.0163 (0.1735)        | 0.0303                 | 0.9645 (0.1365)    | 0.0199             | 1.0082 (0.1705)  | 0.0291         | 0.9081 (0.1248)  | 0.0240      |
| Total MSE               |         |                        | 0.0901                 |                        | 0.1050                 |                    | 0.0674             |                  | 0.0970         |                  | 0.0681      |
| Time per round (in sec) |         | 0.33                   |                        | 3.70                   |                        | 44.69              |                    | 76.55            |                | 0.99             |             |

Notes: The table reports results based on 1000 simulations with 3000 firms. Column (1) shows the true parameter values in the model. Columns (2), (4), (6), (8), (10) report the empirical mean and standard deviation (in parentheses) for the estimated parameters for each of the estimation methods. Columns (3), (5), (7), (9), (11) report the mean squared errors.

For a second comparison, we compare our estimators to a standard CCP estimator where the state variables are discretized and the transition and choice probabilities are estimated using cell values. We discretize the state space by creating dummy variables for each state variable z 1 t , z 2 t , z 3 t , z 4 t and exp( ω t ) based on whether they are above or below their median. However, even this results in empty cell values, so the state space needs to be restricted further. A common approach is to use K-means clustering, but this is not appropriate in the current setting where the state variables are independent by construction. We therefore restrict the state space grid by combining variables z 1 t and z 2 t into a binary variable taking value one whenever both individual dummies take value one. The resulting state space consists of four binary variables, implying 16 cells in the exogenous state space grid. We tried alternative feasible ways of discretizing the state space, but found that these do not lead to improvements over the chosen method. We run 1000 simulations, and the results are shown in Table 2.

Compared to the results from Table 1, the discretized CCP estimator leads to substantially larger bias in some of the estimated parameters. Column (4) shows that the corresponding MSEs are large and generally exceed those obtained using our estimators. This is particularly true for parameters θ V P 1 and θ V P 2 . Overall, the toal MSE increases more than 10-fold from 0 . 067 -0 . 105 across all parameters in Table 1 to 1 . 109 in Table 2. At the same time, our linear semi-gradient method is even three times faster computationally than discretization; this may be related to matrix inversion being more ill-conditioned under discretization.

6.2. Firm entry game. Consider the following firm market entry game, which is similar to that described in Aguirregabiria and Mira (2007). There are i = 1 , ..., 5 firms (players), and we observe their decision to enter ( a itm = 1) or not enter ( a itm = 0) in m = 1 , ..., M different markets for t = 1 , ..., T time periods. Denote a firm's action by j ∈ { 1 , 0 } . The payoff of each firm i is affected by the decision of all the other firms whether to enter, as well as firm i 's previous-period entry decision. Current-period profits when entering are given by

/negationslash

<!-- formula-not-decoded -->

where ln( S tm ) is a measure of consumer market size of market m in period t , and ε itm is a transitory shock that follows a logistic distribution. We assume that ln( S tm ) is continuous and follows an AR(1) process, where the parameters are the same across

Table 2. Simulations: Firm entry problem - Comparison with standard CCP

|                                      | DGP (1)   | TDL (2)          | bias (3)   | MSE (4)   |
|--------------------------------------|-----------|------------------|------------|-----------|
| CCP with discretized state variables |           |                  |            |           |
| θ V P 0                              | 0.5       | 0.1391 (0.2266)  | -0.3609    | 0.1815    |
| θ V P 1                              | 1.0       | 0.7968 (0.5535)  | -0.2032    | 0.3474    |
| θ V P 2                              | -1.0      | -0.4017 (0.2396) | 0.5983     | 0.4154    |
| θ FC 0                               | 1.5       | 1.3799 (0.1300)  | -0.1201    | 0.0313    |
| θ FC 1                               | 1.0       | 0.8655 (0.1392)  | -0.1345    | 0.0374    |
| θ EC 0                               | 1.0       | 0.7859 (0.0891)  | -0.2141    | 0.0538    |
| θ EC 1                               | 1.0       | 0.9011 (0.1809)  | -0.0989    | 0.0425    |
| Total MSE Time per round (in sec)    |           | 0.94             |            | 1.1093    |

Notes: The table reports results based on 1000 simulations with 3000 firms. Column (1) shows the true parameter values in the model. Column (2) reports the empirical mean and standard deviation (in parentheses) for the estimated parameters. Column (3) reports the average bias in the estimated parameters. The mean squared errors are reported in column (4).

markets: ln( S tm ) = α + λ ln( S ( t -1) m ) + u tm . The error term u tm is assumed to follow a normal N (0 , 1) distribution. The profit of not entering is normalized to zero, and the discount factor β is 0 . 95. The parameters θ ∗ ≡ { θ RS , θ RN , θ FC , θ EC } are the structural parameters of interest. The state variables in this setting are given by the current market demand variable S tm , as well as the vector of all firms' previous entry decisions a ( t -1) m = { a i ( t -1) m : i = 1 , ..., 5 } .

To carry out the simulations, we choose values for the structural parameters θ ∗ ( θ RS = 1 , θ RN = 1 , θ FC = 1 . 7 , θ EC = 1), and for the autoregressive process for log market size ( α = 1 . 5, λ = 0 . 5). Wediscretize ln( S tm ) and obtain a transition matrix for the discretized variable with a 10-point support following the method by Tauchen (1986). As in the Monte Carlo experiments for the firm entry problem in Section 6.1, the discretization is for simulations of the data only and we treat the state variables as continuous in our

estimations. We then solve for the Markov-Perfect-Equilibrium of the game. 17 Using the equilibrium (i.e., ergodic) distribution, we generate data for 1000 and for 3000 markets, with T = 2 time periods.

6.2.1. Simulation results - firm entry game. We present the results of 1000 simulations based on the linear semi-gradient method, without employing the locally robust correction. Each round of the simulations begins by generating new data, where the first-period state variables are drawn from the steady-state distribution. In order to assess the sensitivity of our algorithm to different specifications for the basis functions, we parameterize h ( a, x ) and g ( a, x ) using different sets of polynomials in the state variables. In particular, we show results where h ( a, x ) and g ( a, x ) are approximated using a second, third or fourth order polynomial. 18 For all simulations, the choice probabilities η that enter e ( a ′ , x ′ ; η ) are estimated using individual logit models for each firm, where we use a third order polynomial in the state variables as explanatory variables. We then estimate the parameters ω and ξ using equation (5.3). 19 Finally, we obtain estimates for the θ ∗ parameters as the solutions to the pseudo-log-likelihood function (5.1).

The results are shown in Table 3. Panels A, B and C present simulations for the same dataset using different basis functions to parameterize the value function terms h ( a, x ) and g ( a, x ). Column (2) shows that even with 1000 markets our algorithm produces parameter estimates that are closely centered around the true values. The results are generally similar across Panels A to C, although the bias and MSE tends to be lowest for the second order polynomial, and highest for the fourth order polynomial. This is especially the case for the parameter on the number of market entrants, θ RN . To assess these differences formally for the case with 1000 markets, we use the cross-validation procedure described in Section 3.3. The procedure is applied to ten random samples of market size 1000, and we find that the TD error criterion consistently selects the second order polynomial as the optimal set of basis functions. Thus, the proposed crossvalidation method provides useful guidance for choosing the number of basis functions.

17 This is done by finding the firms' conditional value functions ν j ( S tm , a ( t -1) m ) for each of the 2 5 × 10 = 320 possible combinations of the state variables through repeated iteration, and using these to derive the equilibrium choice probabilities p ( S tm , a ( t -1) m ). Based on the equilibrium probabilities, we compute the equilibrium distribution of state variables.

/negationslash

18 For the ω 's relating to parameters θ RS , θ RN , θ FC and for ξ , the terms include a constant, terms up to the second/third/fourth order in the state variables ln( S tm ) and ln(1 + ∑ j = i a j ( t -1) m ), the firm's binary choice a itm and the interactions of a itm with all terms in the state variables. The total number of terms is 12 / 20 / 30. In addition to these terms, the ω ′ s relating to parameter θ EC also contain the term

(1 -a i ( t -1) m ) a itm .

19 Given the symmetric set-up of the game, we pool the data across players in this application.

In a similar version of the firm entry game, Aguirregabiria and Mira (2007) use the NPL algorithm and derive results comparable to ours. Note, however, that for a direct comparison of our results with those obtained using the NPL algorithm, one would need to obtain a non-parametric estimate of the transition density when implementing the latter which is not trivial in practice.

As expected, columns (4) and (5) show that increasing the market size generally reduces the small sample bias in the estimated parameters, and leads to a fall in the empirical standard deviations. In addition to being smaller, the MSE across Panels A to C is also more similar across the three sets of basis functions. As before, we employ the cross-validation method described above to compare these specifications more formally for the case with 3000 markets. In line with the estimation results, we find that all three polynomials now produce very similar sets of mean squared TD errors, even though the second order polynomial continues to be the one that is selected by the criterion. 20 While we view this as further evidence that the proposed cross-validation method can provide useful guidance to choose a suitable set of basis functions, more importantly, the small differences in the results across panels A to C also suggest that our methods prove fairly robust to this choice in practice.

## 7. Conclusions

We propose two new estimators for DDC models which overcome previous computational and statistical limitations by combining traditional CCP estimation approaches with the idea of TD learning from the RL literature. The first approach, linear semigradient, makes use of simple matrix inversion techniques, is computationally very cheap and therefore fast. The second approach, Approximate Value Iteration, can be easily combined with any ML method devised for prediction. Unlike previous estimation methods, our methods are able to easily handle continuous and/or high-dimensional state spaces in settings where a finite dependence property does not hold. This is of particular importance for the estimation of dynamic discrete games. We also propose a locally robust estimator to account for the non-parametric estimation in the first stage. We prove the statistical properties of our estimator and show that it is consistent and converges at parametric rates. A range of Monte Carlo simulations using a dynamic firm entry problem, a dynamic firm entry game and two versions of the famous Rust (1987) engine replacement problem show that the proposed algorithms work well in practice.

20 As before, we compute the TD criterion for ten random samples.

Table 3. Simulations: Firm entry game - Linear semi-gradient

|                         | DGP (1)   | TDL (2)         | MSE (3)      | TDL (4)         | MSE (5)      |
|-------------------------|-----------|-----------------|--------------|-----------------|--------------|
| A. 2nd order polynomial |           | 1000            | markets      | 3000            | markets      |
| θ RS (market size)      | 1.0       | 0.9715 (0.1601) | 0.0264       | 0.9845 (0.0897) | 0.0083       |
| θ RN (n. of entrants)   | 1.0       | 0.8956 (0.5309) | 0.2924       | 0.9581 (0.2904) | 0.0860       |
| θ FC (fixed cost)       | 1.7       | 1.7221 (0.2938) | 0.0867       | 1.6919 (0.1617) | 0.0262       |
| θ EC (entry cost)       | 1.0       | 1.0189 (0.0621) | 0.0042       | 1.0225 (0.0353) | 0.0018       |
| Total MSE               |           |                 | 0.4098       |                 | 0.1222       |
| Time per round (in sec) |           | 0.32            |              | 0.93            |              |
| B. 3rd order polynomial |           | 1000 markets    | 1000 markets | 3000 markets    | 3000 markets |
| θ RS (market size)      | 1.0       | 0.9145 (0.1470) | 0.0289       | 0.9647 (0.0869) | 0.0088       |
| θ RN (n. of entrants)   | 1.0       | 0.6898 (0.4800) | 0.3264       | 0.8857 (0.2797) | 0.0912       |
| θ FC (fixed cost)       | 1.7       | 1.7811 (0.2718) | 0.0804       | 1.7134 (0.1573) | 0.0249       |
| θ EC (entry cost)       | 1.0       | 1.0172 (0.0622) | 0.0042       | 1.0219 (0.0353) | 0.0017       |
| Total MSE               |           |                 | 0.4398       |                 | 0.1266       |
| Time per round (in sec) |           | 0.50            |              | 1.54            |              |
| C. 4th order polynomial |           | 1000 markets    | 1000 markets | 3000 markets    | 3000 markets |
| θ RS (market size)      | 1.0       | 0.8638 (0.1321) | 0.0360       | 0.9455 (0.0846) | 0.0101       |
| θ RN (n. of entrants)   | 1.0       | 0.5067 (0.4231) | 0.4222       | 0.8163 (0.2707) | 0.1070       |
| θ FC (fixed cost)       | 1.7       | 1.8335 (0.2510) | 0.0808       | 1.7337 (0.1530) | 0.0245       |
| θ EC (entry cost)       | 1.0       | 1.0158 (0.0620) | 0.0041       | 1.0212 (0.0352) | 0.0017       |
| Total MSE               |           |                 | 0.5431       |                 | 0.1433       |
| Time per round (in sec) |           | 0.84            |              | 2.64            |              |

Notes: The table reports results for 1000 simulations. Panels A, B and C use different sets of basis functions to parameterize h ( a, x ) and g ( a, x ). Column (1) shows the true parameter values in the model. Columns (2) and (4) report the empirical mean and standard deviation (in parentheses) for the estimated parameters, based on a sample of 1000 and 3000 markets, respectively. The mean squared errors are reported in columns (3) and (5). All results are based on the estimation method without correction function.

Data availability statement. The data and code underlying this article are available in Zenodo, at https://doi.org/10.5281/zenodo.16184776.

## References

- Ackerberg, D., X. Chen, J. Hahn, and Z. Liao (2014): 'Asymptotic Efficiency of Semiparametric Two-Step GMM,' Review of Economic Studies , 81, 919-943.
- Aguirregabiria, V., and A. Magesan (2018): 'Solution and Estimation of Dynamic Discrete Choice Structural Models Using Euler Equations,' Working paper .
- Aguirregabiria, V., and P. Mira (2002): 'Swapping the Nested Fixed Point Algorithm: A Class of Estimators for Discrete Markov Decision Models,' Econometrica , 70, 1519-1543.
- (2007): 'Sequential Estimation of Dynamic Discrete Games,' Econometrica , 75, 1-53.
- (2010): 'Dynamic Discrete Choice Structural Models: A Survey,' Journal of Econometrics , 156, 38-67.
- Almagro, M., and T. Domínguez-Iino (2025): 'Location Sorting and Endogenous Amenities: Evidence from Amsterdam,' Econometrica , 93, 1031-1071.
- Arcidiacono, P., P. Bayer, F. A. Bugni, and J. James (2013): 'Approximating High-Dimensional Dynamic Models: Sieve Value Function Iteration,' in Structural Econometric Models : Emerald Group Publishing Limited, 45-95.
- Arcidiacono, P., and J. B. Jones (2003): 'Finite Mixture Distributions, Sequential Likelihood and the EM Algorithm,' Econometrica , 71, 933-946.
- Arcidiacono, P., and R. A. Miller (2011): 'Conditional Choice Probability Estimation of Dynamic Discrete Choice Models with Unobserved Heterogeneity,' Econometrica , 79, 1823-1867.
- Bajari, P., C. L. Bankard, and J. Levin (2007): 'Estimating Dynamic Models of Imperfect Competition,' Econometrica , 75, 1331-1370.
- Barwick, P. J., and P. A. Pathak (2015): 'The Costs of Free Entry: An Empirical Study of Real Estate Agents in Greater Boston,' The RAND Journal of Economics , 46, 103-145.
- Benítez-Silva, H., G. Hall, G. J. Hitsch, G. Pauletto, and J. Rust (2000): 'A Comparison of Discrete and Parametric Approximation Methods for Continuous-State Dynamic Programming Problems,' Working paper .

- Biau, G. (2012): 'Analysis of a Random Forests Model,' Journal of Machine Learning Research , 13, 1063-1095.
- Buchholz, N., M. Shum, and H. Xu (2021): 'Semiparametric Estimation of Dynamic Discrete Choice Models,' Journal of Econometrics , 223, 312-327.
- Bugni, F. A., and J. Bunting (2021): 'On the Iterated Estimation of Dynamic Discrete Choice Games,' Review of Economic Studies , 88, 1031-1073.
- Chen, X., and Z. Qi (2022): 'On Well-Posedness and Minimax Optimal Rates of Nonparametric Q-function Estimation in Off-Policy Evaluation,' in International Conference on Machine Learning , 3558-3582, PMLR.
- Chernozhukov, V., J. C. Escanciano, H. Ichimura, W. K. Newey, and J. M. Robins (2022): 'Locally Robust Semiparametric Estimation,' Econometrica , 90, 1501-1535.
- Chernozhukov, V., W. K. Newey, and V. Semenova (2019): 'Welfare Analysis in Dynamic Models,' arXiv preprint arXiv:1908.09173 .
- Chernozhukov, V., W. K. Newey, and R. Singh (2018): 'Learning L2Continuous Regression Functionals via Regularized Riesz Representers,' arXiv preprint arXiv:1809.05224 .
- Farrell, M. H., T. Liang, and S. Misra (2021): 'Deep Neural Networks for Estimation and Inference,' Econometrica , 89, 181-213.
- Hastie, T., R. Tibshirani, and J. Friedman (2009): The Elements of Statistical Learning: Data Mining, Inference, and Prediction : Springer, 2nd edition.
- Hotz, V. J., and R. A. Miller (1993): 'Conditional Choice Probabilities and the Estimation of Dynamic Models,' Review of Economic Studies , 60, 497-529.
- Hotz, V. J., R. A. Miller, S. Sanders, and J. Smith (1994): 'A Simulation Estimator for Dynamic Models of Discrete Choice,' Review of Economic Studies , 61, 265-289.
- Ichimura, H., and W. K. Newey (2022): 'The Influence Function of Semiparametric Estimators,' Quantitative Economics , 13, 29-61.
- Kalouptsidi, M. (2014): 'Time to Build and Fluctuations in Bulk Shipping,' American Economic Review , 104, 564-608.
- (2018): 'Detection and Impact of Industrial Subsidies: The Case of Chinese Shipbuilding,' Review of Economic Studies , 85, 1111-1158.

- Keane, M. P., and K. I. Wolpin (1994): 'The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence,' The Review of Economics and Statistics , 648-672.
- Lange, S., T. Gabel, and M. Riedmiller (2012): 'Batch Reinforcement Learning,' in Reinforcement Learning : Springer, 45-73.
- Munos, R., and C. Szepesvári (2008): 'Finite-Time Bounds for Fitted Value Iteration,' Journal of Machine Learning Research , 9, 815-857.
- Newey, W. K. (1997): 'Convergence Rates and Asymptotic Normality for Series Estimators,' Journal of Econometrics , 79, 147-168.
- Newey, W. K., and D. McFadden (1994): 'Large Sample Estimation and Hypothesis Testing,' in Handbook of Econometrics Volume 4: Elsevier, 2111-2245.
- Norets, A. (2012): 'Estimation of Dynamic Discrete Choice Models using Artificial Neural Network Approximations,' Econometric Reviews , 31, 84-106.
- Pesendorfer, M., and P. Schmidt-Dengler (2008): 'Asymptotic Least Squares Estimators for Dynamic Games,' Review of Economic Studies , 75, 901-928.
- Rust, J. (1987): 'Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher,' Econometrica , 55, 999-1033.
- Semenova, V. (2018): 'Machine Learning for Dynamic Models of Imperfect Information and Semiparametric Moment Inequalities,' arXiv preprint arXiv:1808.02569 .
- Sutton, R. S. (1988): 'Learning to Predict by the Methods of Temporal Differences,' Machine Learning , 3, 9-44.
- Sutton, R. S., and A. G. Barto (2018): Reinforcement Learning: An Introduction : MIT Press, Cambridge, MA, 2nd edition.
- Tauchen, G. (1986): 'Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions,' Economics Letters , 20, 177-181.
- Tsitsiklis, J. N., and B. Van Roy (1997): 'An Analysis of Temporal-Difference Learning with Function Approximation,' IEEE Transactions on Automatic Control , 42, 674-690.

## Appendix A. Proofs of main results

For the proofs of Theorems 1-2, we work within a more general setting than in the main text, by letting the distribution of ( a it , x it ) be time-varying. Let P t denote the population distribution of ( a, x ) at time t . Also, let P denote the probability distribution of the

process { ( a 1 , x 1 ) , . . . , ( a T , x T ) } . Note that P ≡ P 1 × · · · × P T . Denote the expectation over P by E [ · ]. We use the o p ( · ) and O p ( · ) notations to denote convergence in probability, and bounded in probability, respectively, under the probability distribution P .

We also need to extend the definitions of P and E [ · ]: Let P denote the relative frequency of occurrence of ( a, x, a ′ , x ′ ) in the data as n →∞ , and E [ · ] the corresponding expectation over P . Note that P is different from P as the latter is the distribution of ( a, x, a, x ′ ) after dropping the time index. However, the two are related as for any function f , we have E [ f ( a, x, a ′ , x ′ )] = ( T -1) -1 ∑ T -1 t =1 E [ f ( a it , x it , a it +1 , x it +1 )]. These updated definitions of P and E [ · ] are applicable whenever we use these notations in the main text.

Note that due to the Markov process assumption, the conditional distribution P ( a t +1 , x t +1 | a t , x t ) is always independent of t (indeed, one could always include t in x ). Hence, P ( a ′ , x ′ | a, x ) ≡ P ( a t +1 , x t +1 | a t , x t ) and E [ f ( a ′ , x ′ ) | a, x ] ≡ E [ f ( a t +1 , x t +1 ) | a t , x t ] for all t . Also note that time stationarity of ( a it , x it ), if it holds, implies P t ≡ P and E t [ · ] ≡ E [ · ] for all t .

A.1. Proof of Theorem 1. Lemma 1 in Online Appendix B.2 implies ω ∗ exists. To prove that ˆ ω exists, it suffices to show that ˆ A := E n [ φ ( βφ ′ -φ ) ⊺ ] is invertible with probability approaching one. Recall that using our notation, ˆ A = ( n ( T -1)) -1 ∑ i ∑ T -1 t =1 φ it ( βφ it +1 -φ it ) ⊺ , while A = ( T -1) -1 ∑ T -1 t =1 E [ φ it ( βφ it +1 -φ it ) ⊺ . We can thus write ∣ ∣ ∣ ˆ A -A ∣ ∣ ∣ ≤ ( T -1) -1 ∑ T -1 t =1 ∣ ∣ ∣ ˆ A t -A t ∣ ∣ ∣ , where ˆ A t := n -1 ∑ i φ it ( βφ it +1 -φ it ) ⊺ and A t := E [ φ it ( βφ it +1 -φ it ) ⊺ ]. By Assumption 1(ii), | φ ( a, x ) | ∞ ≤ M independent of k φ , so

<!-- formula-not-decoded -->

This proves ∣ ∣ ∣ ˆ A t -A t ∣ ∣ ∣ = O p ( k φ / √ n ). But T is fixed, which implies that ∣ ∣ ∣ ˆ A -A ∣ ∣ ∣ = O p ( k φ / √ n ) as well. We thus obtain ¯ λ ( ˆ A ) ≤ ¯ λ ( A ) + ∣ ∣ ∣ ˆ A -A ∣ ∣ ∣ ≤ ¯ λ ( A ) + o p (1). Since ¯ λ ( A ) &lt; 0, this proves that ¯ λ ( ˆ A ) &lt; 0 with probability approaching one, and subsequently, that ˆ A is invertible. This completes the proof of the first claim.

The second claim follows from Lemma 3 in Online Appendix B.2 and Assumption 1(iii). To prove the last claim, we first show that with probability approaching one,

<!-- formula-not-decoded -->

for some C &lt; ∞ . Define b = E [ φz ] and ˆ b = E n [ φz ]. We then have Aω ∗ = b and ˆ A ˆ ω = ˆ b . We can combine the two equations to get

<!-- formula-not-decoded -->

The above implies

<!-- formula-not-decoded -->

We earlier showed ∣ ∣ ∣ ˆ A -A ∣ ∣ ∣ = O p ( k φ / √ n ). Hence, λ ( -ˆ A ) ≥ λ ( -A ) + o p (1), so

<!-- formula-not-decoded -->

with probability approaching one, for any constant c ∈ (0 , 1). Given (A.2) and (A.3),

<!-- formula-not-decoded -->

with probability approaching one.

It remains to bound ∣ ∣ ∣ ˆ b -b ∣ ∣ ∣ and ∣ ∣ ∣ ˆ Aω ∗ -Aω ∗ ∣ ∣ ∣ . As before, we can define ˆ b t = n -1 ∑ i φ it z it and b t = E [ φ it z it ] to obtain

<!-- formula-not-decoded -->

This proves

<!-- formula-not-decoded -->

In a similar vein,

<!-- formula-not-decoded -->

as long as E [ | φ ( βφ -φ ) ⊺ ω ∗ | 2 ] = O ( k φ ) . The latter holds assuming 1(ii)-(iv) since

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality uses ‖ φ ⊺ ω ∗ -h ‖ 2 = O ( k -α φ ) (as shown above), and | h ( · , · ) | ∞ ≤ (1 -β ) -1 | z ( · , · ) | ∞ &lt; (1 -β ) -1 L (which can be easily verified using (2.2) and Assumption

1(iv)). Combining the above, there exists C &lt; ∞ such that | ˆ ω -ω ∗ | ≤ C √ k φ /n, with probability approaching one. We have thus shown (A.1). Now,

<!-- formula-not-decoded -->

where the final inequality follows from the second claim of this theorem and (A.1). The last claim then follows from the above along with the fact that, by Assumption 1(iv), ¯ λ ( E [ φφ ⊺ ]) ≤ ‖ φ ‖ 2 2 ≤ M 2 k φ .

A.2. Proof of Theorem 2. The first two claims follow from steps analogous to those in Theorem 1. We thus need to show that with probability approaching one,

<!-- formula-not-decoded -->

for some C &lt; ∞ . The third claim is a straightforward consequence of this.

Recall that we use cross-fitting to estimate ξ ∗ . Let n 1 , n 2 denote the sample sizes, and ˆ η 1 , ˆ ξ 1 and ˆ η 2 , ˆ ξ 2 the estimates of η and ξ ∗ in the two folds. We shall show that | ˆ ξ 1 -ξ ∗ | = O p ( √ k r /n ) (and similarly | ˆ ξ 2 -ξ ∗ | = O p ( √ k r /n )), and therefore | ˆ ξ -ξ ∗ | = O p ( √ k r /n ). To this end, let A r := E [ rr ⊺ ], b r := E [ r ( a, x ) e ( a ′ , x ′ ; η )], ˆ A (1) r := E (1) n [ rr ⊺ ] and ˆ b (1) r := E (1) n [ r ( a, x ) e ( a ′ , x ′ ; ˆ η 2 )], where E (1) n [ · ] denotes the empirical expectation using only the first fold. Let ς ( a, x, a ′ , x ′ ; η ) := r ( a, x ) e ( a ′ , x ′ ; η ) and ς it ( η ) := r ( a it , x it ) e ( a it +1 , x it +1 ; η ).

Based on the above definitions, we have ˆ A (1) r ˆ ξ 1 = ˆ b (1) r , and A r ξ ∗ = b r . Comparing with the proof of Theorem 1, the only difference is in the treatment of | ˆ b (1) r -b r | . As before, define ˆ b (1) rt := n -1 ∑ i ς it (ˆ η 2 ) and b rt := E [ ς it ( η )]. We then have | ˆ b (1) r -b r | = ( T -1) -1 ∑ T -1 t =1 | ˆ b (1) rt -b rt | . Since T is finite, it suffices to bound | ˆ b (1) rt -b rt | for some arbitrary t . Now, by similar arguments as in the proof of Theorem 1, we have

<!-- formula-not-decoded -->

Hence (A.4) follows once we show

<!-- formula-not-decoded -->

We now prove (A.5). Denoting the set of observations in the second fold by N 2 :

<!-- formula-not-decoded -->

First consider the term R 1 nt . Define

<!-- formula-not-decoded -->

Clearly, E [ δ it |N 2 ] = 0. We then have

<!-- formula-not-decoded -->

Now for any ( a, x, a ′ , x ′ ), from the definition of ς ( · ), with probability approaching one,

<!-- formula-not-decoded -->

where the second inequality follows from Assumption 2(ii), and the third follows from 2(v). 21 In view of (A.6) and (A.7), there exists C &lt; ∞ such that

<!-- formula-not-decoded -->

where the last equality follows by Assumption 2(v). This proves

<!-- formula-not-decoded -->

Next consider the term R 2 nt . Note that E [ ς it ( η )] is twice Fréchet differentiable. Indeed, in the main text we have shown that ∂ η E [ ς it ( η )] = 0, where ∂ η · denotes the Fréchet differential of E [ ς it ( η )] (cf. equation (3.7)). Furthermore, ln η is an infinitely differentiable function of η ( a, x ) with second derivatives bounded by δ -2 (since 1 &gt; η ( a, x ) , ˆ η ( a, x ) &gt; δ under Assumption 2(vi)). This implies ς ( a, x, a ′ , x ′ ; η ) is also infinitely differentiable with respect to η ( a, x ), with second derivatives bounded by δ -2 | r ( a, x ) | &gt; δ -2 √ k r , where the ' ≲ ' is due to Assumption 1(ii) which implies | r ( · ) | ∞ is bounded. The above facts imply,

21 In particular, we have used the fact ˆ η 2 &gt; δ + o p (1) which follows from η &gt; δ and | ˆ η 2 -η | = o p (1).

through a second order Taylor expansion, that

<!-- formula-not-decoded -->

for some C 1 &lt; ∞ . Hence,

<!-- formula-not-decoded -->

(A.8) and (A.9) imply (A.5), leading to the desired claim (A.4).

## A.3. Proof of Theorem 5. Define

<!-- formula-not-decoded -->

Denote by ˜ θ ( l ) the preliminary estimator of θ ∗ from the l -th data fold under cross-fitting.

By similar arguments as in Newey and McFadden (1994), ˜ θ ( l ) is consistent for θ ∗ under Assumptions 1-5. We now prove the stronger statement that

<!-- formula-not-decoded -->

Without loss of generality, take l = 1. By a first order Taylor expansion using the definition of ˜ θ (1) ,

<!-- formula-not-decoded -->

for some ˘ θ (1) such that ∥ ∥ ∥ ˘ θ (1) -θ ∗ ∥ ∥ ∥ ≤ ∥ ∥ ∥ ˜ θ (1) -θ ∗ ∥ ∥ ∥ . Now, E (1) n [ ∇ θ m ( a, x ; ˘ θ (1) , ˆ h (1) , ˆ g (1) ) ] -1 = O p (1) by Assumption 5(ii). Furthermore,

<!-- formula-not-decoded -->

where the second equality follows from Chebyshev's inequality as m ( a, x ; θ ∗ , h, g ) is uniformly bounded under the stated assumptions (continuity of h, g ; compactness of the support X of x ). It remains to bound R 1 . To this end, we use (B.5) in Online Appendix B.4. Note that the Riesz representers ψ h ( · , · ; θ ∗ , h, g ) , ψ g ( · , · ; θ ∗ , h, g ) (defined in B.4.2) are uniformly bounded under compactness of X and smoothness of h . It therefore follows that

<!-- formula-not-decoded -->

where the last equality uses Assumption 5(iii).

To complete the proof of the theorem, it suffices to verify Assumptions 1-3 and 5 in Chernozhukov et al. (2022).

Assumption 1 of Chernozhukov et al. (2022) requires

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first note that the smoothness conditions on h, g, η (imposed in Assumptions 1-3) along with the compactness of the domain X , ensure z ( · ) , e ( · , · ; η ) , h ( · ) , g ( · ) , m ( a, x ; θ ∗ , h, g ) and ψ ( a, x ; θ ∗ , h, g ) are all uniformly bounded in ( a, x ). By standard dynamic programming arguments, they also imply λ ( a, x ; θ ∗ ) is uniformly bounded in ( a, x ). 22 These bounds lead to ‖ ζ ( · , · ; θ ∗ , h, g, η, λ, θ ∗ ) ‖ ∞ ≤ M &lt; ∞ . This shows that the first requirement (A.12) holds. Next, we show (A.13): by the form of m ( a, x ; θ, h, g ) and the fact that

<!-- formula-not-decoded -->

for all ˘ a, x , some straightforward algebra gives

<!-- formula-not-decoded -->

Observe that | h (˘ a, · ) | ∞ &lt; ∞ for each ˘ a by the assumptions of compact support for x and continuity of h . We thus obtain

<!-- formula-not-decoded -->

22 Note that | λ ( · , · ; θ ∗ ) | ∞ ≤ (1 -β ) -1 | ψ ( · , · ; θ ∗ , h, g ) | ∞ by the fixed point definition of λ ( · ).

where the second inequality employs π ( a, x ; θ ∗ , h, g ) ≡ η ( a, x ) &gt; δ &gt; 0, as stated in Assumption 5(iii), and the last equality also follows from the same assumption. This proves (A.13). The third requirement (A.14) follows from the boundedness of λ ( · , · ; θ ∗ ) together with Assumption 5(iii). Finally, (A.15) follows from the boundedness of z ( · ) , h, g, ln η -1 (all of which hold under Assumption 5(iii)), along with the consistency of ˜ θ and Assumption 5(iv).

Assumption 2 of Chernozhukov et al. (2022) requires

<!-- formula-not-decoded -->

To this end, observe that

<!-- formula-not-decoded -->

where ˆ ∆ h := ˆ h -h , ˆ ∆ g := ˆ g -g and ˆ ∆ ln η := ln ˆ η -ln η . In view of the n -1 / 4 consistency of ˜ θ along with Assumptions 5(iii)-(iv), straightforward algebra shows E [∣ ∣ ∣ ˜ ∆( a, x ) ∣ ∣ ∣ ] = o p ( n -1 / 2 ). This proves the first part of (A.17). The second part follows immediately from the consistency of ˜ θ and Assumption 5(iv) after noting that ˆ ∆ h ( · , · ), ˆ ∆ g ( · , · ) and ˆ ∆ ln η ( · , · ) are all uniformly bounded (due to Assumption 5(iii) and the compactness of X ).

Assumption 3 of Chernozhukov et al. (2022) requires existence of some C &lt; ∞ independent of ˜ h, ˜ g and ˜ η such that

<!-- formula-not-decoded -->

for all ∥ ∥ ∥ ˜ h -h ∥ ∥ ∥ 2 , ‖ ˜ g -g ‖ 2 , ‖ ˜ η -η ‖ 2 small enough and where the space of ˜ η is { ˜ η : inf a,x ˜ η ( a, x ) &gt; δ &gt; 0 } . 23 These requirements are verified in Section B.4 in Online Appendix B.

Finally, Assumption 5 of Chernozhukov et al. (2022) is directly equivalent to Assumption 5(ii).

23 The assumption in Chernozhukov et al. (2022) also requires L 2 consistency of ˆ h, ˆ g, ˆ η which is directly stated as Assumption 5(iii).