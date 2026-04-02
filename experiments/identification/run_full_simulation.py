"""
Full reward recovery simulation study.

Runs all six analyses from the simulation appendix using the
book-reading MDP with 15 episodes, 3 content levels, 4 wait levels.
Outputs results as JSON for the LaTeX document.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
import numpy as np
from dataclasses import dataclass
import optax
import time
import json
from pathlib import Path


@dataclass
class MDPConfig:
    n_episodes: int = 15
    n_content_levels: int = 3
    n_wait_levels: int = 4
    n_actions: int = 3
    beta: float = 0.95
    price_cost: float = 2.0
    wait_disutility_per_hour: float = 0.08
    content_utilities: tuple = (0.5, 1.5, 3.0)
    wait_hours: tuple = (1, 6, 12, 24)


def build_mdp(cfg, exit_payoff=0.0):
    n_ep, n_c, n_w, n_a = cfg.n_episodes, cfg.n_content_levels, cfg.n_wait_levels, cfg.n_actions
    n_regular = n_ep * n_c * n_w
    ABS = n_regular
    n_states = n_regular + 1

    content_seq = np.array([2, 2, 1, 0, 1, 0, 1, 2, 2, 2, 1, 0, 1, 2, 2][:n_ep])

    P = np.zeros((n_states, n_a, n_states))
    r = np.zeros((n_states, n_a))

    for p in range(n_ep):
        for c in range(n_c):
            for w in range(n_w):
                s = p * n_c * n_w + c * n_w + w
                cu = cfg.content_utilities[c]
                wc = cfg.wait_disutility_per_hour * cfg.wait_hours[w]
                r[s, 0] = cu - cfg.price_cost
                r[s, 1] = cu - wc
                r[s, 2] = exit_payoff
                if p == n_ep - 1:
                    P[s, :, ABS] = 1.0
                else:
                    nc = content_seq[p + 1]
                    ns = (p + 1) * n_c * n_w + nc * n_w + w
                    P[s, 0, ns] = 1.0
                    P[s, 1, ns] = 1.0
                    P[s, 2, ABS] = 1.0

    P[ABS, :, ABS] = 1.0
    r[ABS, :] = 0.0

    next_s = np.argmax(P, axis=2).astype(int)

    return {
        'P': jnp.array(P), 'r': jnp.array(r),
        'next_s': jnp.array(next_s), 'content_seq': content_seq,
        'n_states': n_states, 'n_regular': n_regular, 'ABS': ABS, 'cfg': cfg,
    }


def solve_mdp(P, r, beta, ABS):
    V = jnp.zeros(P.shape[0])
    for _ in range(500):
        Q = r + beta * jnp.einsum('ijk,k->ij', P, V)
        V = logsumexp(Q, axis=1).at[ABS].set(0.0)
    Q = r + beta * jnp.einsum('ijk,k->ij', P, V)
    Q = Q.at[ABS, :].set(0.0)
    V = logsumexp(Q, axis=1).at[ABS].set(0.0)
    A = Q - V[:, None]
    pi = jax.nn.softmax(Q, axis=1)
    return {'Q': Q, 'V': V, 'A': A, 'pi': pi}


def build_P_skip(mdp, skip=2):
    cfg = mdp['cfg']
    nS, nA, ABS = mdp['n_states'], cfg.n_actions, mdp['ABS']
    n_c, n_w, n_ep = cfg.n_content_levels, cfg.n_wait_levels, cfg.n_episodes
    cs = mdp['content_seq']
    Pt = np.zeros((nS, nA, nS))
    for p in range(n_ep):
        for c in range(n_c):
            for w in range(n_w):
                s = p * n_c * n_w + c * n_w + w
                nb = p + skip
                if nb >= n_ep:
                    Pt[s, 0, ABS] = 1.0
                else:
                    Pt[s, 0, nb * n_c * n_w + cs[nb] * n_w + w] = 1.0
                nw = p + 1
                if nw >= n_ep:
                    Pt[s, 1, ABS] = 1.0
                else:
                    Pt[s, 1, nw * n_c * n_w + cs[nw] * n_w + w] = 1.0
                Pt[s, 2, ABS] = 1.0
    Pt[ABS, :, ABS] = 1.0
    return jnp.array(Pt)


def build_P_paywall(mdp, paywall_episodes):
    cfg = mdp['cfg']
    nS, nA, ABS = mdp['n_states'], cfg.n_actions, mdp['ABS']
    n_c, n_w, n_ep = cfg.n_content_levels, cfg.n_wait_levels, cfg.n_episodes
    cs = mdp['content_seq']
    Pt = np.zeros((nS, nA, nS))
    for p in range(n_ep):
        for c in range(n_c):
            for w in range(n_w):
                s = p * n_c * n_w + c * n_w + w
                nxt = p + 1
                if nxt >= n_ep:
                    Pt[s, 0, ABS] = 1.0
                    Pt[s, 1, ABS] = 1.0
                else:
                    nc = cs[nxt]
                    ns = nxt * n_c * n_w + nc * n_w + w
                    Pt[s, 0, ns] = 1.0
                    if p in paywall_episodes:
                        Pt[s, 1, ABS] = 1.0
                    else:
                        Pt[s, 1, ns] = 1.0
                Pt[s, 2, ABS] = 1.0
    Pt[ABS, :, ABS] = 1.0
    return jnp.array(Pt)


def generate_data(mdp, sol, n_traj, key):
    cfg = mdp['cfg']
    pi = np.array(sol['pi'])
    P_np = np.array(mdp['P'])
    ABS = mdp['ABS']
    n_c, n_w = cfg.n_content_levels, cfg.n_wait_levels

    all_s, all_a, all_sp = [], [], []
    for _ in range(n_traj):
        key, k1, k2 = jr.split(key, 3)
        c0 = int(mdp['content_seq'][0])
        w0 = int(jr.choice(k1, cfg.n_wait_levels))
        s = c0 * n_w + w0
        for t in range(cfg.n_episodes + 2):
            if s == ABS:
                break
            key, k3 = jr.split(key)
            a = int(jr.choice(k3, cfg.n_actions, p=jnp.array(pi[s])))
            sp = int(np.argmax(P_np[s, a]))
            all_s.append(s); all_a.append(a); all_sp.append(sp)
            s = sp
    return {'s': jnp.array(all_s), 'a': jnp.array(all_a), 'sp': jnp.array(all_sp),
            'n': len(all_s)}


def estimate_rf(data, nS, nA, n_iter=5000, lr=0.05):
    """Reduced-form logit MLE via L-BFGS-B for reliable convergence."""
    from scipy.optimize import minimize as sp_minimize

    s, a = data['s'], data['a']

    def loss_fn(Q_flat):
        Q = jnp.array(Q_flat.reshape(nS, nA))
        lp = Q - logsumexp(Q, axis=1, keepdims=True)
        return -jnp.mean(lp[s, a])

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    def wrapper(Q_flat):
        l, g = loss_and_grad(Q_flat)
        return float(l), np.array(g, dtype=np.float64)

    result = sp_minimize(
        wrapper, np.zeros(nS * nA), method='L-BFGS-B', jac=True,
        options={'maxiter': n_iter, 'ftol': 1e-15, 'gtol': 1e-10},
    )
    Q = jnp.array(result.x.reshape(nS, nA))
    A = Q - logsumexp(Q, axis=1, keepdims=True)
    return {'A': A, 'Q': Q, 'pi': jax.nn.softmax(Q, axis=1)}


def estimate_iq(data, mdp, n_iter=10000, Q_init=None):
    """Anchor-constrained MLE: logit likelihood + anchor penalties.

    Combines the convex log-likelihood from reduced-form estimation
    with soft anchor constraints on the implied reward. The likelihood
    pins down Q-function differences across actions (the advantage).
    The anchor penalties pin down Q-function levels by requiring that
    the implied reward r_Q(s,exit) = 0 and r_Q(absorbing,:) = 0.

    This is conceptually close to Kang, Yoganarasimhan & Jain (2025),
    who use anchor actions to achieve identification within an ERM
    framework. The key difference is that we impose anchors as soft
    penalties on the logit MLE rather than as Bellman residual
    constraints.

    The objective is:
        min_Q  -NLL(Q; data) + lambda * [||r_Q(:,exit)||^2 + ||r_Q(ABS,:)||^2]

    where r_Q(s,a) = Q(s,a) - beta * sum_s' P(s'|s,a) V_Q(s').
    """
    from scipy.optimize import minimize as sp_minimize

    nS, nA = mdp['n_states'], mdp['cfg'].n_actions
    beta, P, ABS = mdp['cfg'].beta, mdp['P'], mdp['ABS']
    nR = mdp['n_regular']
    s, a = data['s'], data['a']

    # Anchor penalty weight. Needs to be large enough to enforce
    # the anchors but not so large that it dominates the likelihood.
    # Scale relative to the NLL which is O(1) per observation.
    lam = 50.0

    def loss_fn(Q_flat):
        Q = jnp.array(Q_flat.reshape(nS, nA))
        V = logsumexp(Q, axis=1).at[ABS].set(0.0)

        # Negative log-likelihood (same as reduced-form)
        lp = Q - logsumexp(Q, axis=1, keepdims=True)
        nll = -jnp.mean(lp[s, a])

        # Implied reward via inverse Bellman
        EV = jnp.einsum('ijk,k->ij', P, V)
        r_Q = Q - beta * EV

        # Anchor penalties: r(s, exit) = 0 and r(ABS, :) = 0
        exit_pen = jnp.mean(r_Q[:nR, 2]**2)
        abs_pen = jnp.mean(r_Q[ABS, :]**2)

        return nll + lam * (exit_pen + abs_pen)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    def wrapper(Q_flat):
        l, g = loss_and_grad(Q_flat)
        return float(l), np.array(g, dtype=np.float64)

    if Q_init is not None:
        Q0 = np.array(Q_init).flatten()
    else:
        Q0 = np.zeros(nS * nA)

    result = sp_minimize(
        wrapper, Q0, method='L-BFGS-B', jac=True,
        options={'maxiter': n_iter, 'ftol': 1e-15, 'gtol': 1e-10},
    )
    Q = jnp.array(result.x.reshape(nS, nA))

    V = logsumexp(Q, axis=1).at[ABS].set(0.0)
    EV = jnp.einsum('ijk,k->ij', P, V)
    return {'r': Q - beta * EV, 'pi': jax.nn.softmax(Q, axis=1)}


def estimate_gladius(data, mdp, max_epochs=300):
    """GLADIUS: neural Q + EV networks with Bellman consistency.

    Uses the actual GLADIUSEstimator from the econirl package.
    Extracts the implied reward from metadata['reward_table'].
    """
    from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig
    from econirl.core.types import DDCProblem, Panel, Trajectory
    from econirl.preferences.action_reward import ActionDependentReward

    cfg = mdp['cfg']
    nS, nA = mdp['n_states'], cfg.n_actions
    beta, ABS = cfg.beta, mdp['ABS']

    problem = DDCProblem(
        num_states=nS, num_actions=nA,
        discount_factor=beta, scale_parameter=1.0,
    )

    # Build a simple feature matrix (content utility + price + wait cost)
    # 3 features: buy_indicator, content_utility, wait_cost
    features = np.zeros((nS, nA, 3))
    n_c, n_w, n_ep = cfg.n_content_levels, cfg.n_wait_levels, cfg.n_episodes
    for p in range(n_ep):
        for c in range(n_c):
            for w in range(n_w):
                s = p * n_c * n_w + c * n_w + w
                features[s, 0, 0] = 1.0  # buy indicator
                features[s, 0, 1] = cfg.content_utilities[c]
                features[s, 1, 1] = cfg.content_utilities[c]
                features[s, 1, 2] = cfg.wait_hours[w]
    feature_matrix = jnp.array(features, dtype=jnp.float32)
    utility = ActionDependentReward(feature_matrix, ['buy_ind', 'content', 'wait'])

    transitions = mdp['P'].transpose(1, 0, 2)  # (S, A, S') -> (A, S, S')

    # Build panel from data
    s_np = np.array(data['s'])
    a_np = np.array(data['a'])
    sp_np = np.array(data['sp'])

    # Split into trajectories at absorbing state boundaries
    trajectories = []
    traj_start = 0
    for i in range(len(s_np)):
        if s_np[i] == ABS or i == len(s_np) - 1:
            if i > traj_start:
                sl = slice(traj_start, i)
                trajectories.append(Trajectory(
                    states=jnp.array(s_np[sl], dtype=jnp.int32),
                    actions=jnp.array(a_np[sl], dtype=jnp.int32),
                    next_states=jnp.array(sp_np[sl], dtype=jnp.int32),
                ))
            traj_start = i + 1
    if not trajectories:
        # Fallback: single trajectory
        trajectories.append(Trajectory(
            states=jnp.array(s_np, dtype=jnp.int32),
            actions=jnp.array(a_np, dtype=jnp.int32),
            next_states=jnp.array(sp_np, dtype=jnp.int32),
        ))
    panel = Panel(trajectories=trajectories)

    gl_config = GLADIUSConfig(
        q_hidden_dim=64,
        q_num_layers=2,
        v_hidden_dim=64,
        v_num_layers=2,
        q_lr=1e-3,
        v_lr=1e-3,
        max_epochs=max_epochs,
        batch_size=256,
        bellman_penalty_weight=1.0,
        alternating_updates=True,
        patience=30,
        compute_se=False,
        verbose=False,
    )

    estimator = GLADIUSEstimator(gl_config)
    summary = estimator.estimate(panel, utility, problem, transitions)

    reward_table = jnp.array(summary.metadata['reward_table'])
    return {'r': reward_table, 'pi': summary.policy}


def cf_error(P_cf, r_hat, r_true, beta, ABS, nR):
    oracle = solve_mdp(P_cf, r_true, beta, ABS)
    method = solve_mdp(P_cf, r_hat, beta, ABS)
    return float(jnp.mean(jnp.abs(method['pi'][:nR] - oracle['pi'][:nR])))


def main():
    t_start = time.time()
    results = {}

    cfg = MDPConfig()
    mdp = build_mdp(cfg)
    sol = solve_mdp(mdp['P'], mdp['r'], cfg.beta, mdp['ABS'])
    r_true, V_star, A_star = mdp['r'], sol['V'], sol['A']
    nR, ABS, beta, ns = mdp['n_regular'], mdp['ABS'], cfg.beta, mdp['next_s']

    # Analytical methods (population-level convergence)
    # IQ-Learn and GLADIUS both use the inverse Bellman operator:
    #   r_Q = Q* - beta * V*(s')
    # At population convergence they recover r* exactly.
    r_iq = sol['Q'] - beta * V_star[ns]
    delta = 0.5 * V_star
    r_noanch = r_true + delta[:, None] - beta * delta[ns]

    methods = {
        'Oracle': r_true,
        'AIRL+anchors': r_true,
        'IQ / GLADIUS': r_iq,
        'AIRL-no-anchors': r_noanch,
        'Reduced-form': A_star,
    }

    # --- A: Reward Recovery ---
    print("Analysis A: Reward Recovery")
    results['A'] = {}
    for name, rh in methods.items():
        rt, rh_ = r_true[:nR].flatten(), rh[:nR].flatten()
        mse = float(jnp.mean((rt - rh_)**2))
        corr = float(jnp.corrcoef(rt, rh_)[0, 1])
        print(f"  {name:<22} MSE={mse:.3f}  Corr={corr:.3f}")
        results['A'][name] = {'mse': round(mse, 3), 'corr': round(corr, 3)}

    # --- B: Type II CF (k-skip) ---
    print("\nAnalysis B: Type II CF (k-skip)")
    skips = [1, 2, 3, 5, 7, 10]
    results['B'] = {}
    for k in skips:
        Pt = build_P_skip(mdp, skip=k)
        oracle_pi = solve_mdp(Pt, r_true, beta, ABS)['pi'][:nR]
        row = {}
        for name, rh in methods.items():
            cf_pi = solve_mdp(Pt, rh, beta, ABS)['pi'][:nR]
            err = float(jnp.mean(jnp.abs(cf_pi - oracle_pi)))
            row[name] = round(err, 3)
        print(f"  k={k}: {row}")
        results['B'][str(k)] = row

    # --- C: Shaping sweep ---
    print("\nAnalysis C: Shaping sweep")
    alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    Pt3 = build_P_skip(mdp, skip=3)
    oracle_pi3 = solve_mdp(Pt3, r_true, beta, ABS)['pi'][:nR]
    results['C'] = {}
    for alpha in alphas:
        d = alpha * V_star
        r_shaped = r_true + d[:, None] - beta * d[ns]
        cf_pi = solve_mdp(Pt3, r_shaped, beta, ABS)['pi'][:nR]
        err = float(jnp.mean(jnp.abs(cf_pi - oracle_pi3)))
        corr = float(jnp.corrcoef(r_true[:nR].flatten(), r_shaped[:nR].flatten())[0, 1])
        print(f"  alpha={alpha:.2f}: err={err:.3f}, corr={corr:.3f}")
        results['C'][str(alpha)] = {'error': round(err, 3), 'corr': round(corr, 3)}

    # --- D: Sample size (RF vs Anchor-MLE vs GLADIUS) ---
    print("\nAnalysis D: Sample size (RF, Anchor-MLE, GLADIUS)")
    sample_sizes = [500, 2000, 10000]
    n_seeds = 2
    results['D'] = {}
    for N in sample_sizes:
        rf_errs, iq_errs, gl_errs = [], [], []
        for seed in range(n_seeds):
            data = generate_data(mdp, sol, N, jr.PRNGKey(seed * 1000 + N))
            # RF: L-BFGS-B, fully converged
            rf = estimate_rf(data, mdp['n_states'], cfg.n_actions, n_iter=10000)
            rf_err = cf_error(Pt3, rf['A'], r_true, beta, ABS, nR)
            rf_errs.append(rf_err)
            # Anchor-constrained MLE: warm-start from RF Q
            iq = estimate_iq(data, mdp, n_iter=10000, Q_init=rf['Q'])
            iq_err = cf_error(Pt3, iq['r'], r_true, beta, ABS, nR)
            iq_errs.append(iq_err)
            # GLADIUS: neural Q + EV with 150 epochs for speed
            gl = estimate_gladius(data, mdp, max_epochs=150)
            gl_err = cf_error(Pt3, gl['r'], r_true, beta, ABS, nR)
            gl_errs.append(gl_err)
        rf_mean = np.mean(rf_errs)
        iq_mean = np.mean(iq_errs)
        gl_mean = np.mean(gl_errs)
        print(f"  N={N:>5}: RF={rf_mean:.4f}, Anch-MLE={iq_mean:.4f}, GLADIUS={gl_mean:.4f}  ({n_seeds} seeds)")
        results['D'][str(N)] = {
            'rf': round(rf_mean, 4), 'anchor_mle': round(iq_mean, 4), 'gladius': round(gl_mean, 4)
        }

    # --- E: Anchor misspecification ---
    print("\nAnalysis E: Anchor misspecification")
    epsilons = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    results['E'] = {}
    for eps in epsilons:
        mdp_e = build_mdp(cfg, exit_payoff=eps)
        sol_e = solve_mdp(mdp_e['P'], mdp_e['r'], cfg.beta, mdp_e['ABS'])
        ns_e = mdp_e['next_s']
        V_e = sol_e['V']

        h_misspec = jnp.where(jnp.arange(mdp_e['n_states']) == ABS, 0.0, V_e - eps)
        g_misspec = sol_e['A'] - beta * h_misspec[ns_e] + h_misspec[:, None]
        g_misspec = g_misspec.at[:, 2].set(0.0)

        r_e_flat = mdp_e['r'][:nR].flatten()
        g_flat = g_misspec[:nR].flatten()
        corr = float(jnp.corrcoef(r_e_flat, g_flat)[0, 1])

        Pt_e = build_P_skip(mdp_e, skip=3)
        t2_err = cf_error(Pt_e, g_misspec, mdp_e['r'], cfg.beta, ABS, nR)

        sol_g = solve_mdp(mdp_e['P'], g_misspec, cfg.beta, ABS)
        pi_shift = float(jnp.mean(jnp.abs(sol_g['pi'][:nR] - sol_e['pi'][:nR])))

        print(f"  eps={eps:.1f}: corr={corr:.3f}, t2_err={t2_err:.3f}, pi_shift={pi_shift:.3f}")
        results['E'][str(eps)] = {
            'corr': round(corr, 3), 'type2_err': round(t2_err, 3), 'pi_shift': round(pi_shift, 3)
        }

    # --- F: Content-based paywall ---
    print("\nAnalysis F: Content-based paywall")
    cs = mdp['content_seq']
    paywall_eps = [i for i in range(len(cs)) if cs[i] == 2]
    Pt_pw = build_P_paywall(mdp, paywall_eps)
    oracle_pw = solve_mdp(Pt_pw, r_true, beta, ABS)['pi'][:nR]
    results['F'] = {}
    for name, rh in methods.items():
        cf_pi = solve_mdp(Pt_pw, rh, beta, ABS)['pi'][:nR]
        err = float(jnp.mean(jnp.abs(cf_pi - oracle_pw)))
        print(f"  {name:<22}: {err:.3f}")
        results['F'][name] = round(err, 3)

    print(f"\nTotal: {time.time()-t_start:.1f}s")

    out = Path(__file__).parent / "results" / "full_simulation.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
