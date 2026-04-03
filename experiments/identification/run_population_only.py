"""
Population-only reward recovery study.

Computes Analyses A, B, C, E, F, G from the simulation appendix
without running any finite-sample estimators. This is fast and
produces experiments/identification/results/full_simulation.json
with all population sections populated (including section G).
"""

import json
from pathlib import Path
import subprocess
import sys
import importlib.util

# Local import of metrics without requiring package installation
_metrics_path = Path(__file__).parent / "metrics.py"
_spec = importlib.util.spec_from_file_location("_id_metrics", _metrics_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)  # type: ignore
softmax = _mod.softmax
metrics_for_counterfactual = _mod.metrics_for_counterfactual

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np


class MDPConfig:
    def __init__(self):
        self.n_episodes = 15
        self.n_content_levels = 3
        self.n_wait_levels = 4
        self.n_actions = 3
        self.beta = 0.95
        self.price_cost = 2.0
        self.wait_disutility_per_hour = 0.08
        self.content_utilities = (0.5, 1.5, 3.0)
        self.wait_hours = (1, 6, 12, 24)


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
        'P': jnp.array(P), 'r': jnp.array(r), 'next_s': jnp.array(next_s),
        'content_seq': content_seq, 'n_states': n_states, 'n_regular': n_regular, 'ABS': ABS, 'cfg': cfg,
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
    return {'Q': Q, 'V': V, 'A': A}


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


def solve_mdp_free(P, r, beta):
    V = jnp.zeros(P.shape[0])
    for _ in range(500):
        Q = r + beta * jnp.einsum('ijk,k->ij', P, V)
        V = logsumexp(Q, axis=1)
    Q = r + beta * jnp.einsum('ijk,k->ij', P, V)
    return {'Q': Q, 'V': V}


def main():
    cfg = MDPConfig()
    mdp = build_mdp(cfg)
    sol = solve_mdp(mdp['P'], mdp['r'], cfg.beta, mdp['ABS'])
    r_true, V_star, A_star = mdp['r'], sol['V'], sol['A']
    nR, ABS, beta, ns = mdp['n_regular'], mdp['ABS'], cfg.beta, mdp['next_s']

    r_iq = sol['Q'] - beta * V_star[ns]
    delta = 0.5 * V_star
    r_noanch = r_true + delta[:, None] - beta * delta[ns]

    methods = {
        'Oracle': r_true,
        'AIRL+anchors': r_true,
        'IQ / GLADIUS': r_iq,
        'NFXP': r_true,
        'AIRL-no-anchors': r_noanch,
        'Reduced-form': A_star,
    }

    results = {}

    # A: reward recovery
    results['A'] = {}
    for name, rh in methods.items():
        rt, rh_ = r_true[:nR].flatten(), rh[:nR].flatten()
        mse = float(jnp.mean((rt - rh_)**2))
        corr = float(jnp.corrcoef(rt, rh_)[0, 1])
        results['A'][name] = {'mse': round(mse, 3), 'corr': round(corr, 3)}

    # B: Type II k-skip
    results['B'] = {}
    results.setdefault('metrics', {})
    results['metrics']['B'] = {}
    for k in [1, 2, 3, 5, 7, 10]:
        Pt = build_P_skip(mdp, skip=k)
        oracle_q = solve_mdp(Pt, r_true, beta, ABS)['Q']
        oracle_pi_full = softmax(oracle_q)
        oracle_pi = oracle_pi_full[:nR]
        row = {}
        mrow = {}
        for name, rh in methods.items():
            cf_q = solve_mdp(Pt, rh, beta, ABS)['Q']
            cf_pi_full = softmax(cf_q)
            cf_pi = cf_pi_full[:nR]
            err = float(jnp.mean(jnp.abs(cf_pi - oracle_pi)))
            row[name] = round(err, 3)
            # Complementary metrics
            m = metrics_for_counterfactual(Pt, r_true, beta, oracle_pi_full, cf_pi_full, start_uniform_over=nR)
            mrow[name] = m
        results['B'][str(k)] = row
        results['metrics']['B'][str(k)] = mrow

    # C: shaping sweep (k=3)
    Pt3 = build_P_skip(mdp, skip=3)
    oracle_pi3 = jax.nn.softmax(solve_mdp(Pt3, r_true, beta, ABS)['Q'], axis=1)[:nR]
    results['C'] = {}
    for alpha in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        d = alpha * V_star
        r_shaped = r_true + d[:, None] - beta * d[ns]
        cf_pi = jax.nn.softmax(solve_mdp(Pt3, r_shaped, beta, ABS)['Q'], axis=1)[:nR]
        err = float(jnp.mean(jnp.abs(cf_pi - oracle_pi3)))
        corr = float(jnp.corrcoef(r_true[:nR].flatten(), r_shaped[:nR].flatten())[0, 1])
        results['C'][str(alpha)] = {'error': round(err, 3), 'corr': round(corr, 3)}

    # E: anchor misspecification
    results['E'] = {}
    for eps in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
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
        oracle_e = jax.nn.softmax(solve_mdp(Pt_e, mdp_e['r'], beta, ABS)['Q'], axis=1)[:nR]
        cf_e = jax.nn.softmax(solve_mdp(Pt_e, g_misspec, beta, ABS)['Q'], axis=1)[:nR]
        t2_err = float(jnp.mean(jnp.abs(cf_e - oracle_e)))
        sol_g = solve_mdp(mdp_e['P'], g_misspec, beta, ABS)
        sol_true = solve_mdp(mdp_e['P'], mdp_e['r'], beta, ABS)
        pi_shift = float(jnp.mean(jnp.abs(jax.nn.softmax(sol_g['Q'], axis=1)[:nR] - jax.nn.softmax(sol_true['Q'], axis=1)[:nR])))
        results['E'][str(eps)] = {'corr': round(corr, 3), 'type2_err': round(t2_err, 3), 'pi_shift': round(pi_shift, 3)}

    # F: content-based paywall
    cs = mdp['content_seq']
    paywall_eps = [i for i in range(len(cs)) if cs[i] == 2]
    Pt_pw = build_P_paywall(mdp, paywall_eps)
    oracle_pw_full = softmax(solve_mdp(Pt_pw, r_true, beta, ABS)['Q'])
    oracle_pw = oracle_pw_full[:nR]
    results['F'] = {}
    results['metrics']['F'] = {}
    for name, rh in methods.items():
        cf_full = softmax(solve_mdp(Pt_pw, rh, beta, ABS)['Q'])
        cf_pi = cf_full[:nR]
        err = float(jnp.mean(jnp.abs(cf_pi - oracle_pw)))
        results['F'][name] = round(err, 3)
        m = metrics_for_counterfactual(Pt_pw, r_true, beta, oracle_pw_full, cf_full, start_uniform_over=nR)
        results['metrics']['F'][name] = m

    # G: Fu et al. state-only reward
    mdp_g = build_mdp(cfg)
    r_state_only = np.zeros((mdp_g['n_states'], cfg.n_actions))
    n_c, n_w, n_ep = cfg.n_content_levels, cfg.n_wait_levels, cfg.n_episodes
    for p in range(n_ep):
        for c in range(n_c):
            for w in range(n_w):
                s = p * n_c * n_w + c * n_w + w
                r_state_only[s, :] = cfg.content_utilities[c]
    r_state_only[mdp_g['ABS'], :] = 0.0
    r_state_only = jnp.array(r_state_only)
    mdp_g['r'] = r_state_only
    sol_g = solve_mdp_free(mdp_g['P'], r_state_only, cfg.beta)
    r_true_g = r_state_only
    ns_g = mdp_g['next_s']
    V_star_g = sol_g['V']
    r_iq_g = sol_g['Q'] - beta * V_star_g[ns_g]
    delta_g = 0.5 * V_star_g
    r_noanch_g = r_true_g + delta_g[:, None] - beta * delta_g[ns_g]
    r_mean = float(r_true_g[:nR, 0].mean())
    r_airl_stateonly_g = jnp.broadcast_to((r_true_g[:, 0] - r_mean)[:, None], (mdp_g['n_states'], cfg.n_actions))
    methods_g = {
        'Oracle': r_true_g,
        'AIRL+anchors': r_true_g,
        'AIRL-state-only': r_airl_stateonly_g,
        'AIRL-shaped': r_noanch_g,
        'IQ-Learn': r_iq_g,
        'NFXP': r_true_g,
        'Reduced-form': sol_g['Q'] - logsumexp(sol_g['Q'], axis=1, keepdims=True),
    }
    results['G'] = {'reward': {}, 'type2': {}}
    for name, rh in methods_g.items():
        rt = r_true_g[:nR].flatten(); rh_ = rh[:nR].flatten()
        mse = float(jnp.mean((rt - rh_)**2))
        corr = float(jnp.corrcoef(rt, rh_)[0, 1]) if float(jnp.std(rh_)) > 1e-10 else 0.0
        results['G']['reward'][name] = {'mse': round(mse, 3), 'corr': round(corr, 3)}
    Pt3_g = build_P_skip(mdp_g, skip=3)
    oracle_cf_g = jax.nn.softmax(solve_mdp_free(Pt3_g, r_true_g, beta)['Q'], axis=1)[:nR]
    for name, rh in methods_g.items():
        cf_pi = jax.nn.softmax(solve_mdp_free(Pt3_g, rh, beta)['Q'], axis=1)[:nR]
        err = float(jnp.mean(jnp.abs(cf_pi - oracle_cf_g)))
        results['G']['type2'][name] = round(err, 6)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "full_simulation.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")
    # Repro info
    repro = {
        "python": sys.version,
        "jax_version": getattr(jax, "__version__", "unknown"),
        "numpy_version": getattr(np, "__version__", "unknown"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "config": {
            "beta": cfg.beta,
            "episodes": cfg.n_episodes,
            "content_levels": cfg.n_content_levels,
            "wait_levels": cfg.n_wait_levels,
            "price_cost": cfg.price_cost,
            "wait_cost_per_hour": cfg.wait_disutility_per_hour,
        },
    }
    with open(out_dir / "REPRO.json", 'w') as f:
        json.dump(repro, f, indent=2)


if __name__ == '__main__':
    main()
