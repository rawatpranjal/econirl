"""Quick smoke tests before any paid RunPod compute.

Runs five cheap cells locally on CPU using the high-level
``econirl.estimators`` sklearn-style API:

  1. NFXP on rust-small
  2. CCP on rust-small
  3. MCE-IRL on rust-small (equivalence cross-check)
  4. MCE-IRL on ziebart-small (canonical IRL hero)
  5. NFXP on ziebart-small (control)

Each test fits the estimator on the bundled dataset, checks that the
estimator converged, prints the recovered parameters, the
log-likelihood, the wall-clock time, and (where ground truth is
available) the cosine similarity to the true parameter vector.

Total wall-clock budget: under 5 minutes on a laptop. No GPU and no
RunPod required. If all five pass, the deep-run pipeline's
estimator-and-loader integration is sound and we can move on to
RunPod with confidence.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from econirl.datasets import load_rust_small, load_ziebart_small


_RUST_TRUTH = np.array([0.001, 3.0])


@dataclass
class SmokeResult:
    name: str
    converged: bool
    params: dict[str, float] | None
    log_likelihood: float | None
    wall_clock_s: float
    cosine_similarity: float | None
    exception: str | None


def _cosine_similarity(estimated: Any, truth: np.ndarray) -> float | None:
    if estimated is None:
        return None
    a = np.asarray(estimated).astype(float).flatten()
    if a.shape != truth.shape:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(truth))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, truth) / (na * nb))


def smoke_nfxp_rust() -> SmokeResult:
    from econirl.estimators import NFXP

    df = load_rust_small()
    model = NFXP(n_states=90, discount=0.9999, verbose=False)
    start = time.time()
    try:
        model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
        elapsed = time.time() - start
        return SmokeResult(
            name="NFXP on rust-small",
            converged=bool(model.converged_),
            params=model.params_,
            log_likelihood=model.log_likelihood_,
            wall_clock_s=elapsed,
            cosine_similarity=_cosine_similarity(model.coef_, _RUST_TRUTH),
            exception=None,
        )
    except Exception:
        return SmokeResult(
            name="NFXP on rust-small",
            converged=False,
            params=None,
            log_likelihood=None,
            wall_clock_s=time.time() - start,
            cosine_similarity=None,
            exception=traceback.format_exc(),
        )


def smoke_ccp_rust() -> SmokeResult:
    from econirl.estimators import CCP

    df = load_rust_small()
    model = CCP(n_states=90, discount=0.9999, verbose=False)
    start = time.time()
    try:
        model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
        elapsed = time.time() - start
        return SmokeResult(
            name="CCP on rust-small",
            converged=bool(model.converged_),
            params=model.params_,
            log_likelihood=model.log_likelihood_,
            wall_clock_s=elapsed,
            cosine_similarity=_cosine_similarity(model.coef_, _RUST_TRUTH),
            exception=None,
        )
    except Exception:
        return SmokeResult(
            name="CCP on rust-small",
            converged=False,
            params=None,
            log_likelihood=None,
            wall_clock_s=time.time() - start,
            cosine_similarity=None,
            exception=traceback.format_exc(),
        )


def smoke_mceirl_rust() -> SmokeResult:
    from econirl.estimators import MCEIRL

    df = load_rust_small()
    model = MCEIRL(n_states=90, discount=0.9999, verbose=False)
    start = time.time()
    try:
        model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
        elapsed = time.time() - start
        return SmokeResult(
            name="MCE-IRL on rust-small",
            converged=bool(model.converged_),
            params=model.params_,
            log_likelihood=model.log_likelihood_,
            wall_clock_s=elapsed,
            cosine_similarity=_cosine_similarity(model.coef_, _RUST_TRUTH),
            exception=None,
        )
    except Exception:
        return SmokeResult(
            name="MCE-IRL on rust-small",
            converged=False,
            params=None,
            log_likelihood=None,
            wall_clock_s=time.time() - start,
            cosine_similarity=None,
            exception=traceback.format_exc(),
        )


def smoke_mceirl_ziebart() -> SmokeResult:
    from econirl.estimators import MCEIRL

    df = load_ziebart_small()
    model = MCEIRL(n_states=100, n_actions=5, discount=0.99, verbose=False)
    start = time.time()
    try:
        model.fit(data=df, state="state", action="action", id="taxi_id")
        elapsed = time.time() - start
        return SmokeResult(
            name="MCE-IRL on ziebart-small",
            converged=bool(model.converged_),
            params=model.params_,
            log_likelihood=model.log_likelihood_,
            wall_clock_s=elapsed,
            cosine_similarity=None,
            exception=None,
        )
    except Exception:
        return SmokeResult(
            name="MCE-IRL on ziebart-small",
            converged=False,
            params=None,
            log_likelihood=None,
            wall_clock_s=time.time() - start,
            cosine_similarity=None,
            exception=traceback.format_exc(),
        )


def smoke_nfxp_ziebart() -> SmokeResult:
    from econirl.estimators import NFXP

    df = load_ziebart_small()
    model = NFXP(n_states=100, n_actions=5, discount=0.99, verbose=False)
    start = time.time()
    try:
        model.fit(data=df, state="state", action="action", id="taxi_id")
        elapsed = time.time() - start
        return SmokeResult(
            name="NFXP on ziebart-small",
            converged=bool(model.converged_),
            params=model.params_,
            log_likelihood=model.log_likelihood_,
            wall_clock_s=elapsed,
            cosine_similarity=None,
            exception=None,
        )
    except Exception:
        return SmokeResult(
            name="NFXP on ziebart-small",
            converged=False,
            params=None,
            log_likelihood=None,
            wall_clock_s=time.time() - start,
            cosine_similarity=None,
            exception=traceback.format_exc(),
        )


def _print_result(r: SmokeResult) -> None:
    status = "PASS" if r.converged and r.exception is None else "FAIL"
    print(f"\n[{status}] {r.name}  ({r.wall_clock_s:.1f}s)")
    if r.exception:
        first_line = r.exception.strip().splitlines()[-1]
        print(f"        exception: {first_line}")
        return
    print(f"        converged: {r.converged}")
    print(f"        log_likelihood: {r.log_likelihood:.2f}")
    if r.params:
        for k, v in r.params.items():
            print(f"        {k}: {v:.6f}")
    if r.cosine_similarity is not None:
        print(f"        cosine_similarity_to_truth: {r.cosine_similarity:.4f}")


def main() -> None:
    print("=" * 60)
    print("JSS deep-run smoke tests")
    print("=" * 60)

    results = [
        smoke_nfxp_rust(),
        smoke_ccp_rust(),
        smoke_mceirl_rust(),
        smoke_nfxp_ziebart(),
        smoke_mceirl_ziebart(),
    ]

    for r in results:
        _print_result(r)

    print()
    n_pass = sum(1 for r in results if r.converged and r.exception is None)
    print("=" * 60)
    print(f"Smoke summary: {n_pass} of {len(results)} passed")
    print("=" * 60)

    if n_pass < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
