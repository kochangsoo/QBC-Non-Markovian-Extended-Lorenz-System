"""
Ensemble Validation — N = 1000 Trajectories
============================================

Appendix A.10: N_ens = 1000; ICs drawn from U([1−ε, 1+ε]) for (x,y,z)
with ε = 0.01, w(0) = 0 fixed; seed = 42; diagnostics computed over
τ ∈ [10, 60].
"""

import numpy as np
from bounce_stats import single_trajectory_diagnostics


def generate_ensemble_ics(N_ens=1000, epsilon=0.01, seed=42,
                          nominal_ic=None, w0_fixed=0.0):
    """Generate perturbed initial conditions.

    Parameters
    ----------
    N_ens : int
    epsilon : float
    seed : int
    nominal_ic : array_like, shape (3,) or None
        Nominal [x0, y0, z0]; defaults to [1, 1, 1].
    w0_fixed : float
        Fixed w(0).

    Returns
    -------
    ics : ndarray, shape (N_ens, 4)
    """
    if nominal_ic is None:
        nominal_ic = np.array([1.0, 1.0, 1.0])
    rng = np.random.default_rng(seed)
    perturbed = rng.uniform(
        nominal_ic * (1 - epsilon),
        nominal_ic * (1 + epsilon),
        size=(N_ens, 3)
    )
    w_col = np.full((N_ens, 1), w0_fixed)
    return np.hstack([perturbed, w_col])


def run_ensemble(sigma, beta, rho, kappa, tau_m,
                 N_ens=1000, epsilon=0.01, seed=42,
                 T_total=60.0, tau_start=10.0, tau_end=60.0,
                 dt_max=0.01, rtol=1e-9, atol=1e-12,
                 verbose=True):
    """Run the full ensemble and collect per-trajectory diagnostics.

    Returns
    -------
    results : dict of ndarray
        Keys: asymmetry, bounce_count, mean_interval, std_interval.
        Each is shape (N_ens,).
    """
    ics = generate_ensemble_ics(N_ens, epsilon, seed)

    out = dict(asymmetry=[], bounce_count=[],
               mean_interval=[], std_interval=[])

    for i in range(N_ens):
        if verbose and (i + 1) % 100 == 0:
            print(f"  trajectory {i+1}/{N_ens}")
        diag = single_trajectory_diagnostics(
            ics[i], sigma, beta, rho, kappa, tau_m,
            T_total=T_total, tau_start=tau_start, tau_end=tau_end,
            dt_max=dt_max, rtol=rtol, atol=atol
        )
        for key in out:
            out[key].append(diag[key])

    return {k: np.array(v) for k, v in out.items()}


def ensemble_summary(results):
    """Compute mean ± std and 95 % CI for each diagnostic.

    Parameters
    ----------
    results : dict of ndarray

    Returns
    -------
    summary : dict
        For each key: (mean, std, p2.5, p97.5).
    """
    summary = {}
    for key, vals in results.items():
        valid = vals[~np.isnan(vals)]
        summary[key] = dict(
            mean=np.mean(valid),
            std=np.std(valid, ddof=1),
            p025=np.percentile(valid, 2.5),
            p975=np.percentile(valid, 97.5),
            n_valid=len(valid),
        )
    return summary
