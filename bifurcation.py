"""
Bifurcation Diagrams — ρ-Sweep
===============================

Appendix A.3 / A.11:  ρ ∈ [0, 60], Δρ = 0.2; integrate τ ∈ [0, 250];
discard first 50 %; retain up to 150 local maxima of x(τ) per ρ value.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
from qbc_model import qbc_rhs


def bifurcation_sweep(sigma, beta, kappa, tau_m,
                      rho_min=0.0, rho_max=60.0, d_rho=0.2,
                      T_integrate=250.0, discard_frac=0.5,
                      max_peaks=150, dt_max=0.02,
                      state0=None, rtol=1e-9, atol=1e-12,
                      verbose=True):
    """Sweep ρ and collect local maxima of x(τ) for a bifurcation diagram.

    Parameters
    ----------
    See BifurcationConfig / A.11.

    Returns
    -------
    rho_vals : list of float
    x_maxima : list of ndarray
        For each ρ, the array of x local-maxima values.
    """
    if state0 is None:
        state0 = np.array([1.0, 1.0, 1.0, 0.0])

    rho_vals = np.arange(rho_min + d_rho, rho_max + d_rho / 2, d_rho)
    all_rho = []
    all_xmax = []

    for i, rho in enumerate(rho_vals):
        if verbose and i % 20 == 0:
            print(f"  ρ = {rho:.1f}  ({i+1}/{len(rho_vals)})")

        params = (sigma, beta, rho, kappa, tau_m)
        sol = solve_ivp(qbc_rhs, [0, T_integrate], state0,
                        args=params, method="RK45",
                        rtol=rtol, atol=atol, max_step=dt_max)

        # Discard transient
        t = sol.t
        x = sol.y[0]
        t_cut = T_integrate * discard_frac
        mask = t >= t_cut
        x_ss = x[mask]

        # Find local maxima
        idx = argrelextrema(x_ss, np.greater, order=1)[0]
        peaks = x_ss[idx]
        if len(peaks) > max_peaks:
            peaks = peaks[-max_peaks:]  # keep last ones

        for p in peaks:
            all_rho.append(rho)
            all_xmax.append(p)

        # Use last state as IC for next ρ (continuation)
        state0 = sol.y[:, -1]

    return np.array(all_rho), np.array(all_xmax)
