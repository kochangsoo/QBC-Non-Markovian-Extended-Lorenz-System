"""
Maximal Lyapunov Exponent — Benettin Renormalisation Method
===========================================================

Implements the Benettin et al. (1980) algorithm with Gram–Schmidt
orthogonalisation and periodic renormalisation (Appendix A.5, A.11).

References
----------
[27] Benettin, G. et al., Meccanica 15, 9 (1980).
[28] Wolf, A. et al., Physica D 16, 285 (1985).
"""

import numpy as np
from scipy.integrate import solve_ivp
from qbc_model import qbc_rhs, qbc_jacobian


def _variational_rhs(t, Y, sigma, beta, rho, kappa, tau_m):
    """Combined state + tangent-vector equations for the full Lyapunov
    spectrum (4 exponents).

    Y = [x, y, z, w,  δ₁(16 elements)] — total length 4 + 16 = 20.
    """
    n = 4
    state = Y[:n]
    # Tangent vectors stored column-major: Φ is 4×4
    Phi = Y[n:].reshape(n, n)
    J = qbc_jacobian(state, sigma, beta, rho, kappa, tau_m)
    dstate = qbc_rhs(t, state, sigma, beta, rho, kappa, tau_m)
    dPhi = J @ Phi
    return np.concatenate([dstate, dPhi.ravel()])


def maximal_lyapunov(sigma, beta, rho, kappa, tau_m,
                     state0, T_total=150.0, T_transient=30.0,
                     renorm_interval=0.5, d0=1e-8,
                     rtol=1e-9, atol=1e-12, dt_max=0.01):
    """Compute the maximal Lyapunov exponent λ₁.

    Parameters
    ----------
    state0 : array_like, shape (4,)
        Initial condition [x, y, z, w].
    T_total, T_transient, renorm_interval : float
        See LyapunovConfig.
    d0 : float
        Not used directly (QR method); kept for API compatibility.

    Returns
    -------
    lambda1 : float
        Maximal Lyapunov exponent estimate.
    """
    n = 4
    params = (sigma, beta, rho, kappa, tau_m)

    # --- 1. Remove transient ---
    sol_tr = solve_ivp(qbc_rhs, [0, T_transient], state0,
                       args=params, method="RK45",
                       rtol=rtol, atol=atol, max_step=dt_max,
                       dense_output=False)
    state = sol_tr.y[:, -1]

    # --- 2. Benettin iteration ---
    T_analysis = T_total - T_transient
    n_renorm = int(T_analysis / renorm_interval)

    # Initialise tangent vector (unit)
    Phi0 = np.eye(n)
    sum_log = np.zeros(n)

    for _ in range(n_renorm):
        Y0 = np.concatenate([state, Phi0.ravel()])
        sol = solve_ivp(_variational_rhs, [0, renorm_interval], Y0,
                        args=params, method="RK45",
                        rtol=rtol, atol=atol, max_step=dt_max)
        Y1 = sol.y[:, -1]
        state = Y1[:n]
        Phi = Y1[n:].reshape(n, n)

        # QR decomposition (Gram–Schmidt)
        Q, R = np.linalg.qr(Phi)
        for i in range(n):
            ri = abs(R[i, i])
            if ri > 0:
                sum_log[i] += np.log(ri)
        Phi0 = Q

    lyap_spectrum = sum_log / T_analysis
    return lyap_spectrum[0]  # λ₁


def lyapunov_spectrum(sigma, beta, rho, kappa, tau_m,
                      state0, **kwargs):
    """Full 4-exponent Lyapunov spectrum (for advanced diagnostics).

    Returns
    -------
    spectrum : ndarray, shape (4,)
        [λ₁, λ₂, λ₃, λ₄] sorted descending.
    """
    n = 4
    params = (sigma, beta, rho, kappa, tau_m)
    T_total = kwargs.get("T_total", 150.0)
    T_transient = kwargs.get("T_transient", 30.0)
    renorm_interval = kwargs.get("renorm_interval", 0.5)
    rtol = kwargs.get("rtol", 1e-9)
    atol = kwargs.get("atol", 1e-12)
    dt_max = kwargs.get("dt_max", 0.01)

    sol_tr = solve_ivp(qbc_rhs, [0, T_transient], state0,
                       args=params, method="RK45",
                       rtol=rtol, atol=atol, max_step=dt_max)
    state = sol_tr.y[:, -1]

    T_analysis = T_total - T_transient
    n_renorm = int(T_analysis / renorm_interval)
    Phi0 = np.eye(n)
    sum_log = np.zeros(n)

    for _ in range(n_renorm):
        Y0 = np.concatenate([state, Phi0.ravel()])
        sol = solve_ivp(_variational_rhs, [0, renorm_interval], Y0,
                        args=params, method="RK45",
                        rtol=rtol, atol=atol, max_step=dt_max)
        Y1 = sol.y[:, -1]
        state = Y1[:n]
        Phi = Y1[n:].reshape(n, n)
        Q, R = np.linalg.qr(Phi)
        for i in range(n):
            ri = abs(R[i, i])
            if ri > 0:
                sum_log[i] += np.log(ri)
        Phi0 = Q

    return sum_log / T_analysis
