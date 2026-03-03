"""
Quantum Bounce Core (QBC) — Non-Markovian Extended Lorenz System
================================================================

4D autonomous ODE (Eq. 4 of the manuscript):

    ẋ = σ(y − x)
    ẏ = x(ρ − z) − y
    ż = xy − βz + κw
    ẇ = (xy − w) / τ_m

where w is the memory variable arising from a Volterra-type exponential
decay kernel (Eq. 3).  Setting κ = 0 recovers the classical 3D Lorenz
system with a decoupled, exponentially decaying w.

Reference
---------
Ko, C. (2026). "A Minimal Non-Markovian Lorenz-Like Model for a Quantum
Bounce Core: Dynamical Diagnostics for the Cosmic Dipole Anomaly."
Manuscript for *Chaos, Solitons & Fractals*.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Default parameters (Table 1 / §3)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    sigma=10.0,
    beta=8.0 / 3.0,
    rho=28.0,
    kappa=0.0,
    tau_m=1.0,
)


def qbc_rhs(t, state, sigma, beta, rho, kappa, tau_m):
    """Right-hand side of the 4D QBC system.

    Parameters
    ----------
    t : float
        (Unused — autonomous system.)
    state : array_like, shape (4,)
        [x, y, z, w].
    sigma, beta, rho, kappa, tau_m : float
        Model parameters.

    Returns
    -------
    dstate : ndarray, shape (4,)
    """
    x, y, z, w = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z + kappa * w
    dw = (x * y - w) / tau_m
    return np.array([dx, dy, dz, dw])


def qbc_jacobian(state, sigma, beta, rho, kappa, tau_m):
    """Analytic 4×4 Jacobian (Appendix A.9).

    Parameters
    ----------
    state : array_like, shape (4,)
    sigma, beta, rho, kappa, tau_m : float

    Returns
    -------
    J : ndarray, shape (4, 4)
    """
    x, y, z, w = state
    mu = 1.0 / tau_m
    return np.array([
        [-sigma,    sigma,  0.0,    0.0   ],
        [rho - z,  -1.0,   -x,     0.0   ],
        [y,         x,     -beta,   kappa ],
        [y * mu,    x * mu, 0.0,   -mu    ],
    ])


def equilibria(sigma, beta, rho, kappa, **_):
    """Compute the three fixed points E₀, E₊, E₋ (Proposition 1, §5.1).

    Returns
    -------
    E0 : ndarray, shape (4,)
    Eplus : ndarray, shape (4,)
    Eminus : ndarray, shape (4,)
    """
    E0 = np.array([0.0, 0.0, 0.0, 0.0])
    if rho <= 1.0:
        return E0, E0.copy(), E0.copy()
    a2 = beta * (rho - 1.0) / (1.0 + kappa)
    a = np.sqrt(a2)
    z_eq = rho - 1.0
    Eplus  = np.array([ a,  a, z_eq,  a2])
    Eminus = np.array([-a, -a, z_eq,  a2])
    return E0, Eplus, Eminus


def hopf_boundary_rh(kappa, tau_m, sigma=10.0, beta=8.0/3.0,
                     rho_range=(1.01, 200.0), tol=1e-10):
    """Find ρ_H(κ, τ_m) via Routh–Hurwitz Δ₃ = 0 (Proposition 2, §5.2).

    Uses bisection on the criterion
        Δ₃ := c₁(c₃ c₂ − c₁) − c₃² c₀ = 0.

    Returns
    -------
    rho_H : float or NaN if not found.
    """
    mu = 1.0 / tau_m

    def delta3(rho):
        a2 = beta * (rho - 1.0) / (1.0 + kappa)
        c3 = sigma + beta + 1.0 + mu
        c2 = a2 + beta * sigma + beta + mu * (beta + sigma + 1.0)
        c1 = mu * beta * (rho + sigma) + 2.0 * sigma * a2
        c0 = 2.0 * mu * beta * sigma * (rho - 1.0)
        return c1 * (c3 * c2 - c1) - c3**2 * c0

    lo, hi = rho_range
    flo, fhi = delta3(lo), delta3(hi)
    if flo * fhi > 0:
        return np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = delta3(mid)
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)
