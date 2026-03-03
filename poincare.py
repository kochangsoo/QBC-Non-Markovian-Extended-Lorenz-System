"""
Poincaré Section — z = 25, Upward Crossings
=============================================

Appendix A.4: Record (x, y) at upward crossings of z = 25.
"""

import numpy as np
from scipy.integrate import solve_ivp
from qbc_model import qbc_rhs


def poincare_section(sigma, beta, rho, kappa, tau_m,
                     state0, T_total=300.0, T_transient=50.0,
                     z_section=25.0, dt_max=0.01,
                     rtol=1e-9, atol=1e-12):
    """Compute Poincaré section at z = z_section (upward crossings).

    Returns
    -------
    x_cross : ndarray
    y_cross : ndarray
    """
    params = (sigma, beta, rho, kappa, tau_m)

    # Remove transient
    sol_tr = solve_ivp(qbc_rhs, [0, T_transient], state0,
                       args=params, method="RK45",
                       rtol=rtol, atol=atol, max_step=dt_max)
    state_start = sol_tr.y[:, -1]

    # Main integration
    T_analysis = T_total - T_transient
    sol = solve_ivp(qbc_rhs, [0, T_analysis], state_start,
                    args=params, method="RK45",
                    rtol=rtol, atol=atol, max_step=dt_max)

    t = sol.t
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]

    x_cross, y_cross = [], []
    for i in range(1, len(z)):
        # Upward crossing: z[i-1] < z_section and z[i] >= z_section
        if z[i - 1] < z_section and z[i] >= z_section:
            frac = (z_section - z[i - 1]) / (z[i] - z[i - 1])
            x_cross.append(x[i - 1] + frac * (x[i] - x[i - 1]))
            y_cross.append(y[i - 1] + frac * (y[i] - y[i - 1]))

    return np.array(x_cross), np.array(y_cross)
