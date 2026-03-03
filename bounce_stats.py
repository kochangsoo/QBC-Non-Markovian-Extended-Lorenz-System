"""
Bounce Statistics and Wing-Occupation Asymmetry
================================================

Appendix A.6: A quantum bounce event is defined as x crossing zero
from negative to positive with ẋ > 0.

Wing-occupation asymmetry: P(x > 0) − P(x < 0), computed as the
fraction of integration time spent in each wing.
"""

import numpy as np
from scipy.integrate import solve_ivp
from qbc_model import qbc_rhs


def integrate_trajectory(state0, sigma, beta, rho, kappa, tau_m,
                         T_total=60.0, dt_max=0.01,
                         rtol=1e-9, atol=1e-12, dense=True):
    """Integrate the QBC system and return the solution object.

    Parameters
    ----------
    T_total : float
        Total integration time (includes any desired transient).
    dense : bool
        If True, use dense_output for interpolation.

    Returns
    -------
    sol : OdeSolution
    """
    params = (sigma, beta, rho, kappa, tau_m)
    sol = solve_ivp(qbc_rhs, [0, T_total], state0,
                    args=params, method="RK45",
                    rtol=rtol, atol=atol, max_step=dt_max,
                    dense_output=dense)
    return sol


def detect_bounces(t, x, tau_start=10.0, tau_end=60.0):
    """Detect bounce events (neg → pos zero-crossings of x).

    Parameters
    ----------
    t : ndarray
        Time array.
    x : ndarray
        x(τ) trajectory.
    tau_start, tau_end : float
        Analysis window (A.6).

    Returns
    -------
    bounce_times : ndarray
        Times at which bounces occur.
    bounce_count : int
    intervals : ndarray
        Inter-bounce intervals.
    """
    mask = (t >= tau_start) & (t <= tau_end)
    t_w = t[mask]
    x_w = x[mask]

    bounce_times = []
    for i in range(1, len(x_w)):
        if x_w[i - 1] < 0 and x_w[i] >= 0:
            # Linear interpolation for precise crossing time
            frac = -x_w[i - 1] / (x_w[i] - x_w[i - 1])
            t_cross = t_w[i - 1] + frac * (t_w[i] - t_w[i - 1])
            bounce_times.append(t_cross)

    bounce_times = np.array(bounce_times)
    intervals = np.diff(bounce_times) if len(bounce_times) > 1 else np.array([])
    return bounce_times, len(bounce_times), intervals


def wing_asymmetry(t, x, tau_start=10.0, tau_end=60.0):
    """Compute wing-occupation asymmetry P(x>0) − P(x<0).

    Uses trapezoidal time-weighted fraction.
    """
    mask = (t >= tau_start) & (t <= tau_end)
    t_w = t[mask]
    x_w = x[mask]

    dt = np.diff(t_w)
    x_mid = 0.5 * (x_w[:-1] + x_w[1:])
    T = t_w[-1] - t_w[0]

    p_pos = np.sum(dt[x_mid > 0]) / T
    p_neg = np.sum(dt[x_mid < 0]) / T
    return p_pos - p_neg


def single_trajectory_diagnostics(state0, sigma, beta, rho, kappa, tau_m,
                                  T_total=60.0, tau_start=10.0, tau_end=60.0,
                                  dt_max=0.01, rtol=1e-9, atol=1e-12):
    """Run a single trajectory and extract all bounce/asymmetry diagnostics.

    Returns
    -------
    result : dict
        Keys: asymmetry, bounce_count, mean_interval, std_interval.
    """
    sol = integrate_trajectory(state0, sigma, beta, rho, kappa, tau_m,
                               T_total=T_total, dt_max=dt_max,
                               rtol=rtol, atol=atol, dense=False)
    t = sol.t
    x = sol.y[0]

    asym = wing_asymmetry(t, x, tau_start, tau_end)
    _, n_bounces, intervals = detect_bounces(t, x, tau_start, tau_end)

    return dict(
        asymmetry=asym,
        bounce_count=n_bounces,
        mean_interval=np.mean(intervals) if len(intervals) > 0 else np.nan,
        std_interval=np.std(intervals, ddof=1) if len(intervals) > 1 else np.nan,
    )
