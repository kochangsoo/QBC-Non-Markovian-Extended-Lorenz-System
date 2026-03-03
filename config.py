"""
Configuration — Appendix A.1–A.11 Reproducibility Protocol
===========================================================

All numerical settings are centralised here.  Modify only this file
to explore alternative parameter regimes.
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class IntegratorConfig:
    """A.1  Integrator settings."""
    method: str = "RK45"
    dt_max: float = 0.01          # maximum step size
    rtol: float = 1e-9
    atol: float = 1e-12


@dataclass
class ICConfig:
    """A.2  Initial conditions."""
    x0: float = 1.0
    y0: float = 1.0
    z0: float = 1.0
    w0: float = 0.0               # initially unaccumulated memory
    T_transient: float = 50.0     # transient removal window

    @property
    def state0(self) -> np.ndarray:
        return np.array([self.x0, self.y0, self.z0, self.w0])


@dataclass
class BifurcationConfig:
    """A.3 / A.11  Bifurcation sweep."""
    rho_min: float = 0.0
    rho_max: float = 60.0
    d_rho: float = 0.2            # A.11 resolution
    T_integrate: float = 250.0    # A.11
    discard_frac: float = 0.50    # discard first 50%
    max_peaks_per_rho: int = 150  # A.11
    dt_max: float = 0.02          # A.11


@dataclass
class PoincareConfig:
    """A.4  Poincaré section."""
    z_section: float = 25.0       # z = 25, upward crossings


@dataclass
class LyapunovConfig:
    """A.5  Maximal Lyapunov exponent (Benettin method)."""
    d0: float = 1e-8              # initial perturbation norm
    T_total: float = 150.0        # A.11
    T_transient: float = 30.0     # A.11
    renorm_interval: float = 0.5  # time units between renormalisations


@dataclass
class BounceConfig:
    """A.6  Bounce detection."""
    tau_start: float = 10.0       # start of analysis window
    tau_end: float = 60.0         # end of analysis window


@dataclass
class SweepConfig:
    """A.8  Parameter sweep ranges."""
    kappa_values: Tuple[float, ...] = (0.0, 0.5, 1.0)
    tau_m_values: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)


@dataclass
class EnsembleConfig:
    """A.10  Ensemble protocol."""
    N_ens: int = 1000
    epsilon: float = 0.01         # IC perturbation radius
    seed: int = 42
    w0_fixed: float = 0.0         # w(0) held fixed


@dataclass
class QBCConfig:
    """Master configuration container."""
    # Physical parameters
    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    rho: float = 28.0
    kappa: float = 0.0
    tau_m: float = 1.0

    # Sub-configs
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    ic: ICConfig = field(default_factory=ICConfig)
    bifurcation: BifurcationConfig = field(default_factory=BifurcationConfig)
    poincare: PoincareConfig = field(default_factory=PoincareConfig)
    lyapunov: LyapunovConfig = field(default_factory=LyapunovConfig)
    bounce: BounceConfig = field(default_factory=BounceConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    @property
    def params(self) -> dict:
        return dict(sigma=self.sigma, beta=self.beta, rho=self.rho,
                    kappa=self.kappa, tau_m=self.tau_m)
