# QBC: Non-Markovian Extended Lorenz System — Simulation Code

Reproducibility code for:

> **Ko, C.** (2026). "A Minimal Non-Markovian Lorenz-Like Model for a Quantum Bounce Core: Dynamical Diagnostics for the Cosmic Dipole Anomaly." *Chaos, Solitons & Fractals*.

## Model

The 4D autonomous ODE system (Eq. 4):

```
ẋ = σ(y − x)
ẏ = x(ρ − z) − y
ż = xy − βz + κw
ẇ = (xy − w) / τ_m
```

where `w` is the Volterra-type memory variable with exponential decay kernel. Setting `κ = 0` recovers the classical Lorenz system.

## Quick Start

```bash
pip install -r requirements.txt

# Reproduce all manuscript results
python run_all.py

# Fast test run (reduced resolution)
python run_all.py --quick

# Individual components
python run_all.py --table1        # Hopf boundary ρ_H(κ, τ_m)
python run_all.py --table2        # Lyapunov exponents λ₁(ρ, κ)
python run_all.py --ensemble      # Ensemble validation (N = 1000)
python run_all.py --bifurcation   # Bifurcation diagrams
python run_all.py --poincare      # Poincaré sections
```

## Repository Structure

```
qbc_model.py        Core ODE, Jacobian, equilibria, Hopf boundary
config.py           All numerical settings (Appendix A.1–A.11)
lyapunov.py         Benettin method for Lyapunov spectrum
bifurcation.py      ρ-sweep bifurcation diagram generation
bounce_stats.py     Bounce detection, wing-occupation asymmetry
ensemble.py         N = 1000 ensemble validation protocol
poincare.py         Poincaré sections (z = 25, upward crossings)
run_all.py          Master script reproducing all figures/tables
requirements.txt    Python dependencies
```

## Numerical Protocol (Appendix A)

| Setting | Value |
|---------|-------|
| Integrator | RK45, Δτ = 0.01, rtol = 10⁻⁹, atol = 10⁻¹² |
| Initial conditions | (x, y, z, w) = (1, 1, 1, 0) |
| Transient removal | T_tr = 50 |
| Bifurcation sweep | ρ ∈ [0, 60], Δρ = 0.2, T = 250, discard 50% |
| Poincaré section | z = 25, upward crossings |
| Lyapunov (Benettin) | d₀ = 10⁻⁸, T = 150, T_tr = 30, renorm Δτ = 0.5 |
| Bounce detection | neg→pos zero-crossings of x |
| Ensemble | N = 1000, ε = 0.01, seed = 42, w(0) = 0 fixed |

## Output

Results are saved to `results/` (JSON) and `figures/` (PNG + PDF).

### Figures

| File | Manuscript |
|------|-----------|
| `fig4_bifurcation_k0` | Fig. 4 |
| `fig4a_bifurcation_3panel` | Fig. 4a |
| `fig4b_overlay_lyapunov` | Fig. 4b |
| `fig5_lyapunov` | Fig. 5 |
| `fig6_timeseries` | Fig. 6 |
| `fig7_hopf_boundary` | Fig. 7 |
| `fig8_ensemble_histograms` | Fig. 8 |
| `poincare_sections` | Supplementary |

### Data

| File | Content |
|------|---------|
| `table1_hopf.json` | Table 1: ρ_H(κ, τ_m) |
| `table2_lyapunov.json` | Table 2: λ₁(ρ, κ) |
| `table3_ensemble.json` | Table 3: Ensemble statistics |

## Runtime

Full run: approximately 20–30 minutes on a modern CPU (single core).
Quick mode (`--quick`): approximately 3–5 minutes.

## License

MIT License. See LICENSE file.
