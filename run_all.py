#!/usr/bin/env python3
"""
run_all.py — Reproduce All Figures and Tables
==============================================

This script regenerates every numerical result in the manuscript:

  - Table 1 : Hopf boundary ρ_H(κ, τ_m)
  - Table 2 : Maximal Lyapunov exponent λ₁(ρ, κ)
  - Table 2a: Lyapunov comparison at ρ = 24
  - Table 3 : Ensemble-averaged diagnostics (N = 1000)
  - Fig. 4  : Bifurcation diagram (κ = 0)
  - Fig. 4a : 3-panel bifurcation (κ = 0, 0.5, 1.0)
  - Fig. 4b : Overlay bifurcation + Lyapunov
  - Fig. 5  : Lyapunov exponent vs ρ
  - Fig. 6  : Time series at ρ = 28
  - Fig. 7  : Hopf boundary in (κ, τ_m) space
  - Fig. 8  : Ensemble distributions
  - Poincaré sections (supplementary)

Usage
-----
    python run_all.py                # run everything
    python run_all.py --quick        # fast mode (reduced resolution)
    python run_all.py --table1       # only Table 1
    python run_all.py --ensemble     # only ensemble (Table 3 + Fig. 8)

Output
------
    figures/    PNG and PDF files
    results/    JSON data files

Estimated runtime (full): ~25 min on a single core (Apple M2).
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# Local modules
from qbc_model import qbc_rhs, equilibria, hopf_boundary_rh, DEFAULT_PARAMS
from config import QBCConfig
from lyapunov import maximal_lyapunov, lyapunov_spectrum
from bifurcation import bifurcation_sweep
from bounce_stats import (integrate_trajectory, detect_bounces,
                          wing_asymmetry, single_trajectory_diagnostics)
from ensemble import run_ensemble, ensemble_summary
from poincare import poincare_section

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGDIR = "figures"
RESDIR = "results"


def ensure_dirs():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)


# ===================================================================
# Table 1: Hopf boundary
# ===================================================================
def table1(cfg):
    """ρ_H(κ, τ_m) via Routh–Hurwitz (Table 1)."""
    print("\n=== Table 1: Hopf Boundary ρ_H(κ, τ_m) ===")
    kappas = [0.0, 0.5, 1.0]
    tau_ms = [0.5, 1.0, 2.0, 5.0]
    header = f"{'τ_m':>6}" + "".join(f"{'κ='+str(k):>12}" for k in kappas)
    print(header)
    print("-" * len(header))
    data = {}
    for tm in tau_ms:
        row = f"{tm:6.1f}"
        for k in kappas:
            rh = hopf_boundary_rh(k, tm, cfg.sigma, cfg.beta)
            row += f"{rh:12.2f}"
            data[f"tm{tm}_k{k}"] = rh
        print(row)
    with open(f"{RESDIR}/table1_hopf.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


# ===================================================================
# Table 2: Lyapunov exponents
# ===================================================================
def table2(cfg):
    """λ₁(ρ, κ) for τ_m = 1.0 (Table 2)."""
    print("\n=== Table 2: Maximal Lyapunov Exponent λ₁ ===")
    rhos = [22, 24, 26, 28]
    kappas = [0.0, 0.5, 1.0]
    state0 = cfg.ic.state0
    header = f"{'ρ':>4}" + "".join(f"{'κ='+str(k):>12}" for k in kappas)
    print(header)
    print("-" * len(header))
    data = {}
    for rho in rhos:
        row = f"{rho:4d}"
        for k in kappas:
            lam = maximal_lyapunov(cfg.sigma, cfg.beta, rho, k, 1.0,
                                   state0,
                                   T_total=cfg.lyapunov.T_total,
                                   T_transient=cfg.lyapunov.T_transient,
                                   renorm_interval=cfg.lyapunov.renorm_interval)
            row += f"{lam:12.4f}"
            data[f"rho{rho}_k{k}"] = round(lam, 4)
        print(row)
    with open(f"{RESDIR}/table2_lyapunov.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


# ===================================================================
# Table 2a: Lyapunov comparison at ρ = 24
# ===================================================================
def table2a(cfg):
    """Detailed Lyapunov comparison at ρ = 24 (Table 2a)."""
    print("\n=== Table 2a: Lyapunov at ρ = 24 ===")
    kappas = [0.0, 0.5, 1.0]
    state0 = cfg.ic.state0
    for k in kappas:
        rh = hopf_boundary_rh(k, 1.0, cfg.sigma, cfg.beta)
        lam = maximal_lyapunov(cfg.sigma, cfg.beta, 24.0, k, 1.0,
                               state0,
                               T_total=cfg.lyapunov.T_total,
                               T_transient=cfg.lyapunov.T_transient,
                               renorm_interval=cfg.lyapunov.renorm_interval)
        regime = "chaotic" if lam > 0.01 else ("marginal" if lam > -0.01 else "stable")
        print(f"  κ={k:.1f}  ρ_H={rh:.2f}  λ₁={lam:+.4f}  [{regime}]")


# ===================================================================
# Bifurcation diagrams (Figs. 4, 4a, 4b)
# ===================================================================
def fig4_bifurcation(cfg, quick=False):
    """Bifurcation diagrams."""
    print("\n=== Figs. 4, 4a, 4b: Bifurcation Diagrams ===")
    d_rho = 0.5 if quick else cfg.bifurcation.d_rho
    T_int = 150.0 if quick else cfg.bifurcation.T_integrate

    kappas = [0.0, 0.5, 1.0]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    labels = ["κ = 0 (Markovian)", "κ = 0.5", "κ = 1.0"]

    all_data = {}

    # --- Fig. 4: κ = 0 only ---
    print("  Fig. 4 (κ = 0)...")
    rho_0, xmax_0 = bifurcation_sweep(
        cfg.sigma, cfg.beta, 0.0, 1.0,
        rho_max=cfg.bifurcation.rho_max, d_rho=d_rho,
        T_integrate=T_int, dt_max=cfg.bifurcation.dt_max,
        verbose=False)
    all_data["k0.0"] = (rho_0, xmax_0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(rho_0, xmax_0, s=0.1, c=colors[0], alpha=0.4, rasterized=True)
    ax.set_xlabel("ρ")
    ax.set_ylabel("Local maxima of x(τ)")
    ax.set_title("Bifurcation diagram (κ = 0)")
    ax.set_xlim(0, 60)
    fig.savefig(f"{FIGDIR}/fig4_bifurcation_k0.png")
    fig.savefig(f"{FIGDIR}/fig4_bifurcation_k0.pdf")
    plt.close(fig)

    # --- Fig. 4a: 3-panel ---
    print("  Fig. 4a (3-panel)...")
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    for j, k in enumerate(kappas):
        if k == 0.0:
            rho_j, xmax_j = rho_0, xmax_0
        else:
            rho_j, xmax_j = bifurcation_sweep(
                cfg.sigma, cfg.beta, k, 1.0,
                rho_max=cfg.bifurcation.rho_max, d_rho=d_rho,
                T_integrate=T_int, dt_max=cfg.bifurcation.dt_max,
                verbose=False)
            all_data[f"k{k}"] = (rho_j, xmax_j)
        rh = hopf_boundary_rh(k, 1.0, cfg.sigma, cfg.beta)
        axes[j].scatter(rho_j, xmax_j, s=0.15, c=colors[j],
                        alpha=0.4, rasterized=True)
        axes[j].axvline(rh, color="gray", ls="--", lw=0.8,
                        label=f"ρ_H = {rh:.1f}")
        axes[j].set_ylabel("x_max")
        axes[j].set_title(labels[j])
        axes[j].legend(fontsize=8)
    axes[-1].set_xlabel("ρ")
    axes[-1].set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig4a_bifurcation_3panel.png")
    fig.savefig(f"{FIGDIR}/fig4a_bifurcation_3panel.pdf")
    plt.close(fig)

    # --- Fig. 4b: Overlay + Lyapunov ---
    print("  Fig. 4b (overlay + Lyapunov)...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    for j, k in enumerate(kappas):
        key = f"k{k}"
        rho_j, xmax_j = all_data.get(key, (rho_0, xmax_0))
        ax1.scatter(rho_j, xmax_j, s=0.1, c=colors[j], alpha=0.3,
                    label=labels[j], rasterized=True)
    ax1.set_ylabel("Local maxima of x(τ)")
    ax1.legend(fontsize=8, markerscale=10)

    # Lyapunov exponent vs ρ
    rho_lyap = np.arange(2, 60.1, 2.0 if quick else 1.0)
    state0 = cfg.ic.state0
    for j, k in enumerate(kappas):
        lams = []
        for rho in rho_lyap:
            lam = maximal_lyapunov(
                cfg.sigma, cfg.beta, rho, k, 1.0, state0,
                T_total=80.0 if quick else cfg.lyapunov.T_total,
                T_transient=20.0 if quick else cfg.lyapunov.T_transient,
                renorm_interval=cfg.lyapunov.renorm_interval)
            lams.append(lam)
        ax2.plot(rho_lyap, lams, color=colors[j], lw=1.2, label=labels[j])
    ax2.axhline(0, color="gray", ls=":", lw=0.7)
    ax2.set_xlabel("ρ")
    ax2.set_ylabel("λ₁")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig4b_overlay_lyapunov.png")
    fig.savefig(f"{FIGDIR}/fig4b_overlay_lyapunov.pdf")
    plt.close(fig)


# ===================================================================
# Fig. 5: Lyapunov exponent vs ρ
# ===================================================================
def fig5_lyapunov(cfg, quick=False):
    """λ₁ vs ρ (κ = 0)."""
    print("\n=== Fig. 5: Lyapunov Exponent vs ρ ===")
    rhos = np.arange(2, 60.1, 2.0 if quick else 0.5)
    state0 = cfg.ic.state0
    lams = []
    for rho in rhos:
        lam = maximal_lyapunov(cfg.sigma, cfg.beta, rho, 0.0, 1.0,
                               state0,
                               T_total=cfg.lyapunov.T_total,
                               T_transient=cfg.lyapunov.T_transient,
                               renorm_interval=cfg.lyapunov.renorm_interval)
        lams.append(lam)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rhos, lams, "b-", lw=1)
    ax.axhline(0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel("ρ")
    ax.set_ylabel("λ₁")
    ax.set_title("Maximal Lyapunov exponent (κ = 0)")
    fig.savefig(f"{FIGDIR}/fig5_lyapunov.png")
    fig.savefig(f"{FIGDIR}/fig5_lyapunov.pdf")
    plt.close(fig)


# ===================================================================
# Fig. 6: Time series
# ===================================================================
def fig6_timeseries(cfg):
    """x(τ) time series at ρ = 28."""
    print("\n=== Fig. 6: Time Series ===")
    state0 = cfg.ic.state0
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    kappas = [0.0, 0.5, 1.0]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for i, k in enumerate(kappas):
        sol = integrate_trajectory(state0, cfg.sigma, cfg.beta, 28.0, k, 1.0,
                                   T_total=80.0)
        axes[i].plot(sol.t, sol.y[0], color=colors[i], lw=0.3)
        axes[i].set_ylabel("x(τ)")
        axes[i].set_title(f"κ = {k}")
        axes[i].axhline(0, color="gray", ls=":", lw=0.5)
    axes[-1].set_xlabel("τ")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig6_timeseries.png")
    fig.savefig(f"{FIGDIR}/fig6_timeseries.pdf")
    plt.close(fig)


# ===================================================================
# Fig. 7: Hopf boundary
# ===================================================================
def fig7_hopf(cfg):
    """Hopf boundary ρ_H in (τ_m, κ) space (Fig. 7)."""
    print("\n=== Fig. 7: Hopf Boundary ===")
    tau_ms = np.linspace(0.2, 5.0, 80)
    kappas = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in kappas:
        rhs = [hopf_boundary_rh(k, tm, cfg.sigma, cfg.beta) for tm in tau_ms]
        ax.plot(tau_ms, rhs, lw=1.5, label=f"κ = {k}")
    ax.set_xlabel("τ_m")
    ax.set_ylabel("ρ_H")
    ax.set_title("Hopf boundary")
    ax.legend()
    fig.savefig(f"{FIGDIR}/fig7_hopf_boundary.png")
    fig.savefig(f"{FIGDIR}/fig7_hopf_boundary.pdf")
    plt.close(fig)


# ===================================================================
# Table 3 + Fig. 8: Ensemble
# ===================================================================
def table3_and_fig8(cfg, quick=False):
    """Ensemble-averaged diagnostics (Table 3 + Fig. 8)."""
    print("\n=== Table 3 + Fig. 8: Ensemble (N = 1000) ===")
    N = 100 if quick else cfg.ensemble.N_ens
    kappas = [0.0, 0.5, 1.0]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    all_results = {}

    for k in kappas:
        print(f"  κ = {k} ...")
        res = run_ensemble(
            cfg.sigma, cfg.beta, 28.0, k, 1.0,
            N_ens=N, epsilon=cfg.ensemble.epsilon,
            seed=cfg.ensemble.seed,
            T_total=60.0, tau_start=10.0, tau_end=60.0,
            verbose=False)
        all_results[k] = res
        summ = ensemble_summary(res)
        print(f"    Asymmetry:  {summ['asymmetry']['mean']:+.3f} ± "
              f"{summ['asymmetry']['std']:.3f}")
        print(f"    Bounces:    {summ['bounce_count']['mean']:.1f} ± "
              f"{summ['bounce_count']['std']:.1f}")
        print(f"    Mean int.:  {summ['mean_interval']['mean']:.2f} ± "
              f"{summ['mean_interval']['std']:.2f}")
        print(f"    Std int.:   {summ['std_interval']['mean']:.2f} ± "
              f"{summ['std_interval']['std']:.2f}")

    # Save JSON
    json_data = {}
    for k, res in all_results.items():
        summ = ensemble_summary(res)
        json_data[f"kappa_{k}"] = {
            key: {sk: round(sv, 4) for sk, sv in sval.items()}
            for key, sval in summ.items()
        }
    with open(f"{RESDIR}/table3_ensemble.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # Fig. 8: Histograms
    diag_keys = ["asymmetry", "bounce_count", "mean_interval", "std_interval"]
    diag_labels = [
        "Wing asymmetry P(x>0)−P(x<0)",
        "Bounce count",
        "Mean inter-bounce interval",
        "Std of inter-bounce intervals",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for j, (key, label) in enumerate(zip(diag_keys, diag_labels)):
        ax = axes[j]
        for ki, k in enumerate(kappas):
            vals = all_results[k][key]
            valid = vals[~np.isnan(vals)]
            ax.hist(valid, bins=30, alpha=0.45, color=colors[ki],
                    label=f"κ = {k}", density=True, edgecolor="none")
            ax.axvline(np.mean(valid), color=colors[ki], ls="--", lw=1.2)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle(f"Ensemble distributions (N = {N}, ρ = 28, τ_m = 1.0)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIGDIR}/fig8_ensemble_histograms.png")
    fig.savefig(f"{FIGDIR}/fig8_ensemble_histograms.pdf")
    plt.close(fig)


# ===================================================================
# Poincaré sections (supplementary)
# ===================================================================
def poincare_plots(cfg, quick=False):
    """Poincaré sections at z = 25 for each κ."""
    print("\n=== Poincaré Sections ===")
    kappas = [0.0, 0.5, 1.0]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    T = 150.0 if quick else 300.0
    state0 = cfg.ic.state0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, k in enumerate(kappas):
        xc, yc = poincare_section(cfg.sigma, cfg.beta, 28.0, k, 1.0,
                                  state0, T_total=T)
        axes[i].scatter(xc, yc, s=0.5, c=colors[i], alpha=0.5)
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].set_title(f"κ = {k}")
    fig.suptitle("Poincaré section (z = 25, ρ = 28)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{FIGDIR}/poincare_sections.png")
    fig.savefig(f"{FIGDIR}/poincare_sections.pdf")
    plt.close(fig)


# ===================================================================
# CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all QBC manuscript results.")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced resolution (faster, for testing)")
    parser.add_argument("--table1", action="store_true")
    parser.add_argument("--table2", action="store_true")
    parser.add_argument("--table2a", action="store_true")
    parser.add_argument("--bifurcation", action="store_true")
    parser.add_argument("--lyapunov", action="store_true")
    parser.add_argument("--timeseries", action="store_true")
    parser.add_argument("--hopf", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--poincare", action="store_true")
    args = parser.parse_args()

    cfg = QBCConfig()
    ensure_dirs()

    run_specific = any([args.table1, args.table2, args.table2a,
                        args.bifurcation, args.lyapunov, args.timeseries,
                        args.hopf, args.ensemble, args.poincare])
    t0 = time.time()

    if not run_specific or args.table1:
        table1(cfg)
    if not run_specific or args.table2:
        table2(cfg)
    if not run_specific or args.table2a:
        table2a(cfg)
    if not run_specific or args.bifurcation:
        fig4_bifurcation(cfg, quick=args.quick)
    if not run_specific or args.lyapunov:
        fig5_lyapunov(cfg, quick=args.quick)
    if not run_specific or args.timeseries:
        fig6_timeseries(cfg)
    if not run_specific or args.hopf:
        fig7_hopf(cfg)
    if not run_specific or args.ensemble:
        table3_and_fig8(cfg, quick=args.quick)
    if not run_specific or args.poincare:
        poincare_plots(cfg, quick=args.quick)

    elapsed = time.time() - t0
    print(f"\nDone. Total time: {elapsed:.1f} s")
    print(f"Figures saved to {FIGDIR}/")
    print(f"Data saved to {RESDIR}/")


if __name__ == "__main__":
    main()
