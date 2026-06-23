#!/usr/bin/env python3
"""
Behavioral filter-bank simulation built on the quasicrystal-plate FEM.

================================================================
READ THIS FIRST -- what is real here, and what is assumed.
================================================================

This wires the REAL FEM center frequencies (from
fem_plate_bending_homogenized.py -- the exact element used for every
result in the MEMS paper) into a bank of behavioral bandpass filters,
and plots the resulting frequency response.

Each "channel" is a separate quasicrystal resonator tuned to a different
frequency by HOLE COVERAGE -- which is exactly the tuning knob the paper
found to be dominant. So the CENTER FREQUENCIES below are real,
physics-derived numbers, computed live by the same finite-element code.

Everything ELSE about the filter shape is a BEHAVIORAL ASSUMPTION, not
derived from the plate physics:

  * Q (sharpness / bandwidth) is ASSUMED, not computed. The FEM is
    undamped and has no loss model, so it CANNOT predict Q. The paper
    explicitly states quality factor was not computed.
  * Insertion loss, electrical impedance, and electromechanical
    transduction are NOT modeled (the FEM has no electrical port).
  * Inter-resonator coupling is NOT modeled (each channel is treated
    independently).

So this answers the honest question: "what shape does a filter bank
take, given these REAL center frequencies and an ASSUMED Q?" -- an
illustrative / behavioral model. It does NOT predict the real-world
performance of a fabricated quasicrystal filter. Every Q-dependent
number printed or plotted is conditional on the assumed Q, and is
labeled as such.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- import the real FEM (single source of truth, not copied) ---
FEM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plate_bending_review",
)
sys.path.insert(0, FEM_DIR)
from fem_plate_bending_homogenized import (  # noqa: E402
    Lx, Ly, build_mesh, element_coverage_fractions, assemble,
    clamped_free_dofs, solve_modes, debruijn_quasicrystal_points,
)

# ---- configuration ----
NX = 28
N_FOLD = 8                          # one symmetry order for the whole bank
SEED = 42
COVERAGES = [98, 94, 90, 85, 80]    # 5 channels, each tuned by coverage
ASSUMED_Q = 1000.0                  # <<< ASSUMPTION: not derived from plate physics.
#                                        Real MEMS Q ranges ~1e3 (modest) to ~1e7
#                                        (Li et al.'s device). 1e3 is conservative.


def find_radius_for_coverage(n_fold, target_cov, nx, seed, sub_n=12,
                             r_lo=0.2e-6, r_hi=15.0e-6, tol=0.4, max_iter=40):
    """Bisection-tune hole radius to a target coverage -- same protocol as the paper."""
    holes = debruijn_quasicrystal_points(n_fold, Lx, Ly, offset_seed=seed)
    nodes, quads = build_mesh(Lx, Ly, nx, nx)

    def cov_at(r):
        phi = element_coverage_fractions(nodes, quads, holes, r, sub_n=sub_n)
        return phi.mean() * 100

    lo, hi = r_lo, r_hi
    mid, cov = hi, cov_at(hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cov = cov_at(mid)
        if abs(cov - target_cov) <= tol:
            return mid, cov
        if cov > target_cov:
            lo = mid
        else:
            hi = mid
    return mid, cov


def fem_center_frequency(n_fold, radius, nx, seed):
    """Fundamental frequency from the real homogenized FEM (quadratic exponent)."""
    holes = debruijn_quasicrystal_points(n_fold, Lx, Ly, offset_seed=seed)
    nodes, quads = build_mesh(Lx, Ly, nx, nx)
    phi = element_coverage_fractions(nodes, quads, holes, radius, sub_n=12)
    K, M = assemble(nodes, quads, phi=phi, stiffness_exponent=2.0)
    free = clamped_free_dofs(nodes)
    freqs = solve_modes(K, M, free)
    return freqs[0]


def bandpass_response_db(f, f0, Q):
    """Driven damped-harmonic-oscillator magnitude response, normalized to its own
    peak (0 dB at center). This is the BEHAVIORAL part -- a 2nd-order bandpass
    shape with bandwidth set by the ASSUMED Q, NOT derived from the plate."""
    w = 2 * np.pi * f
    w0 = 2 * np.pi * f0
    H = 1.0 / np.sqrt((w0**2 - w**2)**2 + (w0 * w / Q)**2)
    H_peak = 1.0 / (w0**2 / Q)        # |H| at w = w0
    return 20 * np.log10(H / H_peak)


def main():
    print("=" * 64)
    print("Behavioral quasicrystal filter bank")
    print("  center frequencies: REAL (live FEM)")
    print(f"  filter Q = {ASSUMED_Q:.0f}: ASSUMED (not derived from plate physics)")
    print("=" * 64)

    print("\n=== Step 1: center frequencies from the real FEM (coverage-tuned) ===")
    channels = []
    for cov_t in COVERAGES:
        r, cov = find_radius_for_coverage(N_FOLD, cov_t, NX, SEED)
        f0 = fem_center_frequency(N_FOLD, r, NX, SEED)
        channels.append((cov, f0))
        print(f"  coverage ~{cov:5.1f}%  ->  f0 = {f0/1e6:.5f} MHz   (FEM, real)")

    print(f"\n=== Step 2: behavioral bandpass per channel (ASSUMED Q = {ASSUMED_Q:.0f}) ===")
    print(f"{'channel':>8} {'coverage%':>10} {'center (MHz)':>14} {'-3dB BW (kHz)':>15}")
    for i, (cov, f0) in enumerate(channels):
        bw = f0 / ASSUMED_Q   # high-Q bandwidth approximation
        print(f"{i:>8} {cov:>10.1f} {f0/1e6:>14.5f} {bw/1e3:>15.3f}")

    # ---- frequency-response plot ----
    fmin = min(f for _, f in channels) * 0.985
    fmax = max(f for _, f in channels) * 1.015
    fsweep = np.linspace(fmin, fmax, 6000)
    plt.figure(figsize=(10, 5.5))
    for i, (cov, f0) in enumerate(channels):
        plt.plot(fsweep / 1e6, bandpass_response_db(fsweep, f0, ASSUMED_Q),
                 lw=1.4, label=f"ch{i}: {cov:.0f}% cov, {f0/1e6:.4f} MHz")
        plt.axvline(f0 / 1e6, color="gray", lw=0.4, alpha=0.4)
    plt.axhline(-3, ls="--", c="0.5", lw=0.8, label="-3 dB")
    plt.ylim(-40, 4)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized response (dB)")
    plt.title("BEHAVIORAL quasicrystal filter bank\n"
              f"Center freqs: REAL (FEM, n={N_FOLD}-fold, coverage-tuned)   |   "
              f"Q = {ASSUMED_Q:.0f}: ASSUMED, NOT from plate physics   |   "
              "no loss/transduction/coupling modeled")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filter_bank_response.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")

    print("\n" + "-" * 64)
    print("HONEST REMINDER:")
    print("  Center frequencies are REAL (from the same FEM as the paper).")
    print("  Filter shape / Q / bandwidth are an ASSUMED behavioral model.")
    print("  This shows what a bank WOULD look like, given an assumed Q --")
    print("  it does NOT predict a real fabricated device's performance.")
    print("  Not modeled: energy loss / actual Q, insertion loss, electrical")
    print("  transduction, inter-resonator coupling.")
    print("-" * 64)


if __name__ == "__main__":
    main()
