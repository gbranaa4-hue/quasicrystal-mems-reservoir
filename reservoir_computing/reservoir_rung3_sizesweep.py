#!/usr/bin/env python3
"""
RUNG 3 follow-through: does the quasicrystal's degeneracy-breaking help when
reservoir capacity is SCARCE?

MOTIVATED HYPOTHESIS (not fishing): rung2_3 found the spectra perform
identically at 40 modes, BUT the periodic plate has many more near-degenerate
(redundant) modes (18 vs 3). Degenerate modes waste capacity, and wasted
capacity hurts most when you have LITTLE of it. So the quasicrystal advantage,
IF it exists, should appear at SMALL reservoir size and vanish at large size
(where even the periodic plate has plenty of distinct modes to spare).

This sweeps reservoir size N and races the two real FEM spectra at each size.
A clean prediction with a mechanism -- and an honest test: if quasicrystal
does NOT pull ahead at small N either, then the mode distribution simply
does not matter for this kind of computing, which is itself the finding.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_rung2_3 import (  # noqa: E402
    mode_spectrum, coverage_match_radius, generate_periodic_holes,
    normalize_spectrum, build_coupling, run_reservoir, eval_task,
    NX, N_MODES, TARGET_COV, QC_NFOLD, QC_SEED, OMEGA_LO, OMEGA_HI,
    INPUT_AMP, L, BETA2, BETA3, RES_SEEDS, Lx, Ly,
    debruijn_quasicrystal_points,
)

SIZES = [6, 8, 10, 12, 16, 20, 28, 40]


def near_deg(spec, tol_frac=0.005):
    s = np.sort(spec); gaps = np.diff(s) / s[:-1]
    return int(np.sum(gaps < tol_frac))


def bench_at_size(full_spec, N, u, yA):
    spec = np.sort(full_spec)[:N]
    omega = normalize_spectrum(spec, OMEGA_LO, OMEGA_HI)
    nl, mc = [], []
    for s in RES_SEEDS:
        w_in, C = build_coupling(N, s)
        states = run_reservoir(omega, w_in, C, u, BETA2, BETA3)
        nl.append(eval_task(states, yA))
        m = 0.0
        for k in range(1, 16):
            yk = np.zeros(len(u)); yk[k:] = u[:len(u) - k]
            m += max(0.0, eval_task(states, yk))
        mc.append(m)
    return np.array(nl), np.array(mc)


def main():
    print("=" * 70)
    print("RUNG 3 -- does quasicrystal help when modes are SCARCE? (size sweep)")
    print("=" * 70)

    # real FEM spectra (compute the full 40-mode spectra once, then slice)
    print(f"\nComputing real FEM mode spectra ({N_MODES} modes, {TARGET_COV}% coverage)...")
    qc_holes = debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED)
    qc_r, _ = coverage_match_radius(qc_holes, TARGET_COV, NX)
    qc_spec = mode_spectrum(qc_holes, qc_r, N_MODES, NX)

    per_holes = generate_periodic_holes(9, Lx, Ly)
    per_r, _ = coverage_match_radius(per_holes, TARGET_COV, NX)
    per_spec = mode_spectrum(per_holes, per_r, N_MODES, NX)
    print("done.\n")

    rng = np.random.default_rng(0)
    u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)
    yA = np.zeros(L); yA[2:] = u[1:L-1] * u[0:L-2]

    print(f"{'N':>4} {'QC deg':>7} {'per deg':>8} | "
          f"{'QC nl R^2':>12} {'per nl R^2':>12} | {'QC mem':>8} {'per mem':>8} | winner")
    print("-" * 88)
    rows = []
    for N in SIZES:
        qc_nl, qc_mc = bench_at_size(qc_spec, N, u, yA)
        per_nl, per_mc = bench_at_size(per_spec, N, u, yA)
        qd, pd = near_deg(np.sort(qc_spec)[:N]), near_deg(np.sort(per_spec)[:N])
        d = qc_nl.mean() - per_nl.mean()
        pooled = np.hypot(qc_nl.std(), per_nl.std()) + 1e-9
        sig = abs(d) / pooled
        win = ("QC" if d > 0 else "periodic") if sig >= 1.0 else "tie"
        rows.append((N, qd, pd, qc_nl.mean(), qc_nl.std(), per_nl.mean(), per_nl.std(),
                     qc_mc.mean(), per_mc.mean(), sig, win))
        print(f"{N:>4} {qd:>7} {pd:>8} | {qc_nl.mean():>7.3f}+/-{qc_nl.std():<4.3f} "
              f"{per_nl.mean():>7.3f}+/-{per_nl.std():<4.3f} | "
              f"{qc_mc.mean():>8.2f} {per_mc.mean():>8.2f} | {win} ({sig:.1f}x)")

    rows = np.array([(r[0], r[3], r[4], r[5], r[6], r[7], r[8]) for r in rows])

    # ---- plot ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], fmt="o-", color="#2E5E8C",
                capsize=3, label="quasicrystal")
    a1.errorbar(rows[:, 0], rows[:, 3], yerr=rows[:, 4], fmt="s-", color="#C0392B",
                capsize=3, label="periodic")
    a1.set_title("Nonlinear-task performance vs reservoir size")
    a1.set_xlabel("reservoir size N (number of modes)"); a1.set_ylabel("nonlinear task R^2")
    a1.legend(); a1.grid(alpha=0.3)

    a2.plot(rows[:, 0], rows[:, 5], "o-", color="#2E5E8C", label="quasicrystal")
    a2.plot(rows[:, 0], rows[:, 6], "s-", color="#C0392B", label="periodic")
    a2.set_title("Memory capacity vs reservoir size")
    a2.set_xlabel("reservoir size N (number of modes)"); a2.set_ylabel("memory capacity")
    a2.legend(); a2.grid(alpha=0.3)

    fig.suptitle("Quasicrystal vs periodic reservoir, vs size "
                 "(real FEM spectra; testing the scarce-capacity hypothesis)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reservoir_rung3_sizesweep.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved {out}")

    # ---- honest verdict ----
    small = [r for r in rows if r[0] <= 12]
    qc_better_small = np.mean([r[1] - r[3] for r in small])
    print("\n" + "=" * 70)
    print("HONEST VERDICT (rung 3)")
    print("=" * 70)
    print(f"  Mean QC-minus-periodic nonlinear R^2 at small N (<=12): {qc_better_small:+.3f}")
    print("  If positive and beyond the error bars at small N (and shrinking at large N),")
    print("  the quasicrystal's degeneracy-breaking gives a real edge when modes are")
    print("  scarce. If it's ~0 everywhere, the mode DISTRIBUTION does not matter for")
    print("  this kind of computing -- a clean negative result, equally honest.")
    print("  Either way: REAL frequencies, GENERIC identical nonlinearity, isolating the")
    print("  spectrum's effect -- not a fabricated-device claim.")
    print("=" * 70)


if __name__ == "__main__":
    main()
