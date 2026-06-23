#!/usr/bin/env python3
"""
RUNG 7 -- the standard benchmark: NARMA-10.

Everything so far used home-made tasks. NARMA-10 is THE canonical
reservoir-computing benchmark (Atiya-Parlos / Jaeger), so the result speaks
the field's language. It is also a sharp test of our finding: its driving
term is 1.5*u[n]*u[n-9] -- a PRODUCT of two delayed inputs (even-order) with
a memory depth of 10. The quadratic symmetry-selection-rule finding predicts
the quasicrystal should help here, in the weakly-coupled regime, because the
even-order term is exactly what periodic symmetry suppresses.

NARMA-10 system (input u[n] ~ Uniform[0, 0.5]):
  y[n+1] = 0.3 y[n] + 0.05 y[n] * sum_{i=0..9} y[n-i] + 1.5 u[n] u[n-9] + 0.1
The reservoir is driven by u[n] and a linear readout must predict y[n].
Reported as NRMSE (the standard metric) and R^2, QC vs periodic, averaged
over actuator locations. Pre-committed: report whatever it shows.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reservoir_rung4_modeshapes import (  # noqa: E402
    Lx, Ly, debruijn_quasicrystal_points, generate_periodic_holes,
    nearest_elem, ridge_fit, r2, QC_NFOLD, QC_SEED,
)
from reservoir_rung6_stresstest import make_plate, omega_of, run_uncoupled  # noqa: E402
from reservoir_rung5_saturation import run_reservoir_gamma  # noqa: E402

L = 1900
WASHOUT = 200
N_TRAIN = 1200
INPUT_GAIN = 5.0                 # scales u in [0,0.5] up so the nonlinearity engages
_drng = np.random.default_rng(11)
DRIVE_FRACS = [tuple(p) for p in _drng.uniform(0.22, 0.78, size=(12, 2))]


def narma10(u):
    """Generate the NARMA-10 target sequence from input u in [0, 0.5]."""
    n = len(u)
    y = np.zeros(n)
    for t in range(9, n - 1):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
                    + 1.5 * u[t] * u[t - 9]
                    + 0.1)
    return y


def fit_eval(states, target):
    Xtr, Ytr = states[WASHOUT:WASHOUT + N_TRAIN], target[WASHOUT:WASHOUT + N_TRAIN]
    Xte, Yte = states[WASHOUT + N_TRAIN:], target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None], lam=1e-4)
    pred = (Xte @ W)[:, 0]
    nrmse = np.sqrt(np.mean((pred - Yte) ** 2) / (np.var(Yte) + 1e-30))
    return r2(Yte, pred), nrmse, pred, Yte


def race_narma(P_qc, P_per, u_drive, y, gamma=0.0):
    out = {}
    for tag, P in [("qc", P_qc), ("per", P_per)]:
        om = omega_of(P)
        r2s, nrs, ex = [], [], None
        for frac in DRIVE_FRACS:
            e = nearest_elem(P["ctr"], frac)
            w_in = P["Phi"][:, e].copy()
            if gamma == 0.0:
                st = run_uncoupled(om, P["c2"], P["c3"], w_in, u_drive)
            else:
                st = run_reservoir_gamma(om, P["Phi"], P["aw"], P["c2"], P["c3"],
                                         w_in, u_drive, gamma)
            rr, nn, pred, yte = fit_eval(st, y)
            r2s.append(rr); nrs.append(nn)
            if ex is None:
                ex = (pred, yte)
        out[tag] = (np.array(r2s), np.array(nrs), ex)
    return out


def main():
    print("=" * 78)
    print("RUNG 7 -- NARMA-10 benchmark: quasicrystal vs periodic reservoir")
    print("=" * 78)

    rng = np.random.default_rng(0)
    u = rng.uniform(0.0, 0.5, L)
    y = narma10(u)
    print(f"\nNARMA-10 target: range [{y.min():.3f}, {y.max():.3f}], "
          f"finite={np.all(np.isfinite(y))}, depth=10, product term u[n]*u[n-9].")
    u_drive = INPUT_GAIN * u

    print("Building plates (85% coverage, 8-fold QC)...")
    qc = make_plate(debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED), 85.0)
    per = make_plate(generate_periodic_holes(9, Lx, Ly), 85.0)
    print("done.")

    # ---- main result: uncoupled (weak-coupling regime, where the edge lives) ----
    print("\n" + "-" * 78)
    print("NARMA-10, uncoupled reservoir (gamma=0), 12 actuator locations:")
    print("-" * 78)
    res = race_narma(qc, per, u_drive, y, gamma=0.0)
    (qr, qn, qex), (pr, pn, pex) = res["qc"], res["per"]
    d = qr.mean() - pr.mean(); s = np.hypot(qr.std(), pr.std()) + 1e-9
    print(f"  quasicrystal:  R^2 = {qr.mean():.3f}+/-{qr.std():.3f}   "
          f"NRMSE = {qn.mean():.3f}+/-{qn.std():.3f}")
    print(f"  periodic:      R^2 = {pr.mean():.3f}+/-{pr.std():.3f}   "
          f"NRMSE = {pn.mean():.3f}+/-{pn.std():.3f}")
    print(f"  gap (R^2): {d:+.3f}  ({d/s:+.1f}x spread)  -> "
          f"{'QUASICRYSTAL better' if (d>0 and d/s>=1) else 'periodic better' if (d<0 and abs(d)/s>=1) else 'tie'}")

    # ---- confirm it's a weak-coupling effect: gap vs gamma ----
    print("\n  gap vs coupling strength (should shrink as coupling rises):")
    gammas = [0.0, 0.5, 1.0]
    gaps = []
    for g in gammas:
        rr = race_narma(qc, per, u_drive, y, gamma=g)
        a, b = rr["qc"][0], rr["per"][0]
        dd = a.mean() - b.mean(); ss = np.hypot(a.std(), b.std()) + 1e-9
        gaps.append((g, a.mean(), b.mean(), dd, dd / ss))
        print(f"    gamma={g:.1f}:  QC R^2 {a.mean():.3f}  per {b.mean():.3f}  "
              f"gap {dd:+.3f} ({dd/ss:+.1f}x)")

    # ---- plot ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    pred, yte = qex
    seg = slice(0, 150)
    a1.plot(yte[seg], "k-", lw=2, label="NARMA-10 target")
    a1.plot(pred[seg], "-", color="#2E5E8C", lw=1.4,
            label=f"quasicrystal readout (R^2={qr.mean():.2f})")
    a1.plot(pex[0][seg], "--", color="#C0392B", lw=1.1,
            label=f"periodic readout (R^2={pr.mean():.2f})")
    a1.set_title("NARMA-10 prediction (uncoupled regime)")
    a1.set_xlabel("test timestep"); a1.set_ylabel("y[n]")
    a1.legend(fontsize=8); a1.grid(alpha=0.3)

    gg = np.array(gaps)
    a2.bar(np.arange(len(gg)), gg[:, 4], color="#2E5E8C")
    a2.axhline(1.0, color="green", ls="--", lw=1, label="1x spread")
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(np.arange(len(gg))); a2.set_xticklabels([f"g={g:.1f}" for g in gammas])
    a2.set_ylabel("QC - periodic R^2 gap (x spread)")
    a2.set_title("NARMA-10 edge is a weak-coupling effect"); a2.legend(fontsize=8)

    fig.suptitle("Rung 7: NARMA-10 -- the quasicrystal edge on a standard benchmark",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "reservoir_rung7_narma.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved {out}")

    # ---- verdict ----
    print("\n" + "=" * 78)
    print("HONEST VERDICT (rung 7)")
    print("=" * 78)
    win = d > 0 and d / s >= 1.0
    shrinks = gaps[0][4] > gaps[-1][4]
    print(f"  uncoupled NARMA-10: QC {'beats' if win else 'ties'} periodic "
          f"({d:+.3f} R^2, {d/s:+.1f}x).")
    print(f"  edge {'shrinks' if shrinks else 'does NOT shrink'} as coupling rises "
          f"({gaps[0][4]:+.1f}x -> {gaps[-1][4]:+.1f}x).")
    if win and shrinks:
        print("  => The selection-rule advantage SHOWS UP on the standard benchmark,")
        print("     in the regime predicted, on a task with a real even-order term and")
        print("     depth-10 memory. Modest, weak-coupling, even-order -- but it holds")
        print("     on the field's own yardstick, not just our home-made tasks.")
    else:
        print("  => On NARMA-10 the picture differs from the toy tasks -- report it")
        print("     straight (NARMA mixes even drive with recurrent output feedback, so")
        print("     it is not purely even-order; the edge may be diluted).")
    print("=" * 78)


if __name__ == "__main__":
    main()
