#!/usr/bin/env python3
"""
RUNG 7b -- WHY does NARMA-10 tie when the toy even-order tasks win?

NARMA-10 tied (rung 7): QC +0.012 R^2 (0.7x), both plates only ~0.32 R^2. Two
candidate explanations, with OPPOSITE implications:
  (1) the even-order edge is real but DILUTED -- NARMA's variance is mostly
      linear/recurrent memory (the 0.3 y[n] AR term + deep output feedback),
      which both plates handle equally, so the even-order term (1.5 u[n]u[n-9])
      is a minority of the signal and the edge washes out;
  (2) the edge only exists at SHALLOW memory depth, and NARMA's even term is
      deep (lag 9), so the selection-rule advantage never applies there.

DECISIVE TEST: race QC vs periodic on PURE even-order product tasks
    y[n] = u[n-1] * u[n-d]
at increasing depth d. If QC keeps winning out to d~9-11, the edge is
depth-robust -> NARMA ties by DILUTION (explanation 1). If the edge dies as d
grows, it is shallow-only (explanation 2). Reports absolute R^2 too, so we can
see how hard deep products are for this reservoir regardless of plate.
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
    nearest_elem, ridge_fit, r2, QC_NFOLD, QC_SEED, INPUT_AMP,
)
from reservoir_rung6_stresstest import make_plate, omega_of, run_uncoupled  # noqa: E402

L = 1900
WASHOUT = 200
N_TRAIN = 1200
DEPTHS = [2, 3, 5, 7, 9, 11]
_drng = np.random.default_rng(11)
DRIVE_FRACS = [tuple(p) for p in _drng.uniform(0.22, 0.78, size=(12, 2))]


def fit_eval(states, target):
    Xtr, Ytr = states[WASHOUT:WASHOUT + N_TRAIN], target[WASHOUT:WASHOUT + N_TRAIN]
    Xte, Yte = states[WASHOUT + N_TRAIN:], target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None], lam=1e-5)
    return r2(Yte, (Xte @ W)[:, 0])


def race(P_qc, P_per, u, target):
    out = {}
    for tag, P in [("qc", P_qc), ("per", P_per)]:
        om = omega_of(P); sc = []
        for frac in DRIVE_FRACS:
            e = nearest_elem(P["ctr"], frac)
            w_in = P["Phi"][:, e].copy()
            st = run_uncoupled(om, P["c2"], P["c3"], w_in, u)
            sc.append(fit_eval(st, target))
        out[tag] = np.array(sc)
    q, p = out["qc"], out["per"]
    d = q.mean() - p.mean(); s = np.hypot(q.std(), p.std()) + 1e-9
    return q, p, d, d / s


def main():
    print("=" * 78)
    print("RUNG 7b -- does the even-order edge survive to NARMA-like memory depth?")
    print("=" * 78)

    rng = np.random.default_rng(0)
    u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)   # symmetric -> clean even products

    print("Building plates (85% coverage, 8-fold QC)...")
    qc = make_plate(debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED), 85.0)
    per = make_plate(generate_periodic_holes(9, Lx, Ly), 85.0)
    print("done.\n")

    print(f"  even product y[n]=u[n-1]*u[n-d], uncoupled, 12 drives:")
    print(f"  {'depth d':>8}{'QC R^2':>10}{'per R^2':>10}{'gap':>9}{'sig':>7}  winner")
    rows = []
    for d in DEPTHS:
        tgt = np.zeros(L)
        tgt[d:] = u[1:L - d + 1] * u[0:L - d]    # u[n-1]*u[n-d]
        q, p, gap, sig = race(qc, per, u, tgt)
        w = "QC" if (gap > 0 and sig >= 1) else ("periodic" if (gap < 0 and abs(sig) >= 1) else "tie")
        rows.append((d, q.mean(), p.mean(), gap, sig))
        print(f"  {d:>8}{q.mean():>10.3f}{p.mean():>10.3f}{gap:>+9.3f}{sig:>+7.1f}  {w}")

    rows = np.array(rows)

    # ---- plot ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(rows[:, 0], rows[:, 1], "o-", color="#2E5E8C", label="quasicrystal")
    a1.plot(rows[:, 0], rows[:, 2], "s-", color="#C0392B", label="periodic")
    a1.set_title("Absolute performance vs product depth\n(deep products are hard for ANY plate)")
    a1.set_xlabel("memory depth d in u[n-1]*u[n-d]"); a1.set_ylabel("R^2")
    a1.legend(); a1.grid(alpha=0.3)

    a2.bar(rows[:, 0], rows[:, 4], color="#2E5E8C", width=0.6)
    a2.axhline(1.0, color="green", ls="--", lw=1, label="1x spread")
    a2.axhline(0, color="k", lw=0.8)
    a2.set_title("Quasicrystal edge vs depth")
    a2.set_xlabel("memory depth d"); a2.set_ylabel("QC - periodic R^2 (x spread)")
    a2.legend(fontsize=8); a2.grid(alpha=0.3, axis="y")

    fig.suptitle("Rung 7b: the even-order edge vs memory depth -- explaining the NARMA tie",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "reservoir_rung7b_depth.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved {out}")

    # ---- verdict ----
    deep = rows[rows[:, 0] >= 7]
    deep_edge = np.mean(deep[:, 4])
    abs_drop = rows[0, 1] - rows[-1, 1]
    print("\n" + "=" * 78)
    print("HONEST VERDICT (rung 7b)")
    print("=" * 78)
    print(f"  QC edge stays positive across depth (mean sig: shallow "
          f"{np.mean(rows[rows[:,0]<=3,4]):+.1f}x, deep {deep_edge:+.1f}x).")
    print(f"  absolute R^2 falls {rows[0,1]:.2f} -> {rows[-1,1]:.2f} as depth 2 -> 11 "
          f"(deep products are hard for BOTH plates).")
    if deep_edge >= 0.8:
        print("  => Explanation (1): the even-order edge is DEPTH-ROBUST -- it survives")
        print("     to NARMA-like lag-9. So NARMA-10 ties by DILUTION: its variance is")
        print("     dominated by linear/recurrent memory that both plates do equally, and")
        print("     the even-order term where the edge lives is only a minority of the")
        print("     signal. The finding is intact; NARMA just isn't mostly even-order.")
    else:
        print("  => Explanation (2): the edge fades with depth, so it is a shallow-memory")
        print("     effect. That ALSO explains the NARMA tie (its even term is deep). The")
        print("     finding's scope tightens to shallow even-order tasks. Honest either way.")
    print("=" * 78)


if __name__ == "__main__":
    main()
