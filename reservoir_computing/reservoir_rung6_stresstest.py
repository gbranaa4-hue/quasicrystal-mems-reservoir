#!/usr/bin/env python3
"""
RUNG 6 -- STRESS-TEST the one real finding: does the quasicrystal's
quadratic SYMMETRY-SELECTION-RULE edge generalize, or is it an artifact?

The finding (rung 5c): in the weakly-coupled regime, the quasicrystal beats
the periodic plate on the product task because periodic D4 symmetry zeroes
the product-generating even-order self-nonlinearity c2_i = integral(phi_i^3)
in ~88% of modes, while the quasicrystal keeps it alive. We now try hard to
BREAK that conclusion along three axes. Each has a falsifiable prediction.

  A. TASK GENERALITY (the sharpest test). If the edge is truly a quadratic
     selection rule, it must appear on EVERY even-order task (products,
     squares -- which need c2) and be ABSENT on odd-order tasks (linear,
     cubic, triple-product -- which need c1/c3, and note c3=integral(phi^4)
     has an EVEN integrand so symmetry NEVER kills it). Predict: QC wins the
     even tasks, ties the odd ones. If QC also wins odd tasks, the mechanism
     story is incomplete.

  B. GEOMETRY ROBUSTNESS. Vary hole coverage and the quasicrystal symmetry
     (de Bruijn n-fold). The mechanism is "break the periodic D4 symmetry",
     which ANY quasicrystal does -- so predict the even-task edge persists
     across coverages and n-fold values (magnitude may vary).

  C. PHYSICAL EVEN NONLINEARITY. The a*w^2 term was generic. A real MEMS
     plate gets its even term from ELECTROSTATIC actuation: the pressure
     ~1/(g-w)^2 expands to a w^2 term with coefficient set by gap/bias, and
     a softening sign. Its modal self-projection is STILL proportional to
     integral(phi_i^3) -> same selection rule. Swap the generic even term
     for the electrostatic one (different sign and magnitude, with mechanical
     hardening to bound amplitude) and predict the edge survives -- because
     it is a symmetry property of integral(phi^3), not of the coefficient.

All tests are in the UNCOUPLED regime (gamma=0) where the effect lives, using
a fast independent-oscillator integrator. Reported as-is, whatever happens.
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
    mode_shapes, nearest_elem,
    Lx, Ly, debruijn_quasicrystal_points, generate_periodic_holes,
    coverage_match_radius, ridge_fit, r2,
    NX, N_MODES, TARGET_COV, QC_NFOLD, QC_SEED, OMEGA_LO, OMEGA_HI,
    ZETA, A_QUAD, B_CUBIC, TAU_IN, DT, INPUT_AMP,
)

L = 1600
WASHOUT = 200
N_TRAIN = 1000
_drng = np.random.default_rng(99)
DRIVE_FRACS = [tuple(p) for p in _drng.uniform(0.22, 0.78, size=(12, 2))]


def run_uncoupled(omega, c2, c3, w_in, u_series, a_even=A_QUAD, b_cubic=B_CUBIC,
                  k_soft=0.0):
    """Fast gamma=0 reservoir: each mode is an independent oscillator
        x'' + 2 zeta omega x' + omega^2 x = -a_even c2 x^2 - b_cubic c3 x^3
                                            + k_soft x + w_in u
    (k_soft>0 = electrostatic spring-softening; off by default)."""
    N = len(omega)
    x = np.zeros(N); v = np.zeros(N)
    n_sub = int(round(TAU_IN / DT))
    feats = np.empty((len(u_series), 2 * N + 1))
    o2 = omega**2 - k_soft
    damp = 2 * ZETA * omega
    for n, u in enumerate(u_series):
        uu = w_in * u
        for _ in range(n_sub):
            accel = -o2 * x - damp * v - a_even * c2 * x * x - b_cubic * c3 * x * x * x + uu
            v = v + accel * DT
            x = x + v * DT
        feats[n, :N] = x
        feats[n, N:2*N] = v
        feats[n, -1] = 1.0
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"blew up at sample {n}")
    return feats


def eval_task(states, target):
    Xtr, Ytr = states[WASHOUT:WASHOUT + N_TRAIN], target[WASHOUT:WASHOUT + N_TRAIN]
    Xte, Yte = states[WASHOUT + N_TRAIN:], target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None])
    return r2(Yte, (Xte @ W)[:, 0])


def make_plate(holes, cov, nx=NX, nmodes=N_MODES):
    r, _ = coverage_match_radius(holes, cov, nx)
    f, Phi, aw, ctr = mode_shapes(holes, r, nmodes, nx)
    c2 = (aw[None, :] * Phi**3).sum(axis=1)
    c3 = (aw[None, :] * Phi**4).sum(axis=1)
    return dict(f=f, Phi=Phi, aw=aw, ctr=ctr, c2=c2, c3=c3)


def omega_of(P):
    f = P["f"]
    return OMEGA_LO + (OMEGA_HI - OMEGA_LO) * (f - f.min()) / (f.max() - f.min())


def race_task(P_qc, P_per, u, target, **nl):
    """Average R^2 over drive locations for each plate; return (gap, sig)."""
    out = {}
    for tag, P in [("qc", P_qc), ("per", P_per)]:
        om = omega_of(P)
        sc = []
        for frac in DRIVE_FRACS:
            e = nearest_elem(P["ctr"], frac)
            w_in = P["Phi"][:, e].copy()
            st = run_uncoupled(om, P["c2"], P["c3"], w_in, u, **nl)
            sc.append(eval_task(st, target))
        out[tag] = np.array(sc)
    q, p = out["qc"], out["per"]
    d = q.mean() - p.mean(); s = np.hypot(q.std(), p.std()) + 1e-9
    return q, p, d, d / s


def build_tasks(u):
    n = len(u)
    def sh(k): z = np.zeros(n); z[k:] = u[:n - k]; return z
    u1, u2, u3 = sh(1), sh(2), sh(3)
    even = {
        "u1*u2": u1 * u2, "u1^2": u1 * u1, "u1*u3": u1 * u3,
        "u2*u3": u2 * u3, "u1u2+u2u3": u1 * u2 + u2 * u3,
    }
    odd = {
        "u1 (lin)": u1, "u2 (lin)": u2, "u1^3": u1**3,
        "u1*u2*u3": u1 * u2 * u3, "u1-0.5u2": u1 - 0.5 * u2,
    }
    return even, odd


def main():
    print("=" * 82)
    print("RUNG 6 -- stress-test the quadratic symmetry-selection-rule finding")
    print("=" * 82)

    rng = np.random.default_rng(0)
    u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)
    even, odd = build_tasks(u)

    # base plates at 85% coverage
    print("\nBuilding base plates (85% coverage, 8-fold QC)...")
    qc_holes = debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED)
    per_holes = generate_periodic_holes(9, Lx, Ly)
    qc = make_plate(qc_holes, 85.0)
    per = make_plate(per_holes, 85.0)
    print("done.")

    # ---------- A. task generality ----------
    print("\n" + "-" * 82)
    print("A. TASK GENERALITY -- even-order tasks should light up, odd-order should tie")
    print("-" * 82)
    print(f"  {'task':<14}{'order':>6}{'QC R^2':>10}{'per R^2':>10}{'gap':>9}{'sig':>7}  winner")
    rowsA = []
    for name, tgt in list(even.items()):
        q, p, d, sg = race_task(qc, per, u, tgt)
        w = "QC" if (sg >= 1.0 and d > 0) else ("periodic" if sg <= -1.0 else "tie")
        rowsA.append((name, "even", q.mean(), p.mean(), d, sg, w))
        print(f"  {name:<14}{'even':>6}{q.mean():>10.3f}{p.mean():>10.3f}{d:>+9.3f}{sg:>+7.1f}  {w}")
    for name, tgt in list(odd.items()):
        q, p, d, sg = race_task(qc, per, u, tgt)
        w = "QC" if (sg >= 1.0 and d > 0) else ("periodic" if sg <= -1.0 else "tie")
        rowsA.append((name, "odd", q.mean(), p.mean(), d, sg, w))
        print(f"  {name:<14}{'odd':>6}{q.mean():>10.3f}{p.mean():>10.3f}{d:>+9.3f}{sg:>+7.1f}  {w}")

    even_gaps = [r[4] for r in rowsA if r[1] == "even"]
    odd_gaps = [r[4] for r in rowsA if r[1] == "odd"]
    even_win = np.mean([1.0 if (r[5] >= 1.0 and r[4] > 0) else 0.0 for r in rowsA if r[1] == "even"])
    odd_tie = np.mean([1.0 if abs(r[5]) < 1.0 else 0.0 for r in rowsA if r[1] == "odd"])
    print(f"\n  mean gap  even = {np.mean(even_gaps):+.3f}   odd = {np.mean(odd_gaps):+.3f}")
    print(f"  QC wins {even_win*100:.0f}% of even tasks; {odd_tie*100:.0f}% of odd tasks are ties")

    # ---------- B. geometry robustness ----------
    print("\n" + "-" * 82)
    print("B. GEOMETRY ROBUSTNESS -- does the even-task edge persist across geometry?")
    print("-" * 82)
    tgt = even["u1*u2"]
    rowsB = []
    print("  coverage sweep (8-fold QC vs periodic), task u1*u2:")
    for cov in [78.0, 85.0, 92.0]:
        P_qc = qc if cov == 85.0 else make_plate(qc_holes, cov)
        P_pe = per if cov == 85.0 else make_plate(per_holes, cov)
        q, p, d, sg = race_task(P_qc, P_pe, u, tgt)
        rowsB.append((f"cov{int(cov)}", d, sg))
        print(f"    coverage {cov:>4.0f}%   QC {q.mean():.3f}  per {p.mean():.3f}  "
              f"gap {d:+.3f} ({sg:+.1f}x)")
    print("  quasicrystal symmetry sweep (n-fold) at 85% vs same periodic:")
    for nf in [5, 7, 8, 12]:
        P_qc = qc if nf == QC_NFOLD else make_plate(
            debruijn_quasicrystal_points(nf, Lx, Ly, offset_seed=QC_SEED), 85.0)
        q, p, d, sg = race_task(P_qc, per, u, tgt)
        rowsB.append((f"{nf}-fold", d, sg))
        print(f"    {nf:>2}-fold QC    QC {q.mean():.3f}  per {p.mean():.3f}  "
              f"gap {d:+.3f} ({sg:+.1f}x)")

    # ---------- C. physical (electrostatic) even nonlinearity ----------
    print("\n" + "-" * 82)
    print("C. PHYSICAL EVEN NONLINEARITY -- electrostatic-style even term + mech. hardening")
    print("-" * 82)
    # electrostatic: even term SOFTENING (opposite sign), different magnitude,
    # plus a spring-softening linear term; mechanical cubic kept for stability.
    es = dict(a_even=-0.6, b_cubic=1.0, k_soft=0.15)
    print(f"  model: a_even={es['a_even']} (softening), b_cubic={es['b_cubic']} "
          f"(mech. hardening), k_soft={es['k_soft']}")
    q, p, d, sg = race_task(qc, per, u, even["u1*u2"], **es)
    qo, po, do, sgo = race_task(qc, per, u, odd["u1^3"], **es)
    print(f"    even task u1*u2 :  QC {q.mean():.3f}  per {p.mean():.3f}  gap {d:+.3f} ({sg:+.1f}x)")
    print(f"    odd  task u1^3  :  QC {qo.mean():.3f}  per {po.mean():.3f}  gap {do:+.3f} ({sgo:+.1f}x)")

    # ---------- plot ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
    names = [r[0] for r in rowsA]; gaps = [r[4] for r in rowsA]
    cols = ["#2E5E8C" if r[1] == "even" else "#999999" for r in rowsA]
    a1.bar(range(len(rowsA)), gaps, color=cols)
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xticks(range(len(rowsA))); a1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    a1.set_ylabel("QC - periodic task R^2 gap")
    a1.set_title("A. Even-order tasks (blue) light up; odd-order (gray) tie")

    labels = [r[0] for r in rowsB]; sigs = [r[2] for r in rowsB]
    a2.bar(range(len(rowsB)), sigs, color="#2E5E8C")
    a2.axhline(1.0, color="green", ls="--", lw=1, label="1x spread (signal threshold)")
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(range(len(rowsB))); a2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    a2.set_ylabel("QC even-task edge (x spread)")
    a2.set_title("B. Edge persists across coverage & QC symmetry"); a2.legend(fontsize=8)

    fig.suptitle("Rung 6: stress-testing the quadratic symmetry-selection-rule edge", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "reservoir_rung6_stresstest.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved {out}")

    # ---------- verdict ----------
    print("\n" + "=" * 82)
    print("HONEST VERDICT (rung 6)")
    print("=" * 82)
    A_ok = even_win >= 0.8 and odd_tie >= 0.6 and np.mean(even_gaps) > 0
    B_ok = np.mean([1.0 if s >= 1.0 else 0.0 for _, _, s in rowsB]) >= 0.6
    C_ok = sg >= 1.0 and abs(sgo) < 1.0
    print(f"  A. task generality : {'PASS' if A_ok else 'FAIL'} "
          f"(even mean gap {np.mean(even_gaps):+.3f} win {even_win*100:.0f}%; "
          f"odd ties {odd_tie*100:.0f}%)")
    print(f"  B. geometry        : {'PASS' if B_ok else 'FAIL'} "
          f"({np.mean([1.0 if s>=1.0 else 0.0 for _,_,s in rowsB])*100:.0f}% of configs keep the edge)")
    print(f"  C. electrostatic   : {'PASS' if C_ok else 'FAIL'} "
          f"(even {sg:+.1f}x, odd {sgo:+.1f}x)")
    if A_ok and B_ok and C_ok:
        print("\n  => The finding HOLDS UP. The quasicrystal edge is specifically an")
        print("     EVEN-ORDER effect (even tasks win, odd tasks tie), robust across")
        print("     coverage and quasicrystal symmetry, and survives a physically-")
        print("     grounded electrostatic even nonlinearity. It is a genuine symmetry")
        print("     selection rule, not an artifact of one task / geometry / model.")
        print("     Still MODEST and confined to the weakly-coupled regime -- but real.")
    else:
        print("\n  => The finding is NARROWER than claimed (see which axis FAILED above).")
        print("     Report the limitation honestly; do not inflate the scope.")
    print("=" * 82)


if __name__ == "__main__":
    main()
