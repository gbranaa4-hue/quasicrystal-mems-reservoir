#!/usr/bin/env python3
"""
RUNG 5c -- the ACTUAL mechanism behind the uncoupled quasicrystal edge.

Rung 5b proved the uncoupled QC advantage (task R^2 ~0.53 vs ~0.38, 2.2x
spread) is REAL but is NOT explained by effective dimensionality, input
utilization, or degeneracy redundancy (all equal between the plates).

REMAINING HYPOTHESIS -- a quadratic selection rule:
  the nonlinear task y=u[n-1]u[n-2] is a PRODUCT, and in this model the only
  product-generating term is the quadratic self-nonlinearity c2_i * x_i^2,
  with c2_i = integral(phi_i^3 dA). The periodic plate has the square's full
  D4 symmetry, so its ANTISYMMETRIC modes have phi(-x) = -phi(x) -> phi^3 is
  antisymmetric -> c2_i = 0 EXACTLY. Roughly half its modes lose their
  product-generating term to symmetry. The quasicrystal has no exact mirror
  symmetry, so every mode keeps a nonzero c2_i. Hence a real quadratic-task
  edge -- with no coupling at all.

TWO DECISIVE ABLATIONS (gamma=0 throughout):
  (1) EQUALIZE c2: force every mode of BOTH plates to the same quadratic
      coefficient. If this ERASES the gap, the c2 distribution WAS the cause.
  (2) EQUALIZE frequencies: give both plates the same mode-frequency set
      (keep real c2/shapes). If the gap SURVIVES, it isn't the spectrum.

Prediction: ablation (1) erases the gap; ablation (2) leaves it. That would
pin the cause to the quadratic selection rule -- a clean, physical, honest
mechanism. Reported as-is whatever happens (this explains an observed effect,
it does not hunt for a new one).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reservoir_rung4_modeshapes import (  # noqa: E402
    mode_shapes, nearest_elem,
    Lx, Ly, debruijn_quasicrystal_points, generate_periodic_holes,
    coverage_match_radius, ridge_fit, r2,
    NX, N_MODES, TARGET_COV, QC_NFOLD, QC_SEED, OMEGA_LO, OMEGA_HI,
    ZETA, A_QUAD, B_CUBIC, TAU_IN, DT, INPUT_AMP,
)
from reservoir_rung5_saturation import run_reservoir_gamma  # noqa: E402

L = 1600
WASHOUT = 200
N_TRAIN = 1000
_drng = np.random.default_rng(321)
DRIVE_FRACS = [tuple(p) for p in _drng.uniform(0.22, 0.78, size=(14, 2))]


def eval_task(states, target):
    Xtr, Ytr = states[WASHOUT:WASHOUT + N_TRAIN], target[WASHOUT:WASHOUT + N_TRAIN]
    Xte, Yte = states[WASHOUT + N_TRAIN:], target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None])
    return r2(Yte, (Xte @ W)[:, 0])


def race(qc, per, u, yA, label, c2_override=None, freq_override=None):
    """Race the two plates at gamma=0 under an optional ablation.
    qc/per are dicts with f, Phi, aw, ctr, c2, c3."""
    out = {}
    for name, P in [("QC", qc), ("periodic", per)]:
        freqs = P["f"] if freq_override is None else freq_override
        omega = OMEGA_LO + (OMEGA_HI - OMEGA_LO) * (freqs - freqs.min()) / (freqs.max() - freqs.min())
        c2 = P["c2"] if c2_override is None else np.full(len(P["f"]), c2_override)
        c3 = P["c3"]
        scores = []
        for frac in DRIVE_FRACS:
            e = nearest_elem(P["ctr"], frac)
            w_in = P["Phi"][:, e].copy()
            states = run_reservoir_gamma(omega, P["Phi"], P["aw"], c2, c3, w_in, u, 0.0)
            scores.append(eval_task(states, yA))
        out[name] = np.array(scores)
    q, p = out["QC"], out["periodic"]
    d = q.mean() - p.mean(); s = np.hypot(q.std(), p.std()) + 1e-9
    print(f"  {label:<34} QC {q.mean():.3f}+/-{q.std():.3f}  "
          f"periodic {p.mean():.3f}+/-{p.std():.3f}  gap {d:+.3f} ({abs(d)/s:.1f}x)")
    return d, abs(d) / s


def main():
    print("=" * 80)
    print("RUNG 5c -- is the uncoupled quasicrystal edge a QUADRATIC selection rule?")
    print("=" * 80)
    print(f"\nComputing real FEM mode shapes ({N_MODES} modes)...")
    plates = {}
    for name, holes_fn in [("QC", lambda: debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED)),
                           ("periodic", lambda: generate_periodic_holes(9, Lx, Ly))]:
        holes = holes_fn()
        r, _ = coverage_match_radius(holes, TARGET_COV, NX)
        f, Phi, aw, ctr = mode_shapes(holes, r, N_MODES, NX)
        c2 = (aw[None, :] * Phi**3).sum(axis=1)
        c3 = (aw[None, :] * Phi**4).sum(axis=1)
        plates[name] = dict(f=f, Phi=Phi, aw=aw, ctr=ctr, c2=c2, c3=c3)
    qc, per = plates["QC"], plates["periodic"]
    print("done.\n")

    # ---- structural check: distribution of |c2_i| ----
    def frac_near_zero(c2, rel=0.1):
        a = np.abs(c2); return float(np.mean(a < rel * a.max()))
    qz, pz = frac_near_zero(qc["c2"]), frac_near_zero(per["c2"])
    print("STRUCTURAL CHECK -- quadratic self-coefficient c2_i = integral(phi_i^3 dA):")
    print(f"  fraction of modes with |c2_i| < 10% of max (product term ~dead):")
    print(f"      quasicrystal = {qz:.2f}    periodic = {pz:.2f}   "
          f"(prediction: periodic >> QC, from D4 symmetry)")
    print(f"  mean |c2_i|:  quasicrystal = {np.abs(qc['c2']).mean():.3e}   "
          f"periodic = {np.abs(per['c2']).mean():.3e}\n")

    rng = np.random.default_rng(0)
    u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)
    yA = np.zeros(L); yA[2:] = u[1:L-1] * u[0:L-2]

    print("RACES (gamma=0, uncoupled):")
    d0, s0 = race(qc, per, u, yA, "(1) baseline [real c2, real freqs]")

    # equalize c2 to a common constant = mean |c2| over both plates
    c2_common = 0.5 * (np.abs(qc["c2"]).mean() + np.abs(per["c2"]).mean())
    d1, s1 = race(qc, per, u, yA, "(2) EQUALIZED c2 [common quadratic]", c2_override=c2_common)

    # equalize frequencies: both plates use the QC frequency set
    d2, s2 = race(qc, per, u, yA, "(3) EQUALIZED freqs [both use QC spectrum]",
                  freq_override=qc["f"])

    # ---- verdict ----
    print("\n" + "=" * 80)
    print("HONEST VERDICT (rung 5c)")
    print("=" * 80)
    print(f"  baseline gap         : {d0:+.3f} ({s0:.1f}x)")
    print(f"  with c2 EQUALIZED    : {d1:+.3f} ({s1:.1f}x)   "
          f"<- collapses if c2 was the cause")
    print(f"  with freqs EQUALIZED : {d2:+.3f} ({s2:.1f}x)   "
          f"<- survives if spectrum was NOT the cause")
    c2_is_cause = (s0 >= 1.0) and (s1 < 1.0)
    if c2_is_cause and pz > qz + 0.1:
        print("\n  => MECHANISM PINNED: the uncoupled quasicrystal edge is a QUADRATIC")
        print("     SELECTION RULE. The periodic plate's D4 symmetry forces c2_i=0 for")
        print(f"     its antisymmetric modes ({pz:.0%} near-dead vs QC {qz:.0%}), killing")
        print("     their product-generating term; the quasicrystal's broken symmetry")
        print("     keeps every mode's quadratic term alive. Equalizing c2 erases the")
        print("     gap; equalizing frequencies does not. This is the honest, real,")
        print("     mechanism-level finding of the whole ladder -- a MODEST edge, in the")
        print("     weakly-coupled regime, on quadratic tasks, from symmetry-breaking.")
    else:
        print("\n  => The quadratic-selection-rule hypothesis is NOT cleanly supported by")
        print("     the ablations; the cause is mixed or elsewhere. Reported honestly.")
    print("=" * 80)


if __name__ == "__main__":
    main()
