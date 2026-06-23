#!/usr/bin/env python3
"""
PEER REVIEW / CROSS-VALIDATION of the shallow even-order quasicrystal edge.

The selection-rule THEOREM is exact (proven + 1e-9 numeric). This script
adversarially stress-tests the COMPUTING claim (QC beats periodic on shallow
even-order tasks, weak coupling), trying to BREAK it four ways a referee would:

  CHECK 1 -- proper statistics. Vary the INPUT SEED (not just drive location)
    and use blocked k-fold cross-validation, then a PAIRED test (t-test +
    Wilcoxon + bootstrap CI) over (seed x drive) pairs. The edge must be
    significant under a real test, not a "gap/spread" heuristic.

  CHECK 2 -- feature-scaling confound (the deepest objection). QC has larger
    |c2| (0.34 vs 0.08), so its nonlinear features have larger amplitude; a
    fixed ridge lambda might just favor bigger features. If STANDARDIZING the
    feature columns (z-score) KILLS the edge, the mechanism story is wrong
    (it was amplitude). If the edge SURVIVES standardization, it is genuinely
    about which modes can REPRESENT products (the selection rule), since a
    c2=0 (periodic, symmetric) mode stays product-blind at any scale.

  CHECK 3 -- lambda sensitivity. The edge must not depend on a lucky ridge
    regularization; sweep lambda over decades.

  CHECK 4 -- is it SYMMETRY or QUASICRYSTALLINITY? (the framing objection.)
    Add a LOW-SYMMETRY PERIODIC plate (a sheared/anisotropic grid that breaks
    D4). If it ALSO gets a chunk of the edge over the D4 grid, the correct
    claim is "break the point symmetry", not "use a quasicrystal" -- a
    necessary correction to the writeup. We predict the even-task edge tracks
    (1 - dead-fraction), regardless of periodic vs aperiodic.

Sanity controls included: an ODD task (must tie for everyone) and the c2
dead-fraction of each plate (the structural driver).
"""
import os
import sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reservoir_rung4_modeshapes import (  # noqa: E402
    Lx, Ly, debruijn_quasicrystal_points, generate_periodic_holes,
    nearest_elem, ridge_fit, r2, QC_NFOLD, QC_SEED, INPUT_AMP,
)
from reservoir_rung6_stresstest import make_plate, omega_of, run_uncoupled  # noqa: E402

L = 1700
WASHOUT = 200
SEEDS = [0, 1, 2, 3, 4]
_drng = np.random.default_rng(5)
DRIVE_FRACS = [tuple(p) for p in _drng.uniform(0.22, 0.78, size=(8, 2))]


def generate_sheared_grid(n_side, domain_x, domain_y, shear=0.30, aniso=0.78,
                          off=0.04):
    """A PERIODIC lattice with BROKEN point symmetry: shear + anisotropy +
    off-center shift remove the mirrors, the 4-fold and (mostly) the inversion
    of the D4 grid, while staying a regular repeating arrangement."""
    span = 0.86 * min(domain_x, domain_y)
    g = np.linspace(-span / 2, span / 2, n_side)
    gx, gy = np.meshgrid(g, g)
    X = gx + shear * gy
    Y = aniso * gy
    # mild non-centrosymmetric warp to break C2 as well
    X = X + off * span * (gy / span) ** 2
    pts = np.column_stack([X.ravel() + domain_x / 2 + off * span,
                           Y.ravel() + domain_y / 2])
    return pts


def dead_fraction(c2, rel=0.10):
    a = np.abs(c2)
    return float(np.mean(a < rel * a.max()))


def kfold_r2(states, target, lam, standardize, k=3, purge=40):
    """Blocked k-fold CV R^2 (contiguous test blocks, purge gap to kill
    reservoir-memory leakage across the train/test boundary)."""
    post = np.arange(WASHOUT, len(target))
    folds = np.array_split(post, k)
    out = []
    for i in range(k):
        te = folds[i]
        lo, hi = te[0], te[-1]
        tr = np.array([j for f in folds for j in f if j < lo - purge or j > hi + purge])
        Xtr, Ytr, Xte, Yte = states[tr], target[tr], states[te], target[te]
        if standardize:
            mu = Xtr[:, :-1].mean(0); sd = Xtr[:, :-1].std(0) + 1e-9
            Xtr = np.column_stack([(Xtr[:, :-1] - mu) / sd, np.ones(len(Xtr))])
            Xte = np.column_stack([(Xte[:, :-1] - mu) / sd, np.ones(len(Xte))])
        W = ridge_fit(Xtr, Ytr[:, None], lam=lam)
        out.append(r2(Yte, (Xte @ W)[:, 0]))
    return float(np.mean(out))


def run_all(plate, seed, drive_frac):
    """Reservoir states for one (plate, seed, drive) on a fresh input."""
    rng = np.random.default_rng(1000 + seed)
    u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)
    om = omega_of(plate)
    e = nearest_elem(plate["ctr"], drive_frac)
    w_in = plate["Phi"][:, e].copy()
    states = run_uncoupled(om, plate["c2"], plate["c3"], w_in, u)
    even = np.zeros(L); even[2:] = u[1:L - 1] * u[0:L - 2]    # u[n-1]*u[n-2]
    odd = np.zeros(L); odd[1:] = u[:L - 1]                    # u[n-1] (linear)
    return states, even, odd


def collect(plate, lam=1e-4, standardize=False, task="even"):
    """R^2 for every (seed, drive) pair -> array of length seeds*drives."""
    vals = []
    for s in SEEDS:
        for fr in DRIVE_FRACS:
            st, ev, od = run_all(plate, s, fr)
            tgt = ev if task == "even" else od
            vals.append(kfold_r2(st, tgt, lam, standardize))
    return np.array(vals)


def paired(qc_vals, pe_vals, label):
    gap = qc_vals - pe_vals
    t_p = stats.ttest_rel(qc_vals, pe_vals).pvalue
    try:
        w_p = stats.wilcoxon(gap).pvalue
    except ValueError:
        w_p = float("nan")
    boot = [np.mean(rng_.choice(gap, len(gap))) for rng_ in
            [np.random.default_rng(b) for b in range(2000)]]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    frac_pos = np.mean(gap > 0)
    print(f"  {label:<34} mean gap {gap.mean():+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  "
          f"t p={t_p:.1e}  wilcoxon p={w_p:.1e}  QC>per in {frac_pos*100:.0f}% of pairs")
    return gap.mean(), lo, hi, t_p


def main():
    print("=" * 90)
    print("PEER REVIEW / CROSS-VALIDATION -- shallow even-order quasicrystal edge")
    print("=" * 90)

    print("\nBuilding plates (85% coverage)...")
    qc = make_plate(debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED), 85.0)
    pe = make_plate(generate_periodic_holes(9, Lx, Ly), 85.0)
    lowsym = make_plate(generate_sheared_grid(9, Lx, Ly), 85.0)
    print("done.")
    print(f"  c2 'dead' fraction:  QC={dead_fraction(qc['c2']):.2f}  "
          f"periodic-D4={dead_fraction(pe['c2']):.2f}  "
          f"low-sym periodic={dead_fraction(lowsym['c2']):.2f}")
    print(f"  ({len(SEEDS)} input seeds x {len(DRIVE_FRACS)} drives = "
          f"{len(SEEDS)*len(DRIVE_FRACS)} paired samples per comparison)")

    # baseline reservoir states reused implicitly via collect()
    print("\n" + "-" * 90)
    print("CHECK 1 -- proper statistics (seeds x drives, k-fold CV, paired tests)")
    print("-" * 90)
    qc_e = collect(qc, task="even")
    pe_e = collect(pe, task="even")
    print(f"  even task  QC R^2 = {qc_e.mean():.3f}+/-{qc_e.std():.3f}   "
          f"periodic R^2 = {pe_e.mean():.3f}+/-{pe_e.std():.3f}")
    g1 = paired(qc_e, pe_e, "even task (QC vs periodic-D4)")
    # odd control
    qc_o = collect(qc, task="odd"); pe_o = collect(pe, task="odd")
    paired(qc_o, pe_o, "ODD control (must be ~tie)")

    print("\n" + "-" * 90)
    print("CHECK 2 -- feature-scaling confound: does standardizing kill the edge?")
    print("-" * 90)
    qc_es = collect(qc, standardize=True, task="even")
    pe_es = collect(pe, standardize=True, task="even")
    print(f"  even task (standardized features)  QC {qc_es.mean():.3f}  per {pe_es.mean():.3f}")
    paired(qc_es, pe_es, "even task, STANDARDIZED")

    print("\n" + "-" * 90)
    print("CHECK 3 -- lambda sensitivity (edge must survive across decades)")
    print("-" * 90)
    for lam in [1e-6, 1e-4, 1e-2, 1e-1]:
        a = collect(qc, lam=lam, task="even"); b = collect(pe, lam=lam, task="even")
        gap = a - b
        print(f"  lambda={lam:.0e}:  QC {a.mean():.3f}  per {b.mean():.3f}  "
              f"gap {gap.mean():+.3f}  (QC>per in {np.mean(gap>0)*100:.0f}% of pairs)")

    print("\n" + "-" * 90)
    print("CHECK 4 -- symmetry vs quasicrystallinity: does a LOW-SYM PERIODIC plate")
    print("           also get the edge over the D4 grid?")
    print("-" * 90)
    ls_e = collect(lowsym, task="even")
    print(f"  even task R^2:  D4-periodic {pe_e.mean():.3f}   "
          f"low-sym periodic {ls_e.mean():.3f}   quasicrystal {qc_e.mean():.3f}")
    paired(ls_e, pe_e, "low-sym periodic vs D4 periodic")
    paired(qc_e, ls_e, "quasicrystal vs low-sym periodic")

    # ---- review verdict ----
    print("\n" + "=" * 90)
    print("REVIEW VERDICT")
    print("=" * 90)
    edge_sig = g1[3] < 0.05 and g1[1] > 0
    surv_std = (qc_es.mean() - pe_es.mean()) > 0
    sym_not_qc = ls_e.mean() > pe_e.mean() + 0.25 * (qc_e.mean() - pe_e.mean())
    print(f"  [1] edge significant under paired test (p<0.05, CI excludes 0): {edge_sig}")
    print(f"  [2] edge survives feature standardization (not a scaling artifact): {surv_std}")
    print(f"  [3] (see lambda table above -- edge should persist across decades)")
    print(f"  [4] low-sym PERIODIC also beats D4 periodic (=> it's SYMMETRY, "
          f"not quasicrystallinity): {sym_not_qc}")
    print()
    if edge_sig and surv_std:
        print("  The computing edge is STATISTICALLY REAL and NOT a feature-scaling")
        print("  artifact. ", end="")
    else:
        print("  WARNING: the edge failed a core robustness check -- see above. ", end="")
    if sym_not_qc:
        print("CORRECTION REQUIRED: the effect is driven by BROKEN POINT")
        print("  SYMMETRY, not aperiodicity per se -- a low-symmetry periodic plate")
        print("  captures much of it. The writeup must claim 'break the symmetry',")
        print("  with the quasicrystal as one (convenient, maximal) way to do so.")
    else:
        print("the quasicrystal specifically retains the")
        print("  largest edge; low-symmetry periodic does not fully close the gap.")
    print("=" * 90)


if __name__ == "__main__":
    main()
