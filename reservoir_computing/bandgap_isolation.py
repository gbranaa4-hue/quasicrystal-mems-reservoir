#!/usr/bin/env python3
"""
BANDGAP ISOLATION TEST -- does keeping only the int(phi^3)-active modes help?

Claim under test: a bandgap could isolate only the even-order-ACTIVE modes
(c2=int(phi^3) != 0), so the SILENCED modes don't "add noise" -- raising SNR.
Honest counter-worry: the silenced modes are LINEAR; they may be carrying the
MEMORY the active modes need to form a product. Stripping them could HURT.

We test directly. In the uncoupled regime the modes are independent, so
"isolating" modes = selecting which modes' features reach the readout (exactly
what a bandgap would do physically). For each substrate we partition modes into
ACTIVE (|c2| above threshold) and SILENCED, and evaluate the even task
u[n-1]*u[n-2] reading out from:
    (a) ALL modes        (baseline)
    (b) ACTIVE only      (the bandgap-isolation proposal)
    (c) SILENCED only     (control -- should fail: no product-generating term)
at several readout-noise levels. If (b) >= (a), isolation helps (claim holds);
if (b) < (a), the silenced modes were providing needed memory (claim wrong).
"""
import os
import sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FEM_DIR = os.path.join(os.path.dirname(HERE), "plate_bending_review")
sys.path.insert(0, FEM_DIR)
from fem_plate_bending_homogenized import Lx as L0, debruijn_quasicrystal_points  # noqa: E402
from reservoir_rung4_modeshapes import ridge_fit, r2, nearest_elem  # noqa: E402
from reservoir_rung6_stresstest import run_uncoupled  # noqa: E402
from peer_review_cavity import cavity_modes, periodic_rect  # noqa: E402

N_MODES = 40
OMEGA_LO, OMEGA_HI = 0.5, 2.5
INPUT_AMP = 1.0
L = 1800
WASHOUT = 200
SEEDS = [0, 1, 2]
DRIVES = [(0.40, 0.55), (0.60, 0.42), (0.33, 0.66), (0.52, 0.62)]


def norm_omega(plate):
    f = plate["freqs"]
    return OMEGA_LO + (OMEGA_HI - OMEGA_LO) * (f - f.min()) / (f.max() - f.min())


def kfold_r2(F, target, lam=1e-6, k=3, purge=40):
    post = np.arange(WASHOUT, len(target)); folds = np.array_split(post, k)
    out = []
    for i in range(k):
        te = folds[i]; lo, hi = te[0], te[-1]
        tr = np.array([j for f in folds for j in f if j < lo - purge or j > hi + purge])
        W = ridge_fit(F[tr], target[tr, None], lam=lam)
        out.append(r2(target[te], (F[te] @ W)[:, 0]))
    return float(np.mean(out))


def eval_subset(feats, target, idx, noise, rng):
    N = (feats.shape[1] - 1) // 2
    cols = list(idx) + [N + i for i in idx] + [2 * N]   # x_idx, v_idx, bias
    F = feats[:, cols].copy()
    if noise > 0 and len(idx) > 0:
        F[:, :-1] = F[:, :-1] + rng.normal(0.0, noise * (F[:, :-1].std(0) + 1e-12), F[:, :-1].shape)
    return kfold_r2(F, target)


def run_substrate(plate, name):
    om = norm_omega(plate)
    active = np.where(np.abs(plate["c2"]) > 0.10 * np.abs(plate["c2"]).max())[0]
    silenced = np.where(np.abs(plate["c2"]) <= 0.10 * np.abs(plate["c2"]).max())[0]
    print(f"\n{name}: {len(active)} active modes, {len(silenced)} silenced "
          f"(silenced fraction {len(silenced)/N_MODES:.2f})")
    results = {nz: {"all": [], "active": [], "silenced": []} for nz in [0.0, 0.05, 0.10]}
    for s in SEEDS:
        rng = np.random.default_rng(100 + s); u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)
        et = np.zeros(L); et[2:] = u[1:L - 1] * u[0:L - 2]
        for di, fr in enumerate(DRIVES):
            e = nearest_elem(plate["ctr"], fr); w_in = plate["Wc"][:, e].copy()
            feats = run_uncoupled(om, plate["c2"], plate["c3"], w_in, u)
            for nz in results:
                nrng = np.random.default_rng(900 + s * 10 + di)
                results[nz]["all"].append(eval_subset(feats, et, range(N_MODES), nz, nrng))
                results[nz]["active"].append(eval_subset(feats, et, active, nz, nrng))
                results[nz]["silenced"].append(eval_subset(feats, et, silenced, nz, nrng))
    print(f"  {'noise':>6}{'ALL modes':>12}{'ACTIVE only':>13}{'SILENCED only':>15}"
          f"{'active-all':>12}")
    rows = []
    for nz in [0.0, 0.05, 0.10]:
        a = np.array(results[nz]["all"]); b = np.array(results[nz]["active"])
        c = np.array(results[nz]["silenced"])
        gain = b.mean() - a.mean()
        p = stats.ttest_rel(b, a).pvalue if len(b) > 1 else float("nan")
        rows.append((nz, a.mean(), b.mean(), c.mean(), gain, p))
        print(f"  {nz*100:>5.0f}%{a.mean():>12.3f}{b.mean():>13.3f}{c.mean():>15.3f}"
              f"{gain:>+12.3f}")
    return rows, len(active)


def main():
    print("=" * 82)
    print("BANDGAP ISOLATION -- does keeping only the int(phi^3)-active modes help?")
    print("=" * 82)
    sq = cavity_modes(periodic_rect(9, 9, L0, L0), L0, L0, 24, 24, 0.0)
    par = cavity_modes(periodic_rect(9, 9, L0, L0), L0, L0, 24, 24, 0.35)
    qc = cavity_modes(debruijn_quasicrystal_points(8, L0, L0, offset_seed=42), L0, L0, 24, 24, 0.0)

    out = {}
    for plate, nm in [(sq, "square (D4)"), (par, "parallelogram (C2)"), (qc, "quasicrystal")]:
        out[nm] = run_substrate(plate, nm)

    # ---- verdict ----
    print("\n" + "=" * 82)
    print("VERDICT -- is bandgap isolation of active modes a real win?")
    print("=" * 82)
    for nm, (rows, nact) in out.items():
        clean = rows[0]; noisy = rows[2]   # noise 0% and 10%
        print(f"  {nm:<20} active modes={nact:>2}:")
        print(f"     clean (0% noise):  active-only {clean[2]:+.3f} vs all {clean[1]:+.3f} "
              f"-> gain {clean[4]:+.3f}")
        print(f"     noisy (10% noise): active-only {noisy[2]:+.3f} vs all {noisy[1]:+.3f} "
              f"-> gain {noisy[4]:+.3f}")
    # silenced-only should be ~0 everywhere
    sil_clean = np.mean([out[nm][0][0][3] for nm in out])
    print(f"\n  SILENCED-only even-task R^2 (mean, clean): {sil_clean:+.3f}  "
          f"(should be ~0 -> they carry no product-generating term)")
    # is isolation a clean win?
    clean_gains = [out[nm][0][0][4] for nm in out]
    noisy_gains = [out[nm][0][2][4] for nm in out]
    print(f"  mean active-minus-all gain:  clean {np.mean(clean_gains):+.3f}   "
          f"noisy {np.mean(noisy_gains):+.3f}")
    if np.mean(noisy_gains) > 0.01 and np.mean(clean_gains) > -0.02:
        print("\n  => CLAIM HOLDS (with nuance): the silenced modes carry NO even-order")
        print("     signal (silenced-only fails), and dropping them does not hurt the clean")
        print("     even task while it HELPS under readout noise -- a real SNR benefit. So")
        print("     bandgap isolation of the active modes is a sound, if modest, idea.")
    elif np.mean(clean_gains) < -0.03:
        print("\n  => CLAIM WRONG: dropping the silenced modes HURTS -- they were providing")
        print("     memory the active modes need. Isolation is counterproductive.")
    else:
        print("\n  => MIXED: isolation is roughly neutral; no clear SNR win in this model.")
    print("=" * 82)


if __name__ == "__main__":
    main()
