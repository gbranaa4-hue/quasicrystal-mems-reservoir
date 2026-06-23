#!/usr/bin/env python3
"""
RUNG 5b -- WHY does the quasicrystal win in the UNCOUPLED limit?

Rung 5 produced a surprise: with coupling OFF (gamma=0, a bank of independent
shape-driven nonlinear oscillators), the quasicrystal beat the periodic plate
~1.6x on the nonlinear task -- and the edge vanished as coupling was added.
So the advantage is NOT the coupling network (confirmed inert). It must come
from the modes' INDIVIDUAL shape-derived properties.

HYPOTHESIS (ties the whole ladder together): a reservoir's power is set by how
many INDEPENDENT features its readout can use. With RANDOM input weights
(rungs 2-3) the periodic plate's 18 near-degenerate modes still got
independent kicks, so degeneracy was harmless. With SHAPE-DERIVED input
weights w_in=phi_i(x_drive), degenerate/symmetric modes get CORRELATED drive
and produce REDUNDANT feature columns -> wasted readout dimensionality. The
quasicrystal, with almost no degeneracy, wastes almost nothing.

PREDICTION (measured directly here, gamma=0):
  the quasicrystal's reservoir state has a HIGHER effective dimensionality
  (participation ratio of the feature covariance) and LESS feature
  redundancy than the periodic plate's -> which would explain the task gap.
If instead the effective dimensionalities are equal, this hypothesis is wrong
and the gap is something else -- reported honestly either way.

This explains an observed difference; it does not hunt for a new one.
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
_drng = np.random.default_rng(123)
DRIVE_FRACS = [tuple(p) for p in _drng.uniform(0.22, 0.78, size=(24, 2))]   # bigger ensemble


def participation_ratio(F):
    """Effective number of independent feature dimensions: (sum lambda)^2 /
    sum lambda^2 of the feature covariance. = full rank if all equal,
    collapses toward 1 as features become redundant/correlated."""
    Fc = F - F.mean(axis=0, keepdims=True)
    cov = (Fc.T @ Fc) / Fc.shape[0]
    lam = np.linalg.eigvalsh(cov)
    lam = lam[lam > 1e-12]
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def near_deg_pairs(freqs, tol_frac=0.02):
    s = np.argsort(freqs); f = freqs[s]
    pairs = []
    for i in range(len(f) - 1):
        if (f[i + 1] - f[i]) / f[i] < tol_frac:
            pairs.append((s[i], s[i + 1]))
    return pairs


def eval_task(states, target):
    Xtr, Ytr = states[WASHOUT:WASHOUT + N_TRAIN], target[WASHOUT:WASHOUT + N_TRAIN]
    Xte, Yte = states[WASHOUT + N_TRAIN:], target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None])
    return r2(Yte, (Xte @ W)[:, 0])


def analyze(name, freqs, Phi, aw, ctr, u, yA):
    omega = OMEGA_LO + (OMEGA_HI - OMEGA_LO) * (freqs - freqs.min()) / (freqs.max() - freqs.min())
    c2 = (aw[None, :] * Phi**3).sum(axis=1)
    c3 = (aw[None, :] * Phi**4).sum(axis=1)
    pairs = near_deg_pairs(freqs)

    r2s, prs, win_eff, dup = [], [], [], []
    for frac in DRIVE_FRACS:
        e = nearest_elem(ctr, frac)
        w_in = Phi[:, e].copy()
        states = run_reservoir_gamma(omega, Phi, aw, c2, c3, w_in, u, gamma=0.0)
        Xstate = states[WASHOUT:, :len(freqs)]            # mode displacements only
        r2s.append(eval_task(states, yA))
        prs.append(participation_ratio(states[WASHOUT:, :2 * len(freqs)]))
        # input utilization: effective # of modes the point drive excites
        wi2 = w_in ** 2
        win_eff.append((wi2.sum() ** 2) / (wi2 ** 2).sum())
        # redundancy among near-degenerate mode pairs: mean |corr| of their
        # feature columns (high = degenerate modes give duplicate features)
        if pairs:
            cc = []
            for (a_, b_) in pairs:
                ca, cb = Xstate[:, a_], Xstate[:, b_]
                d = ca.std() * cb.std()
                cc.append(abs(np.mean((ca - ca.mean()) * (cb - cb.mean())) / d) if d > 0 else 0.0)
            dup.append(np.mean(cc))
    r2s, prs, win_eff = np.array(r2s), np.array(prs), np.array(win_eff)
    print(f"  {name:<13} task R^2 = {r2s.mean():.3f}+/-{r2s.std():.3f} | "
          f"effective feature dim = {prs.mean():5.1f}+/-{prs.std():4.1f} | "
          f"modes excited by drive = {win_eff.mean():4.1f} | "
          f"near-deg pairs = {len(pairs)} (redundancy "
          f"{np.mean(dup):.2f})" if pairs else
          f"  {name:<13} task R^2 = {r2s.mean():.3f}+/-{r2s.std():.3f} | "
          f"effective feature dim = {prs.mean():5.1f}+/-{prs.std():4.1f} | "
          f"modes excited by drive = {win_eff.mean():4.1f} | near-deg pairs = 0")
    return r2s, prs, win_eff, (np.mean(dup) if pairs else 0.0), len(pairs)


def main():
    print("=" * 78)
    print("RUNG 5b -- WHY the quasicrystal wins uncoupled: effective dimensionality")
    print("=" * 78)
    print(f"\nComputing real FEM mode shapes ({N_MODES} modes)... ({len(DRIVE_FRACS)} drives)")
    qc_holes = debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED)
    qc_r, _ = coverage_match_radius(qc_holes, TARGET_COV, NX)
    qc_f, qc_Phi, qc_aw, qc_ctr = mode_shapes(qc_holes, qc_r, N_MODES, NX)

    per_holes = generate_periodic_holes(9, Lx, Ly)
    per_r, _ = coverage_match_radius(per_holes, TARGET_COV, NX)
    per_f, per_Phi, per_aw, per_ctr = mode_shapes(per_holes, per_r, N_MODES, NX)
    print("done.\n")

    rng = np.random.default_rng(0)
    u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)
    yA = np.zeros(L); yA[2:] = u[1:L-1] * u[0:L-2]

    print("UNCOUPLED reservoir (gamma=0), 24 drive locations:")
    qr, qp, qw, qd, qnp = analyze("quasicrystal", qc_f, qc_Phi, qc_aw, qc_ctr, u, yA)
    pr, pp, pw, pd, pnp = analyze("periodic", per_f, per_Phi, per_aw, per_ctr, u, yA)

    # ---- verdict ----
    dr = qr.mean() - pr.mean(); sr = np.hypot(qr.std(), pr.std()) + 1e-9
    dpr = qp.mean() - pp.mean(); spr = np.hypot(qp.std(), pp.std()) + 1e-9
    print("\n" + "=" * 78)
    print("HONEST VERDICT (rung 5b)")
    print("=" * 78)
    print(f"  task gap:               QC - periodic = {dr:+.3f}  ({abs(dr)/sr:.1f}x spread)")
    print(f"  effective-dim gap:      QC - periodic = {dpr:+.1f}   ({abs(dpr)/spr:.1f}x spread)")
    print(f"  near-degenerate pairs:  QC={qnp} (redundancy {qd:.2f})  "
          f"periodic={pnp} (redundancy {pd:.2f})")
    print(f"  modes excited by drive: QC={qw.mean():.1f}  periodic={pw.mean():.1f}")
    explains = (dr > 0) and (dpr > 0) and (abs(dpr) / spr >= 1.0)
    if explains:
        print("\n  => CONFIRMED: the quasicrystal provides MORE independent features")
        print("     (higher effective dimensionality), and the periodic plate has more")
        print("     near-degenerate modes giving redundant features under shape-derived")
        print("     drive. The uncoupled task gap is explained by feature dimensionality,")
        print("     NOT by the nonlinear coupling network. This finally connects the")
        print("     rung-3 degeneracy count (3 vs 18) to a real performance consequence --")
        print("     but only in the UNCOUPLED, shape-driven regime, and the effect is")
        print("     modest. An honest, mechanism-level explanation, not a device claim.")
    else:
        print("\n  => NOT explained by effective dimensionality -- the uncoupled gap comes")
        print("     from something else (self-nonlinearity coefficients, frequency")
        print("     spacing). Reported honestly; hypothesis not supported.")
    print("=" * 78)


if __name__ == "__main__":
    main()
