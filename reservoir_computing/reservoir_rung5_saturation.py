#!/usr/bin/env python3
"""
RUNG 5 -- CONFIRM THE MECHANISM (not fish for a win).

Rungs 2-4 found the quasicrystal's structural richness real but inert. The
PROPOSED EXPLANATION was: a periodic plate is already "rich enough", so the
quasicrystal's extra richness sits past saturation and buys nothing. That is
a claim, and a claim must be tested, not just asserted.

THE TEST: dial the one ingredient that differs between the plates -- the
inter-mode COUPLING (2.4x denser for the quasicrystal in rung 4) -- from
ZERO up to full strength, and race the two plates at every level.

    coupling strength gamma:
        0.0  = each mode is an INDEPENDENT nonlinear oscillator (its own
               shape-derived self-quadratic/-cubic terms, NO mode mixing).
               This is the deliberately IMPOVERISHED lower bound.
        1.0  = the full rung-4 model (modes coupled through the real spatial
               overlap of their shapes).
        >1.0 = over-driven coupling, to see the far side of any plateau.

WHAT EACH OUTCOME MEANS (stated before running):
  * If performance RISES with gamma then PLATEAUS  -> the reservoir really
    does respond to coupling/richness, so the test is SENSITIVE, and a
    plateau is the saturation we hypothesized. Showing BOTH plates land on
    the SAME plateau confirms "both are above saturation."
  * If the QC and periodic curves lie on top of each other at EVERY gamma --
    including the sensitive mid-rise where coupling is being actively added,
    and the impoverished gamma=0 end -- then the coupling STRUCTURE genuinely
    does not matter, confirming the mechanism rather than fishing for a win.
  * If they DIVERGE at low gamma and converge high -> structure matters when
    scarce; that would REVISE the story. Honest either way.

This does NOT hunt for a configuration where the quasicrystal wins: gamma
scales coupling identically for both plates, and we pre-commit to reporting
the curves whatever shape they take.
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

# sweep / ensemble settings (a touch lighter than rung 4 -- many runs)
GAMMAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.3]
L = 1600
WASHOUT = 200
N_TRAIN = 1000
_drng = np.random.default_rng(7)
DRIVE_FRACS = [tuple(p) for p in _drng.uniform(0.22, 0.78, size=(12, 2))]


def run_reservoir_gamma(omega, Phi, aw, c2, c3, w_in, u_series, gamma):
    """Mode-shape reservoir with COUPLING scaled by gamma.

    Nonlinear force on mode i = SELF term (mode i alone) + gamma*(FULL field
    projection - SELF). gamma=0 -> uncoupled oscillators; gamma=1 -> full
    rung-4 coupling. c2,c3 are the per-mode self-coefficients
    (c2_i = integral phi_i^3 dA, c3_i = integral phi_i^4 dA)."""
    N = len(omega)
    x = np.zeros(N); v = np.zeros(N)
    n_sub = int(round(TAU_IN / DT))
    feats = np.empty((len(u_series), 2 * N + 1))
    for n, u in enumerate(u_series):
        for _ in range(n_sub):
            W = Phi.T @ x
            full2 = Phi @ (aw * W * W)
            full3 = Phi @ (aw * W * W * W)
            self2 = c2 * x * x
            self3 = c3 * x * x * x
            f2 = self2 + gamma * (full2 - self2)
            f3 = self3 + gamma * (full3 - self3)
            accel = (-(omega**2) * x - 2 * ZETA * omega * v
                     - A_QUAD * f2 - B_CUBIC * f3 + w_in * u)
            v = v + accel * DT
            x = x + v * DT
        feats[n, :N] = x
        feats[n, N:2*N] = v
        feats[n, -1] = 1.0
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"blew up at sample {n}, gamma={gamma}")
    return feats


def eval_task(states, target):
    Xtr, Ytr = states[WASHOUT:WASHOUT + N_TRAIN], target[WASHOUT:WASHOUT + N_TRAIN]
    Xte, Yte = states[WASHOUT + N_TRAIN:], target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None])
    return r2(Yte, (Xte @ W)[:, 0])


def race_at_gamma(freqs, Phi, aw, c2, c3, centers, u, yA, gamma):
    omega = OMEGA_LO + (OMEGA_HI - OMEGA_LO) * (freqs - freqs.min()) / (freqs.max() - freqs.min())
    nl, mc = [], []
    for frac in DRIVE_FRACS:
        e = nearest_elem(centers, frac)
        w_in = Phi[:, e].copy()
        states = run_reservoir_gamma(omega, Phi, aw, c2, c3, w_in, u, gamma)
        nl.append(eval_task(states, yA))
        m = 0.0
        for k in range(1, 16):
            yk = np.zeros(len(u)); yk[k:] = u[:len(u) - k]
            m += max(0.0, eval_task(states, yk))
        mc.append(m)
    return np.array(nl), np.array(mc)


def self_coeffs(Phi, aw):
    c2 = (aw[None, :] * Phi**3).sum(axis=1)   # integral phi_i^3 dA
    c3 = (aw[None, :] * Phi**4).sum(axis=1)   # integral phi_i^4 dA
    return c2, c3


def main():
    print("=" * 76)
    print("RUNG 5 -- saturation test: does coupling STRUCTURE matter anywhere?")
    print("=" * 76)

    print(f"\nComputing real FEM mode shapes ({N_MODES} modes, {TARGET_COV}% coverage)...")
    qc_holes = debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED)
    qc_r, _ = coverage_match_radius(qc_holes, TARGET_COV, NX)
    qc_f, qc_Phi, qc_aw, qc_ctr = mode_shapes(qc_holes, qc_r, N_MODES, NX)
    qc_c2, qc_c3 = self_coeffs(qc_Phi, qc_aw)

    per_holes = generate_periodic_holes(9, Lx, Ly)
    per_r, _ = coverage_match_radius(per_holes, TARGET_COV, NX)
    per_f, per_Phi, per_aw, per_ctr = mode_shapes(per_holes, per_r, N_MODES, NX)
    per_c2, per_c3 = self_coeffs(per_Phi, per_aw)
    print("done.\n")

    rng = np.random.default_rng(0)
    u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)
    yA = np.zeros(L); yA[2:] = u[1:L-1] * u[0:L-2]

    print(f"{'gamma':>6} | {'QC nl R^2':>16} {'per nl R^2':>16} {'nl winner':>10} | "
          f"{'QC mem':>8} {'per mem':>8} {'mem winner':>11}")
    print("-" * 92)

    res = []
    for g in GAMMAS:
        qn, qm = race_at_gamma(qc_f, qc_Phi, qc_aw, qc_c2, qc_c3, qc_ctr, u, yA, g)
        pn, pm = race_at_gamma(per_f, per_Phi, per_aw, per_c2, per_c3, per_ctr, u, yA, g)

        def who(a, b):
            d = a.mean() - b.mean(); pooled = np.hypot(a.std(), b.std()) + 1e-9
            if abs(d) / pooled < 1.0:
                return "tie"
            return ("QC" if d > 0 else "periodic") + f"({abs(d)/pooled:.1f}x)"

        res.append((g, qn.mean(), qn.std(), pn.mean(), pn.std(),
                    qm.mean(), qm.std(), pm.mean(), pm.std()))
        print(f"{g:>6.2f} | {qn.mean():>8.3f}+/-{qn.std():<6.3f} "
              f"{pn.mean():>8.3f}+/-{pn.std():<6.3f} {who(qn, pn):>10} | "
              f"{qm.mean():>8.2f} {pm.mean():>8.2f} {who(qm, pm):>11}")

    res = np.array(res)

    # ---- plot the saturation curves ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.errorbar(res[:, 0], res[:, 1], yerr=res[:, 2], fmt="o-", color="#2E5E8C",
                capsize=3, label="quasicrystal")
    a1.errorbar(res[:, 0], res[:, 3], yerr=res[:, 4], fmt="s-", color="#C0392B",
                capsize=3, label="periodic")
    a1.axvline(1.0, color="gray", ls=":", lw=1)
    a1.text(0.0, a1.get_ylim()[0], " uncoupled\n (impoverished)", fontsize=8, va="bottom")
    a1.set_title("Nonlinear task vs coupling strength")
    a1.set_xlabel("coupling strength gamma (0=uncoupled, 1=full rung-4)")
    a1.set_ylabel("nonlinear task R^2"); a1.legend(); a1.grid(alpha=0.3)

    a2.errorbar(res[:, 0], res[:, 5], yerr=res[:, 6], fmt="o-", color="#2E5E8C",
                capsize=3, label="quasicrystal")
    a2.errorbar(res[:, 0], res[:, 7], yerr=res[:, 8], fmt="s-", color="#C0392B",
                capsize=3, label="periodic")
    a2.axvline(1.0, color="gray", ls=":", lw=1)
    a2.set_title("Memory capacity vs coupling strength")
    a2.set_xlabel("coupling strength gamma"); a2.set_ylabel("memory capacity")
    a2.legend(); a2.grid(alpha=0.3)

    fig.suptitle("Rung 5: dialing coupling from impoverished -> full. Does the "
                 "quasicrystal's 2.4x-denser coupling ever matter?", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "reservoir_rung5_saturation.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved {out}")

    # ---- honest verdict ----
    nl_rise = res[:, 1].max() - res[res[:, 0] == 0.0, 1][0]
    # largest QC-minus-periodic gap on the nonlinear task across all gamma, in spread units
    gaps = (res[:, 1] - res[:, 3]) / (np.hypot(res[:, 2], res[:, 4]) + 1e-9)
    biggest = gaps[np.argmax(np.abs(gaps))]
    g_at = res[np.argmax(np.abs(gaps)), 0]
    print("\n" + "=" * 76)
    print("HONEST VERDICT (rung 5)")
    print("=" * 76)
    print(f"  Nonlinear-task rise from uncoupled->best: {nl_rise:+.3f} R^2 "
          f"-> the reservoir {'DOES' if nl_rise > 0.05 else 'does NOT clearly'} "
          f"respond to coupling (test {'is' if nl_rise > 0.05 else 'may not be'} sensitive).")
    print(f"  Largest QC-vs-periodic separation across ALL gamma: {biggest:+.1f}x spread "
          f"(at gamma={g_at:.2f}).")
    if abs(biggest) < 1.0:
        print("  => The two plates track each other at EVERY coupling level, including")
        print("     the sensitive mid-rise and the impoverished gamma=0 end. The coupling")
        print("     STRUCTURE does not matter anywhere. MECHANISM CONFIRMED: both plates")
        print("     sit on the same saturation curve; the quasicrystal's extra coupling")
        print("     density is genuinely redundant, not merely unused at full strength.")
    else:
        print("  => They SEPARATE somewhere -- structure matters in some regime. This")
        print("     REVISES the saturation story; report where and how much, honestly.")
    print("=" * 76)


if __name__ == "__main__":
    main()
