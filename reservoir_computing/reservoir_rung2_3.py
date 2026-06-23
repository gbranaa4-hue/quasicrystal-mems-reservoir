#!/usr/bin/env python3
"""
PHYSICAL RESERVOIR COMPUTING -- rungs 2 & 3 (the discovery attempt).

THE QUESTION (genuinely open -- nobody fed me the answer):
  Does a QUASICRYSTAL plate's mode structure make a BETTER reservoir than
  a PERIODIC plate's, holding everything else equal?

HOW THIS IS CONTROLLED (so a difference, if any, means something):
  1. Pull the REAL mode spectra (first N natural frequencies) from the
     project's FEM for two plates -- one quasicrystal-perforated, one
     periodic-perforated -- MATCHED TO THE SAME COVERAGE. So the only
     structural difference is periodic vs. aperiodic arrangement.
  2. Normalize BOTH spectra to the same frequency range, so the comparison
     isolates the INTERNAL DISTRIBUTION/SPACING of the modes (the
     quasicrystal-vs-periodic signature), not just absolute scale.
  3. Build IDENTICAL nonlinear reservoirs that differ ONLY in their mode
     frequencies (same coupling, same input weights, same nonlinearity,
     same everything else, for each random seed).
  4. Average over several random reservoir seeds and report mean +/- std --
     a difference only counts if it exceeds that spread (same signal-vs-noise
     discipline as the MEMS paper).

HONEST SCOPE (do not overclaim):
  The mode FREQUENCIES are real (FEM). The nonlinear coupling that turns the
  modes into a reservoir is a GENERIC model, identical for both cases -- so
  this isolates the effect of the mode-frequency DISTRIBUTION specifically,
  not the plate's true nonlinear coefficients (which would need a full
  nonlinear FEM, a further rung). This asks: "does the quasicrystal mode
  SPECTRUM help?" -- a real, novel, and answerable question -- not "does a
  fabricated quasicrystal reservoir work."
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plate_bending_review",
)
sys.path.insert(0, FEM_DIR)
from fem_plate_bending_homogenized import (  # noqa: E402
    Lx, Ly, build_mesh, element_coverage_fractions, assemble,
    clamped_free_dofs, solve_modes, debruijn_quasicrystal_points,
)

# ---- FEM / spectrum settings ----
NX = 28
N_MODES = 40
TARGET_COV = 85.0          # same coverage for both plates (fair comparison)
QC_NFOLD = 8
QC_SEED = 42

# ---- reservoir settings (same tuned regime as rung 1) ----
ZETA = 0.20
BETA2 = 0.6
BETA3 = 1.0
TAU_IN = 1.0
DT = 0.02
INPUT_AMP = 1.5
OMEGA_LO, OMEGA_HI = 0.5, 2.5   # both spectra normalized into this range
L = 2000
WASHOUT = 200
N_TRAIN = 1200
RES_SEEDS = [1, 2, 3, 4, 5, 6]  # random reservoir realizations to average over


# ------------------------------------------------------------------ geometry
def generate_periodic_holes(n_side, domain_x, domain_y):
    """A regular square grid of hole centers in the central region of the
    domain -- the periodic counterpart to the quasicrystal point set."""
    span = 0.9 * min(domain_x, domain_y)
    xs = np.linspace(domain_x / 2 - span / 2, domain_x / 2 + span / 2, n_side)
    ys = np.linspace(domain_y / 2 - span / 2, domain_y / 2 + span / 2, n_side)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def coverage_match_radius(holes, target_cov, nx, sub_n=12,
                          r_lo=0.2e-6, r_hi=15.0e-6, tol=0.4, max_iter=40):
    nodes, quads = build_mesh(Lx, Ly, nx, nx)

    def cov_at(r):
        return element_coverage_fractions(nodes, quads, holes, r, sub_n=sub_n).mean() * 100

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


def mode_spectrum(holes, radius, n_modes, nx):
    nodes, quads = build_mesh(Lx, Ly, nx, nx)
    phi = element_coverage_fractions(nodes, quads, holes, radius, sub_n=12)
    K, M = assemble(nodes, quads, phi=phi, stiffness_exponent=2.0)
    free = clamped_free_dofs(nodes)
    freqs = solve_modes(K, M, free, n_modes=n_modes)
    return np.sort(np.asarray(freqs))[:n_modes]


def normalize_spectrum(freqs, lo, hi):
    f = np.asarray(freqs, float)
    return lo + (hi - lo) * (f - f.min()) / (f.max() - f.min())


# ------------------------------------------------------------------ reservoir
def build_coupling(N, seed):
    rng = np.random.default_rng(seed)
    w_in = rng.uniform(-1.0, 1.0, N)
    C = rng.uniform(-1.0, 1.0, (N, N)) * (rng.random((N, N)) < 0.2)
    np.fill_diagonal(C, 0.0)
    C *= 0.08
    return w_in, C


def run_reservoir(omega, w_in, C, u_series, beta2, beta3):
    N = len(omega)
    rowsum = C.sum(axis=1)
    x = np.zeros(N); v = np.zeros(N)
    n_sub = int(round(TAU_IN / DT))
    feats = np.empty((len(u_series), 2 * N + 1))
    for n, u in enumerate(u_series):
        for _ in range(n_sub):
            coupling = C @ x - rowsum * x
            accel = (-(omega**2) * x - 2 * ZETA * omega * v
                     - beta2 * x**2 - beta3 * x**3 + w_in * u + coupling)
            v = v + accel * DT
            x = x + v * DT
        feats[n, :N] = x
        feats[n, N:2*N] = v
        feats[n, -1] = 1.0
        if not np.all(np.isfinite(x)):
            raise RuntimeError("reservoir blew up")
    return feats


def ridge_fit(X, Y, lam=1e-6):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def r2(yt, yp):
    ss = np.sum((yt - yp) ** 2); tot = np.sum((yt - yt.mean()) ** 2)
    return 1.0 - ss / tot if tot > 0 else 0.0


def eval_task(states, target):
    Xtr, Ytr = states[WASHOUT:WASHOUT + N_TRAIN], target[WASHOUT:WASHOUT + N_TRAIN]
    Xte, Yte = states[WASHOUT + N_TRAIN:], target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None])
    return r2(Yte, (Xte @ W)[:, 0])


def benchmark(omega, u, yA):
    """Return (nonlinear-task R^2, memory capacity) averaged over reservoir seeds."""
    nl_scores, mc_scores = [], []
    for s in RES_SEEDS:
        w_in, C = build_coupling(len(omega), s)
        states = run_reservoir(omega, w_in, C, u, BETA2, BETA3)
        nl_scores.append(eval_task(states, yA))
        mc = 0.0
        for k in range(1, 16):
            yk = np.zeros(len(u)); yk[k:] = u[:len(u) - k]
            mc += max(0.0, eval_task(states, yk))
        mc_scores.append(mc)
    return np.array(nl_scores), np.array(mc_scores)


def main():
    print("=" * 72)
    print("RESERVOIR COMPUTING rungs 2&3 -- quasicrystal vs periodic mode spectrum")
    print("=" * 72)

    # ---- real mode spectra from the FEM, matched coverage ----
    print(f"\n=== Step 1: real FEM mode spectra (first {N_MODES} modes, {TARGET_COV}% coverage) ===")
    qc_holes = debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED)
    qc_r, qc_cov = coverage_match_radius(qc_holes, TARGET_COV, NX)
    qc_spec = mode_spectrum(qc_holes, qc_r, N_MODES, NX)
    print(f"  quasicrystal (n={QC_NFOLD}): {len(qc_holes)} holes, r={qc_r*1e6:.3f}um, "
          f"cov={qc_cov:.1f}%, {len(qc_spec)} modes "
          f"[{qc_spec.min()/1e6:.4f}..{qc_spec.max()/1e6:.4f} MHz]")

    per_holes = generate_periodic_holes(9, Lx, Ly)
    per_r, per_cov = coverage_match_radius(per_holes, TARGET_COV, NX)
    per_spec = mode_spectrum(per_holes, per_r, N_MODES, NX)
    print(f"  periodic (9x9 grid): {len(per_holes)} holes, r={per_r*1e6:.3f}um, "
          f"cov={per_cov:.1f}%, {len(per_spec)} modes "
          f"[{per_spec.min()/1e6:.4f}..{per_spec.max()/1e6:.4f} MHz]")

    # a quick structural metric: how many NEAR-DEGENERATE mode pairs each has
    def near_deg(spec, tol_frac=0.005):
        s = np.sort(spec); gaps = np.diff(s) / s[:-1]
        return int(np.sum(gaps < tol_frac))
    print(f"  near-degenerate adjacent mode pairs:  quasicrystal={near_deg(qc_spec)}, "
          f"periodic={near_deg(per_spec)}")

    # ---- normalize both spectra to the same range (isolate the distribution) ----
    qc_w = normalize_spectrum(qc_spec, OMEGA_LO, OMEGA_HI)
    per_w = normalize_spectrum(per_spec, OMEGA_LO, OMEGA_HI)

    # ---- race them ----
    print(f"\n=== Step 2: race the two reservoirs ({len(RES_SEEDS)} seeds each, "
          f"normalized to identical range) ===")
    rng = np.random.default_rng(0)
    u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)
    yA = np.zeros(L); yA[2:] = u[1:L-1] * u[0:L-2]   # nonlinear task

    qc_nl, qc_mc = benchmark(qc_w, u, yA)
    per_nl, per_mc = benchmark(per_w, u, yA)

    print(f"\n  {'metric':<26}{'quasicrystal':>18}{'periodic':>16}")
    print("  " + "-" * 58)
    print(f"  {'nonlinear task R^2':<26}{qc_nl.mean():>10.3f} +/-{qc_nl.std():<5.3f}"
          f"{per_nl.mean():>9.3f} +/-{per_nl.std():<5.3f}")
    print(f"  {'memory capacity':<26}{qc_mc.mean():>10.2f} +/-{qc_mc.std():<5.2f}"
          f"{per_mc.mean():>9.2f} +/-{per_mc.std():<5.2f}")

    def verdict(qc, per, name):
        d = qc.mean() - per.mean()
        pooled = np.hypot(qc.std(), per.std()) + 1e-9
        sig = abs(d) / pooled
        who = "quasicrystal" if d > 0 else "periodic"
        if sig < 1.0:
            return f"  {name}: difference {d:+.3f} is WITHIN noise ({sig:.1f}x spread) -> no clear winner"
        return f"  {name}: {who} better by {abs(d):.3f} ({sig:.1f}x the spread) -> real difference"

    print()
    print(verdict(qc_nl, per_nl, "nonlinear task"))
    print(verdict(qc_mc, per_mc, "memory capacity"))

    # ---- plots ----
    fig, (axS, axB) = plt.subplots(1, 2, figsize=(13, 5))
    axS.plot(np.sort(qc_spec) / 1e6, "o-", ms=3, label="quasicrystal", color="#2E5E8C")
    axS.plot(np.sort(per_spec) / 1e6, "s-", ms=3, label="periodic", color="#C0392B")
    axS.set_title("Real FEM mode spectra (matched coverage)")
    axS.set_xlabel("mode index"); axS.set_ylabel("frequency (MHz)")
    axS.legend(); axS.grid(alpha=0.3)

    x = np.arange(2)
    w = 0.35
    axB.bar(x - w/2, [qc_nl.mean(), qc_mc.mean() / 15], w, yerr=[qc_nl.std(), qc_mc.std() / 15],
            label="quasicrystal", color="#2E5E8C", capsize=4)
    axB.bar(x + w/2, [per_nl.mean(), per_mc.mean() / 15], w, yerr=[per_nl.std(), per_mc.std() / 15],
            label="periodic", color="#C0392B", capsize=4)
    axB.set_xticks(x); axB.set_xticklabels(["nonlinear task\n(R^2)", "memory capacity\n(/15, normalized)"])
    axB.set_title("Reservoir performance (mean +/- std over seeds)")
    axB.legend(); axB.grid(alpha=0.3, axis="y")

    fig.suptitle("Does the quasicrystal mode spectrum make a better reservoir? "
                 "(real FEM frequencies; generic identical nonlinearity)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reservoir_rung2_3_results.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved {out}")

    print("\n" + "=" * 72)
    print("HONEST NOTE: frequencies REAL (FEM); nonlinearity is a GENERIC identical")
    print("model for both, so this isolates the effect of the mode-frequency")
    print("DISTRIBUTION, not the plate's true nonlinear dynamics. Whatever the result,")
    print("it is an honest finding about the spectrum -- not a fabricated-device claim.")
    print("=" * 72)


if __name__ == "__main__":
    main()
