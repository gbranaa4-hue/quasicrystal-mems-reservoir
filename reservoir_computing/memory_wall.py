#!/usr/bin/env python3
"""
THE MEMORY WALL -- the make-or-break test for any sequence-model ambition.

A language/sequence model needs dependencies over hundreds-plus of steps. Our
reservoir's product depth died by d~5 at the default damping. Before any
"hybrid LLM" talk is worth a line of code, one question must be answered
honestly: how deep can this reservoir remember AT ALL, when we push the knobs
that control memory as hard as they go -- and does the even-order symmetry edge
survive anywhere near a useful depth?

Memory in a damped oscillator decays as exp(-zeta*omega*k) over k input
samples, so the lever is the DAMPING zeta (the analog of an echo-state leak
rate): memory depth ~ 1/(zeta*omega). We sweep zeta down from 0.20 to 0.01
(quality factor Q=1/2zeta from 2.5 to 50) and measure:
  (A) LINEAR memory depth -- how far back u[n-k] can be reconstructed;
  (B) the EVEN-ORDER PRODUCT depth -- how deep u[n-1]*u[n-d] can be computed
      (this needs memory AND nonlinearity together, the harder demand);
  (C) whether the quasicrystal-vs-periodic even-order edge survives once we
      have tuned for long memory -- or whether the memory-nonlinearity
      tradeoff (capacity is conserved; Dambre et al. 2012) closes it.
Also watches for the OTHER wall: at low damping a driven nonlinear resonator
loses the fading-memory (echo-state) property to instability/chaos.
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
    nearest_elem, ridge_fit, r2, QC_NFOLD, QC_SEED, A_QUAD, B_CUBIC, TAU_IN, DT,
)
from reservoir_rung6_stresstest import make_plate, omega_of  # noqa: E402

L = 2500
WASHOUT = 300
N_TRAIN = 1500
INPUT_AMP = 0.8
ZETAS = [0.20, 0.10, 0.05, 0.02, 0.01]
KMAX = 50
DEPTHS = [2, 3, 5, 8, 12, 18, 25, 35]
_drng = np.random.default_rng(17)
DRIVE_FRACS = [tuple(p) for p in _drng.uniform(0.22, 0.78, size=(5, 2))]


def run_uncoupled_z(omega, c2, c3, w_in, u, zeta):
    N = len(omega)
    x = np.zeros(N); v = np.zeros(N)
    n_sub = int(round(TAU_IN / DT))
    o2 = omega**2; damp = 2 * zeta * omega
    feats = np.empty((len(u), 2 * N + 1))
    blew = False
    for n, uu in enumerate(u):
        drive = w_in * uu
        for _ in range(n_sub):
            accel = -o2 * x - damp * v - A_QUAD * c2 * x * x - B_CUBIC * c3 * x * x * x + drive
            v = v + accel * DT
            x = x + v * DT
        feats[n, :N] = x; feats[n, N:2*N] = v; feats[n, -1] = 1.0
        if not np.all(np.isfinite(x)):
            blew = True; feats[n:] = 0.0; feats[n:, -1] = 1.0
            break
    return feats, blew


def split_eval(states, target):
    Xtr, Ytr = states[WASHOUT:WASHOUT + N_TRAIN], target[WASHOUT:WASHOUT + N_TRAIN]
    Xte, Yte = states[WASHOUT + N_TRAIN:], target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None], lam=1e-6)
    return r2(Yte, (Xte @ W)[:, 0])


def echo_state_ok(omega, c2, c3, w_in, zeta, u):
    """Fading-memory check: two runs from different initial states must
    converge (the reservoir must FORGET its start). Returns final-state
    divergence; ~0 means the echo-state property holds."""
    N = len(omega); n_sub = int(round(TAU_IN / DT))
    o2 = omega**2; damp = 2 * zeta * omega
    xs = [np.zeros(N), np.random.default_rng(3).uniform(-1, 1, N)]
    vs = [np.zeros(N), np.zeros(N)]
    for uu in u:
        drive = w_in * uu
        for s in range(2):
            x, v = xs[s], vs[s]
            for _ in range(n_sub):
                a = -o2 * x - damp * v - A_QUAD * c2 * x * x - B_CUBIC * c3 * x * x * x + drive
                v = v + a * DT; x = x + v * DT
            xs[s], vs[s] = x, v
    if not (np.all(np.isfinite(xs[0])) and np.all(np.isfinite(xs[1]))):
        return np.inf
    return float(np.linalg.norm(xs[0] - xs[1]) / (np.linalg.norm(xs[0]) + 1e-9))


def profile(plate, zeta, u, kmax):
    """Average over drives: linear R^2(k) curve and product R^2(d) curve."""
    omega = omega_of(plate)
    mem = np.zeros(kmax + 1); prod = {d: [] for d in DEPTHS}
    blon = 0
    for fr in DRIVE_FRACS:
        e = nearest_elem(plate["ctr"], fr); w_in = plate["Phi"][:, e].copy()
        st, blew = run_uncoupled_z(omega, plate["c2"], plate["c3"], w_in, u, zeta)
        blon += int(blew)
        mk = np.zeros(kmax + 1)
        for k in range(1, kmax + 1):
            yk = np.zeros(len(u)); yk[k:] = u[:len(u) - k]
            mk[k] = max(0.0, split_eval(st, yk))
        mem += mk
        for d in DEPTHS:
            yd = np.zeros(len(u)); yd[d:] = u[1:len(u) - d + 1] * u[0:len(u) - d]
            prod[d].append(split_eval(st, yd))
    mem /= len(DRIVE_FRACS)
    prodm = {d: np.mean(prod[d]) for d in DEPTHS}
    return mem, prodm, blon


def depth_at(curve_x, curve_y, thr):
    """Largest x where y still exceeds thr (linear interp on the crossing)."""
    last = 0.0
    for x, y in zip(curve_x, curve_y):
        if y >= thr:
            last = x
    return last


def main():
    print("=" * 84)
    print("THE MEMORY WALL -- pushing damping to extend memory; does the edge survive?")
    print("=" * 84)
    rng = np.random.default_rng(0)
    u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)

    print("\nBuilding plates (85% coverage, N=40)...")
    qc = make_plate(debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED), 85.0)
    pe = make_plate(generate_periodic_holes(9, Lx, Ly), 85.0)
    print("done.\n")

    print(f"{'zeta':>6}{'Q':>6}{'ESP?':>6} | "
          f"{'lin k(R2>.5)':>13}{'lin k(R2>.1)':>13}{'tot MC':>8} | "
          f"{'prod depth(R2>.3)':>18}{'best prod R2':>13}")
    print("-" * 84)
    qc_mem_by_zeta = {}
    rows = []
    for z in ZETAS:
        esp = echo_state_ok(omega_of(qc), qc["c2"], qc["c3"],
                            qc["Phi"][:, nearest_elem(qc["ctr"], DRIVE_FRACS[0])].copy(), z, u)
        mem, prodm, blon = profile(qc, z, u, KMAX)
        qc_mem_by_zeta[z] = mem
        ks = np.arange(KMAX + 1)
        k50 = depth_at(ks, mem, 0.5); k10 = depth_at(ks, mem, 0.1)
        totmc = mem.sum()
        pdepth = depth_at(DEPTHS, [prodm[d] for d in DEPTHS], 0.3)
        bestp = max(prodm.values())
        espf = "yes" if esp < 0.05 else ("NO" if np.isfinite(esp) else "blew")
        rows.append((z, k50, k10, totmc, pdepth, bestp))
        print(f"{z:>6.2f}{1/(2*z):>6.1f}{espf:>6} | "
              f"{k50:>13.0f}{k10:>13.0f}{totmc:>8.1f} | "
              f"{pdepth:>18.0f}{bestp:>13.3f}"
              f"{'  (some runs unstable)' if blon else ''}")

    # ---- edge test: at the LONGEST-memory STABLE zeta, does QC still beat periodic? ----
    # pick the smallest zeta whose ESP held (fading memory intact)
    stable_zetas = [z for z in ZETAS
                    if echo_state_ok(omega_of(qc), qc["c2"], qc["c3"],
                                     qc["Phi"][:, nearest_elem(qc["ctr"], DRIVE_FRACS[0])].copy(), z, u) < 0.05]
    z_best = min(stable_zetas) if stable_zetas else ZETAS[0]
    print("\n" + "-" * 84)
    print(f"EDGE-AT-DEPTH test at the longest-memory STABLE damping zeta={z_best} "
          f"(Q={1/(2*z_best):.0f}):")
    print("-" * 84)
    qm, qp, _ = profile(qc, z_best, u, KMAX)
    pm, pp, _ = profile(pe, z_best, u, KMAX)
    # linear recall at a deep lag, and the deepest even product that is doable
    for k in [5, 10, 20]:
        print(f"  linear recall u[n-{k:>2}]:   QC R2={qm[k]:.3f}   periodic R2={pm[k]:.3f}   "
              f"gap {qm[k]-pm[k]:+.3f}")
    print("  even product u[n-1]*u[n-d]:")
    for d in DEPTHS:
        g = qp[d] - pp[d]
        tag = "QC" if g > 0.02 else ("per" if g < -0.02 else "~tie")
        print(f"    d={d:>2}:  QC R2={qp[d]:.3f}   periodic R2={pp[d]:.3f}   gap {g:+.3f}  {tag}")

    # ---- plot ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for z in ZETAS:
        a1.plot(np.arange(KMAX + 1), qc_mem_by_zeta[z], "-", label=f"zeta={z} (Q={1/(2*z):.0f})")
    a1.axhline(0.5, color="gray", ls=":", lw=1)
    a1.set_title("Linear memory profile vs damping (quasicrystal)")
    a1.set_xlabel("delay k (input samples back)"); a1.set_ylabel("reconstruction R^2 of u[n-k]")
    a1.legend(fontsize=8); a1.grid(alpha=0.3); a1.set_ylim(0, 1.02)

    zz = [r[0] for r in rows]
    a2.plot(zz, [r[1] for r in rows], "o-", color="#2E5E8C", label="linear depth (R^2>0.5)")
    a2.plot(zz, [r[4] for r in rows], "s-", color="#C0392B", label="EVEN-PRODUCT depth (R^2>0.3)")
    a2.set_xscale("log"); a2.invert_xaxis()
    a2.set_title("Memory wall: linear vs even-order product depth")
    a2.set_xlabel("damping zeta (lower = more memory ->)"); a2.set_ylabel("achievable depth (samples)")
    a2.legend(fontsize=8); a2.grid(alpha=0.3)

    fig.suptitle("The memory wall: how deep can this reservoir remember, and compute products?",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "memory_wall.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved {out}")

    # ---- verdict ----
    best_lin = max(r[1] for r in rows)
    best_prod = max(r[4] for r in rows)
    print("\n" + "=" * 84)
    print("HONEST VERDICT -- the memory wall")
    print("=" * 84)
    print(f"  deepest LINEAR memory (R^2>0.5) across all stable damping: {best_lin:.0f} samples")
    print(f"  deepest EVEN-ORDER PRODUCT (R^2>0.3) across all damping:    {best_prod:.0f} samples")
    print(f"  a sequence/language model needs dependencies over ~10^2-10^4 steps.")
    if best_prod < 20:
        print("  => The nonlinear (product) memory wall sits at single-to-low-double digits")
        print("     even after maximizing memory: the memory-nonlinearity tradeoff caps the")
        print("     useful depth far below sequence-model scale. Linear memory can be pushed")
        print("     deeper, but the EVEN-ORDER computation (the part the symmetry edge helps)")
        print("     does not follow. This is the honest floor: as a standalone sequence")
        print("     processor at language scale, the device is orders of magnitude short.")
    else:
        print("  => Product memory extends further than expected -- worth a closer look.")
    print("=" * 84)


if __name__ == "__main__":
    main()
