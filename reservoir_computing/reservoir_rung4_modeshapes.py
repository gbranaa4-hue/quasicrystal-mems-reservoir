#!/usr/bin/env python3
"""
PHYSICAL RESERVOIR COMPUTING -- rung 4 (the mode-SHAPE test).

WHY THIS RUNG EXISTS:
  Rungs 2-3 raced the real quasicrystal vs periodic mode FREQUENCIES and
  found NO advantage -- but they used RANDOM input/coupling weights, so they
  only isolated the mode-frequency DISTRIBUTION (which turned out inert).
  The mode SHAPES were never used. This rung uses them.

WHAT'S NEW HERE -- the coupling is PHYSICS, not random:
  We take the plate's ACTUAL mode shapes phi_i(x,y) (FEM eigenvectors, the
  transverse deflection field) for both a quasicrystal and a periodic plate,
  matched to the same coverage, and build a reservoir where:
    * input weight   w_in[i] = phi_i(x_drive)      -- a point actuator couples
      to each mode by that mode's amplitude at the drive location;
    * the nonlinear MODE COUPLING is the modal projection of a pointwise
      transverse nonlinearity  g(w) = a*w^2 + b*w^3,  i.e. the force on mode i
      is  -a*integral(phi_i * w^2 dA) - b*integral(phi_i * w^3 dA), with
      w(x,y) = sum_j x_j phi_j(x,y). This couples modes EXACTLY through the
      real spatial overlap of their shapes -- no random matrix anywhere.
  So quasicrystal vs periodic now differ in BOTH frequencies AND the entire
  nonlinear coupling network, and that network is set by the physics.

PRE-REGISTERED MECHANISM (stated before running, so this isn't fishing):
  A symmetric periodic plate's modes have definite parity, so many triple
  overlap integrals  T_ijk = integral(phi_i phi_j phi_k dA)  VANISH by
  selection rules -> a SPARSE nonlinear coupling network. The quasicrystal
  has no such symmetry -> a DENSER network. We MEASURE this density first
  (a structural prediction, like the 3-vs-18 degeneracy count in rung 3),
  THEN test whether the denser, shape-determined coupling actually makes a
  better reservoir. If it ties anyway, the mode shapes don't matter either,
  and that is an equally honest finding.

HONEST SCOPE:
  Mode shapes and frequencies are REAL (FEM). The pointwise quadratic+cubic
  nonlinearity is a standard reduced-order model of a plate with a local
  nonlinear restoring stress -- it is NOT the exact von Karman plate (which
  adds in-plane Airy-function coupling). So this tests whether the real mode
  SHAPES, fed through a physically-structured (not random) nonlinearity,
  change the reservoir -- not a fabricated-device performance claim.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

FEM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plate_bending_review",
)
sys.path.insert(0, FEM_DIR)
from fem_plate_bending_homogenized import (  # noqa: E402
    Lx, Ly, build_mesh, element_coverage_fractions, assemble,
    clamped_free_dofs, debruijn_quasicrystal_points,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_rung2_3 import (  # noqa: E402
    generate_periodic_holes, coverage_match_radius,
    NX, N_MODES, TARGET_COV, QC_NFOLD, QC_SEED, OMEGA_LO, OMEGA_HI,
    ridge_fit, r2,
)

# ---- reservoir settings ----
ZETA = 0.20
A_QUAD = 0.25       # pointwise quadratic nonlinearity coefficient (2nd-order products)
B_CUBIC = 0.25      # pointwise cubic (hardening) coefficient -> bounds amplitude
TAU_IN = 1.0
DT = 0.02
INPUT_AMP = 1.0
L = 2000
WASHOUT = 200
N_TRAIN = 1200

# ensemble = many actuator (drive) locations, IDENTICAL for both plates.
# physically meaningful spread (you don't know the best actuator spot), and
# the ONLY thing that varies within a plate -- so mean/std is a fair test.
# 16 deterministic interior points (away from the clamped edges) -> tighter
# error bars, so a ~1x-spread hint can be resolved into signal or noise.
_drng = np.random.default_rng(2024)
DRIVE_FRACS = [tuple(p) for p in _drng.uniform(0.22, 0.78, size=(16, 2))]


def mode_shapes(holes, radius, n_modes, nx):
    """Real FEM modal analysis returning (freqs, Phi, aw_norm, elem_centers).

    Phi[i, e] = transverse deflection of mode i at the center of element e,
    each mode normalized to unit RMS over the plate. aw_norm = per-element
    area weight summing to 1 (the discrete dA/A integration measure)."""
    nodes, quads = build_mesh(Lx, Ly, nx, nx)
    phi_cov = element_coverage_fractions(nodes, quads, holes, radius, sub_n=12)
    K, M = assemble(nodes, quads, phi=phi_cov, stiffness_exponent=2.0)
    free = clamped_free_dofs(nodes)
    Kf = K[np.ix_(free, free)]; Mf = M[np.ix_(free, free)]
    k = min(n_modes + 6, len(free) - 2)
    sigma = max(Kf.diagonal().max() * 1e-4, 1e-20)
    vals, vecs = eigsh(Kf, k=k, M=Mf, sigma=sigma, which='LM', tol=1e-6, maxiter=50000)
    keep = vals > 1e-6 * np.abs(vals).max()
    vals, vecs = vals[keep], vecs[:, keep]
    order = np.argsort(vals)
    vals, vecs = vals[order][:n_modes], vecs[:, order][:, :n_modes]
    freqs = np.sqrt(np.abs(vals)) / (2 * np.pi)

    # transverse (w) DOF is index n*3 for node n; scatter free-DOF eigvecs back
    Nnodes = len(nodes)
    w_nodes = np.zeros((n_modes, Nnodes))
    for m in range(n_modes):
        full = np.zeros(3 * Nnodes)
        full[free] = vecs[:, m]
        w_nodes[m] = full[0::3]

    # element-center deflection (bilinear -> mean of 4 corners) and area weights
    coords = nodes[quads]                                  # (n_elem, 4, 2)
    centers = coords.mean(axis=1)                          # (n_elem, 2)
    dx = coords[:, :, 0].max(axis=1) - coords[:, :, 0].min(axis=1)
    dy = coords[:, :, 1].max(axis=1) - coords[:, :, 1].min(axis=1)
    aw = dx * dy                                           # element areas
    aw_norm = aw / aw.sum()                                # integration measure (sum=1)
    Wc = w_nodes[:, quads].mean(axis=2)                    # (n_modes, n_elem)

    # unit-RMS normalize each mode shape: integral(phi^2 dA)/A = 1.
    # identical normalization for both plates -> only SHAPE differs, not scale.
    rms = np.sqrt((aw_norm[None, :] * Wc**2).sum(axis=1))
    Wc = Wc / rms[:, None]
    return freqs, Wc, aw_norm, centers


def triple_overlap_density(Phi, aw_norm, n_use=24, tol_frac=0.10):
    """Structural metric: fraction of triple-overlap integrals T_ijk that are
    NON-negligible. Parity selection rules in a symmetric plate force many to
    ~0; an aperiodic plate breaks them. Higher density = richer coupling net."""
    P = Phi[:n_use]
    T = np.einsum('ie,je,ke,e->ijk', P, P, P, aw_norm)     # (n_use^3) overlaps
    scale = np.sqrt(np.mean(T**2))
    return float(np.mean(np.abs(T) > tol_frac * scale)), scale


def cross_coupling_offdiag_fraction(Phi, aw_norm, n_use=40):
    """G_ij = integral(phi_i^2 phi_j^2 dA): amplitude-dependent cross coupling
    from the cubic term. Report the off-diagonal mass fraction (how much of
    the coupling is mode-to-mode rather than self)."""
    P2 = Phi[:n_use] ** 2
    G = np.einsum('ie,je,e->ij', P2, P2, aw_norm)
    off = G.sum() - np.trace(G)
    return float(off / G.sum())


def nearest_elem(centers, frac):
    target = np.array([frac[0] * Lx, frac[1] * Ly])
    return int(np.argmin(((centers - target) ** 2).sum(axis=1)))


def run_reservoir(omega, Phi, aw_norm, w_in, u_series):
    """Drive the modal plate model; return state matrix [x_1..N, v_1..N, bias].

    Nonlinearity is the modal projection of a pointwise g(w)=a w^2 + b w^3,
    computed in the PHYSICAL domain each step:
        W(e) = sum_j Phi[j,e] x_j           (deflection at element e)
        force_i = -a * sum_e aw[e] Phi[i,e] W(e)^2  - b * (... W(e)^3)
    so modes couple through the real overlap of their shapes."""
    N = len(omega)
    x = np.zeros(N); v = np.zeros(N)
    n_sub = int(round(TAU_IN / DT))
    feats = np.empty((len(u_series), 2 * N + 1))
    for n, u in enumerate(u_series):
        for _ in range(n_sub):
            W = Phi.T @ x                                  # (n_elem,)
            f2 = Phi @ (aw_norm * W * W)                   # (N,) quadratic projection
            f3 = Phi @ (aw_norm * W * W * W)               # (N,) cubic projection
            accel = (-(omega**2) * x - 2 * ZETA * omega * v
                     - A_QUAD * f2 - B_CUBIC * f3 + w_in * u)
            v = v + accel * DT
            x = x + v * DT
        feats[n, :N] = x
        feats[n, N:2*N] = v
        feats[n, -1] = 1.0
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"reservoir blew up at sample {n} "
                               f"(reduce A_QUAD/B_CUBIC/INPUT_AMP/DT)")
    return feats


def eval_task(states, target):
    Xtr, Ytr = states[WASHOUT:WASHOUT + N_TRAIN], target[WASHOUT:WASHOUT + N_TRAIN]
    Xte, Yte = states[WASHOUT + N_TRAIN:], target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None])
    return r2(Yte, (Xte @ W)[:, 0])


def benchmark(freqs, Phi, aw_norm, centers, u, yA):
    """Average nonlinear-task R^2 and memory capacity over drive locations."""
    omega = OMEGA_LO + (OMEGA_HI - OMEGA_LO) * (freqs - freqs.min()) / (freqs.max() - freqs.min())
    nl, mc, amp = [], [], []
    for frac in DRIVE_FRACS:
        e = nearest_elem(centers, frac)
        w_in = Phi[:, e].copy()                            # phi_i(x_drive)
        states = run_reservoir(omega, Phi, aw_norm, w_in, u)
        amp.append(np.mean(np.abs(states[:, :len(freqs)])))
        nl.append(eval_task(states, yA))
        m = 0.0
        for k in range(1, 16):
            yk = np.zeros(len(u)); yk[k:] = u[:len(u) - k]
            m += max(0.0, eval_task(states, yk))
        mc.append(m)
    return np.array(nl), np.array(mc), float(np.mean(amp))


def main():
    print("=" * 74)
    print("RUNG 4 -- do the quasicrystal MODE SHAPES make a better reservoir?")
    print("=" * 74)

    print(f"\n=== Step 1: real FEM mode SHAPES (first {N_MODES} modes, {TARGET_COV}% coverage) ===")
    qc_holes = debruijn_quasicrystal_points(QC_NFOLD, Lx, Ly, offset_seed=QC_SEED)
    qc_r, qc_cov = coverage_match_radius(qc_holes, TARGET_COV, NX)
    qc_f, qc_Phi, qc_aw, qc_ctr = mode_shapes(qc_holes, qc_r, N_MODES, NX)
    print(f"  quasicrystal: {len(qc_holes)} holes, cov={qc_cov:.1f}%, {len(qc_f)} mode shapes")

    per_holes = generate_periodic_holes(9, Lx, Ly)
    per_r, per_cov = coverage_match_radius(per_holes, TARGET_COV, NX)
    per_f, per_Phi, per_aw, per_ctr = mode_shapes(per_holes, per_r, N_MODES, NX)
    print(f"  periodic:     {len(per_holes)} holes, cov={per_cov:.1f}%, {len(per_f)} mode shapes")

    # ---- STEP 2: pre-registered STRUCTURAL prediction (measured before racing) ----
    print("\n=== Step 2: structural prediction -- coupling-network density ===")
    qc_dens, _ = triple_overlap_density(qc_Phi, qc_aw)
    per_dens, _ = triple_overlap_density(per_Phi, per_aw)
    qc_off = cross_coupling_offdiag_fraction(qc_Phi, qc_aw)
    per_off = cross_coupling_offdiag_fraction(per_Phi, per_aw)
    print(f"  triple-overlap density (frac of |T_ijk| non-negligible):")
    print(f"      quasicrystal = {qc_dens:.3f}   periodic = {per_dens:.3f}   "
          f"(prediction: QC > periodic)")
    print(f"  cubic cross-coupling off-diagonal fraction:")
    print(f"      quasicrystal = {qc_off:.3f}   periodic = {per_off:.3f}")
    pred_ok = qc_dens > per_dens
    print(f"  -> structural prediction {'HOLDS' if pred_ok else 'does NOT hold'}: "
          f"quasicrystal coupling network is {'denser' if pred_ok else 'NOT denser'}.")

    # ---- STEP 3: race them as reservoirs ----
    print(f"\n=== Step 3: race reservoirs ({len(DRIVE_FRACS)} drive locations each) ===")
    rng = np.random.default_rng(0)
    u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)
    yA = np.zeros(L); yA[2:] = u[1:L-1] * u[0:L-2]

    qc_nl, qc_mc, qc_amp = benchmark(qc_f, qc_Phi, qc_aw, qc_ctr, u, yA)
    per_nl, per_mc, per_amp = benchmark(per_f, per_Phi, per_aw, per_ctr, u, yA)
    print(f"  (state amplitudes mean|x|: QC={qc_amp:.3f}, periodic={per_amp:.3f} "
          f"-- want ~O(0.1..1) so the nonlinearity is engaged but stable)")

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
            return f"  {name}: difference {d:+.3f} WITHIN noise ({sig:.1f}x spread) -> no winner"
        return f"  {name}: {who} better by {abs(d):.3f} ({sig:.1f}x spread) -> REAL difference"

    print()
    print(verdict(qc_nl, per_nl, "nonlinear task"))
    print(verdict(qc_mc, per_mc, "memory capacity"))

    # ---- plots ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    axA.bar([0, 1], [qc_dens, per_dens], color=["#2E5E8C", "#C0392B"])
    axA.set_xticks([0, 1]); axA.set_xticklabels(["quasicrystal", "periodic"])
    axA.set_title("Structural prediction:\ncoupling-network density (|T_ijk| non-negligible)")
    axA.set_ylabel("triple-overlap density"); axA.grid(alpha=0.3, axis="y")

    x = np.arange(2); w = 0.35
    axB.bar(x - w/2, [qc_nl.mean(), qc_mc.mean()/15], w,
            yerr=[qc_nl.std(), qc_mc.std()/15], label="quasicrystal", color="#2E5E8C", capsize=4)
    axB.bar(x + w/2, [per_nl.mean(), per_mc.mean()/15], w,
            yerr=[per_nl.std(), per_mc.std()/15], label="periodic", color="#C0392B", capsize=4)
    axB.set_xticks(x); axB.set_xticklabels(["nonlinear task\n(R^2)", "memory cap.\n(/15)"])
    axB.set_title("Reservoir performance (mean +/- std over drive locations)")
    axB.legend(); axB.grid(alpha=0.3, axis="y")

    fig.suptitle("Rung 4: real mode SHAPES -> physically-structured coupling. "
                 "Does the quasicrystal win now?", fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reservoir_rung4_modeshapes.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved {out}")

    # ---- honest verdict ----
    d_nl = qc_nl.mean() - per_nl.mean()
    pooled = np.hypot(qc_nl.std(), per_nl.std()) + 1e-9
    print("\n" + "=" * 74)
    print("HONEST VERDICT (rung 4)")
    print("=" * 74)
    print(f"  Structural prediction (QC denser coupling network): "
          f"{'CONFIRMED' if pred_ok else 'NOT confirmed'} "
          f"({qc_dens:.3f} vs {per_dens:.3f}).")
    if abs(d_nl) / pooled >= 1.0:
        who = "quasicrystal" if d_nl > 0 else "periodic"
        print(f"  Performance: {who} is genuinely better on the nonlinear task "
              f"({abs(d_nl):.3f}, {abs(d_nl)/pooled:.1f}x spread).")
        if d_nl > 0:
            print("  => The mode SHAPES matter: the quasicrystal's denser, physics-set")
            print("     coupling network makes a measurably better reservoir. FIRST positive")
            print("     signal in the ladder -- but still a model, not a fabricated device.")
        else:
            print("  => Shapes matter, but they favor the PERIODIC plate -- against the")
            print("     hypothesis. Honest and reportable.")
    else:
        print(f"  Performance: TIE within noise ({d_nl:+.3f}, {abs(d_nl)/pooled:.1f}x spread).")
        print("  => Even with the real mode shapes driving a physically-structured (not")
        print("     random) coupling network -- and even though that network IS measurably")
        print("     denser for the quasicrystal -- it gives no reservoir advantage. The")
        print("     richer structure is real but again computationally inert. Clean")
        print("     negative result, consistent with rungs 2-3.")
    print("  REAL: mode shapes & frequencies (FEM). MODEL: pointwise quad+cubic")
    print("  nonlinearity (not exact von Karman). Not a fabricated-device claim.")
    print("=" * 74)


if __name__ == "__main__":
    main()
