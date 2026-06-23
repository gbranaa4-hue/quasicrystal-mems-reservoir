"""
Re-run of the gap 0-1 bandgap-width analysis (paper Section 4.3) using the
quadratic stiffness exponent, which is the paper's main, better-justified
choice (Section 3.3) -- the original bandgap analysis used the linear
exponent because it predates the exponent resolution, an inconsistency
flagged by peer review as needing correction before the bandgap comparison
to photonic-quasicrystal literature can be trusted.

Protocol mirrors the original (linear-exponent) bandgap analysis exactly,
changing only stiffness_exponent: 1.0 -> 2.0.
  - Coverage-matched (80% target, +/-0.4%) radii via bisection, nx=28,
    quad+SRI homogenized element (same element as the paper's main results)
  - First 15 eigenfrequencies per configuration
  - gap 0-1 width = (f[1]-f[0]) / mean(f[0],f[1]) * 100%
  - 3 seeds per symmetry order (same seed count as the original analysis)
"""
import numpy as np
from fem_plate_bending_homogenized import (
    Lx, Ly, build_mesh, element_coverage_fractions, assemble,
    clamped_free_dofs, solve_modes, debruijn_quasicrystal_points
)

NX = 28
TARGET_COV = 80.0
n_folds = [3, 6, 8, 12]
seeds = [42, 7, 123]


def find_radius_for_coverage(n_fold, target_cov, nx, seed=42, sub_n=12,
                              r_lo=0.3e-6, r_hi=10.0e-6, tol=0.4, max_iter=40):
    holes = debruijn_quasicrystal_points(n_fold, Lx, Ly, offset_seed=seed)
    nodes, quads = build_mesh(Lx, Ly, nx, nx)

    def cov_at(r):
        phi = element_coverage_fractions(nodes, quads, holes, r, sub_n=sub_n)
        return phi.mean() * 100

    lo, hi = r_lo, r_hi
    mid, cov_mid = hi, cov_at(hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cov_mid = cov_at(mid)
        if abs(cov_mid - target_cov) <= tol:
            return mid, cov_mid
        if cov_mid > target_cov:
            lo = mid
        else:
            hi = mid
    return mid, cov_mid


print(f"=== Coverage-matching bisection (target {TARGET_COV}%, nx={NX}) ===")
matched = {}
for nf in n_folds:
    r, cov = find_radius_for_coverage(nf, TARGET_COV, NX, seed=42)
    matched[nf] = r
    print(f"  n_fold={nf:2d}  r={r*1e6:.4f} um  achieved coverage={cov:.2f}%")

print(f"\n=== Gap 0-1 width, QUADRATIC exponent, 15 modes, 3 seeds per symmetry order ===")
per_nf_gaps = {}
for nf in n_folds:
    r = matched[nf]
    gaps = []
    for s in seeds:
        holes = debruijn_quasicrystal_points(nf, Lx, Ly, offset_seed=s)
        nodes, quads = build_mesh(Lx, Ly, NX, NX)
        phi = element_coverage_fractions(nodes, quads, holes, r, sub_n=12)
        cov = phi.mean() * 100
        K, M = assemble(nodes, quads, phi=phi, stiffness_exponent=2.0)
        free = clamped_free_dofs(nodes)
        freqs = solve_modes(K, M, free, n_modes=15)
        if len(freqs) < 2:
            print(f"  n_fold={nf:2d}  seed={s:4d}  cov={cov:.2f}%  -- fewer than 2 modes found, skipped")
            continue
        f0, f1 = freqs[0], freqs[1]
        gap = (f1 - f0) / ((f0 + f1) / 2.0) * 100.0
        gaps.append(gap)
        print(f"  n_fold={nf:2d}  seed={s:4d}  cov={cov:.2f}%  f0={f0/1e6:.5f} MHz  f1={f1/1e6:.5f} MHz  gap0-1={gap:.2f}%")
    per_nf_gaps[nf] = gaps

print("\n=== Summary: gap 0-1 width by symmetry order (quadratic exponent) ===")
for nf in n_folds:
    g = np.array(per_nf_gaps[nf])
    print(f"  n_fold={nf:2d}  mean={g.mean():.2f}%  std={g.std():.2f}%  (n={len(g)} seeds)")

means = np.array([np.mean(per_nf_gaps[nf]) for nf in n_folds])
spread = (means.max() - means.min()) / means.mean() * 100
print(f"\nSpread across n_fold (quadratic exponent): {spread:.2f}%")
