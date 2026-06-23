"""
Full rerun of the Section 3.4 centerpiece table (coverage-matched symmetry
comparison, 98/90/80% coverage, quadratic exponent, 5 seeds) with the
corrected de Bruijn multigrid geometry (fem_plate_bending_homogenized.py).
"""
import numpy as np
from fem_plate_bending_homogenized import (
    Lx, Ly, build_mesh, element_coverage_fractions, assemble,
    clamped_free_dofs, solve_modes, debruijn_quasicrystal_points
)

NX = 28
n_folds = [3, 6, 8, 12]
coverages = [98, 90, 80]
seeds = [42, 7, 123, 314, 271]


def find_radius_for_coverage(n_fold, target_cov, nx, seed=42, sub_n=12,
                              r_lo=0.3e-6, r_hi=15.0e-6, tol=0.4, max_iter=40):
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


print("=== Coverage-matching bisection (corrected geometry, nx=28, seed=42 for radius derivation) ===")
matched = {}
for cov_target in coverages:
    for nf in n_folds:
        r, cov = find_radius_for_coverage(nf, cov_target, NX, seed=42)
        matched[(cov_target, nf)] = r
        print(f"  target={cov_target}%  n_fold={nf:2d}  r={r*1e6:.4f} um  achieved={cov:.2f}%")

print("\n=== Full symmetry comparison, 5 seeds, quadratic exponent ===")
all_results = {}
for cov_target in coverages:
    print(f"\n--- coverage target {cov_target}% ---")
    per_nf = {}
    for nf in n_folds:
        r = matched[(cov_target, nf)]
        f1s = []
        for s in seeds:
            holes = debruijn_quasicrystal_points(nf, Lx, Ly, offset_seed=s)
            nodes, quads = build_mesh(Lx, Ly, NX, NX)
            phi = element_coverage_fractions(nodes, quads, holes, r, sub_n=12)
            cov = phi.mean() * 100
            K, M = assemble(nodes, quads, phi=phi, stiffness_exponent=2.0)
            free = clamped_free_dofs(nodes)
            freqs = solve_modes(K, M, free)
            f1s.append(freqs[0])
            print(f"  n_fold={nf:2d}  seed={s:4d}  cov={cov:.2f}%  f1={freqs[0]/1e6:.5f} MHz")
        per_nf[nf] = np.array(f1s)
    all_results[cov_target] = per_nf

print("\n=== Summary ===")
for cov_target in coverages:
    means = {nf: all_results[cov_target][nf].mean() for nf in n_folds}
    stds = {nf: all_results[cov_target][nf].std() for nf in n_folds}
    vals = np.array(list(means.values()))
    spread = (vals.max() - vals.min()) / vals.mean() * 100
    print(f"\ncoverage={cov_target}%:")
    for nf in n_folds:
        print(f"  n_fold={nf:2d}  mean={means[nf]/1e6:.5f} MHz  std={stds[nf]/1e6*100/(means[nf]/1e6):.3f}% (of mean)")
    print(f"  spread across n_fold: {spread:.2f}%")
