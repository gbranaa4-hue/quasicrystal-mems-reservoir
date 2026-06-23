"""
Spot-check the stiffness-penalization exponent resolution (Section 3.3) with
the corrected de Bruijn geometry: does the quadratic exponent still better
match a fine-mesh, exponent-independent reference at the working resolution
(nx=28), for a representative n=8, ~80%-coverage configuration?
"""
import numpy as np
from fem_plate_bending_homogenized import (
    Lx, Ly, build_mesh, element_coverage_fractions, assemble,
    clamped_free_dofs, solve_modes, debruijn_quasicrystal_points
)

N_FOLD = 8
SEED = 42
TARGET_COV = 80.0


def find_radius_for_coverage(n_fold, target_cov, nx, seed, sub_n=12,
                              r_lo=0.2e-6, r_hi=15.0e-6, tol=0.4, max_iter=40):
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


r28, cov28 = find_radius_for_coverage(N_FOLD, TARGET_COV, 28, SEED)
print(f"radius matched at nx=28: r={r28*1e6:.4f}um, coverage={cov28:.2f}%")
holes = debruijn_quasicrystal_points(N_FOLD, Lx, Ly, offset_seed=SEED)

print("\n=== exponent sensitivity across mesh resolution (same hole radius held fixed) ===")
for nx in [28, 60, 100, 160]:
    nodes, quads = build_mesh(Lx, Ly, nx, nx)
    phi = element_coverage_fractions(nodes, quads, holes, r28, sub_n=12)
    f1_lin = None; f1_quad = None
    for exp, label in [(1.0, 'linear'), (2.0, 'quadratic')]:
        K, M = assemble(nodes, quads, phi=phi, stiffness_exponent=exp)
        free = clamped_free_dofs(nodes)
        freqs = solve_modes(K, M, free)
        if exp == 1.0: f1_lin = freqs[0]
        else: f1_quad = freqs[0]
    diff = abs(f1_lin - f1_quad) / ((f1_lin+f1_quad)/2) * 100
    print(f"  nx={nx:3d}  cov={phi.mean()*100:.2f}%  f1_linear={f1_lin/1e6:.5f} MHz  f1_quadratic={f1_quad/1e6:.5f} MHz  diff={diff:.2f}%")
