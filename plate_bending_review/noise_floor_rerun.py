"""
Mesh-convergence noise floor, corrected geometry: representative n=8
configuration (seed=42), at each target coverage, computed at nx=24,28,34,
holding hole geometry (radius derived at nx=28) fixed across mesh
resolutions -- matching the original Section 3.4 protocol exactly.
"""
import numpy as np
from fem_plate_bending_homogenized import (
    Lx, Ly, build_mesh, element_coverage_fractions, assemble,
    clamped_free_dofs, solve_modes, debruijn_quasicrystal_points
)

N_FOLD = 8
SEED = 42
coverages = [98, 90, 80]


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


for cov_target in coverages:
    r, cov28 = find_radius_for_coverage(N_FOLD, cov_target, 28, SEED)
    holes = debruijn_quasicrystal_points(N_FOLD, Lx, Ly, offset_seed=SEED)
    f1s = {}
    for nx in [24, 28, 34]:
        nodes, quads = build_mesh(Lx, Ly, nx, nx)
        phi = element_coverage_fractions(nodes, quads, holes, r, sub_n=12)
        K, M = assemble(nodes, quads, phi=phi, stiffness_exponent=2.0)
        free = clamped_free_dofs(nodes)
        freqs = solve_modes(K, M, free)
        f1s[nx] = freqs[0]
        print(f"  coverage_target={cov_target}%  nx={nx}  cov={phi.mean()*100:.2f}%  f1={freqs[0]/1e6:.5f} MHz")
    vals = np.array(list(f1s.values()))
    noise = (vals.max() - vals.min()) / vals.mean() * 100
    print(f"  --> mesh-convergence noise floor at {cov_target}% coverage: {noise:.2f}%\n")
