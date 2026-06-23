"""
Sensitivity check for the phi floor (Section 2.2): phi = max(phi, PHI_FLOOR)
is applied as standard SIMP-style regularization to avoid a singular mass
matrix for fully-covered elements. PHI_FLOOR=1e-3 was asserted as "standard"
without testing whether the reported results actually depend on it -- this
tests that directly, at the representative n=8, 80%-coverage configuration
used throughout Section 3, across PHI_FLOOR spanning three orders of
magnitude (1e-2 down to 1e-5), holding the hole geometry (radius) fixed.
"""
import numpy as np
from fem_plate_bending_homogenized import (
    Lx, Ly, build_mesh, debruijn_quasicrystal_points,
    quad_mindlin_sri_element, D, nu, Gs, rho, h
)
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

NX = 28
N_FOLD = 8
SEED = 42


def element_coverage_fractions_floor(nodes, quads, holes, radii, sub_n=12, phi_floor=1e-3):
    if not hasattr(radii, "__len__"):
        radii = np.full(len(holes), radii)
    coords = nodes[quads]
    xmin = coords[:, :, 0].min(axis=1); xmax = coords[:, :, 0].max(axis=1)
    ymin = coords[:, :, 1].min(axis=1); ymax = coords[:, :, 1].max(axis=1)
    su, sv = np.meshgrid((np.arange(sub_n) + 0.5) / sub_n, (np.arange(sub_n) + 0.5) / sub_n)
    su = su.ravel(); sv = sv.ravel()
    n_quads = len(quads)
    phi = np.ones(n_quads)
    for qi in range(n_quads):
        px = xmin[qi] + su * (xmax[qi] - xmin[qi])
        py = ymin[qi] + sv * (ymax[qi] - ymin[qi])
        covered = np.zeros(len(px), dtype=bool)
        for (hx, hy), r in zip(holes, radii):
            covered |= (px - hx) ** 2 + (py - hy) ** 2 < r ** 2
        phi[qi] = 1.0 - covered.mean()
    return np.maximum(phi, phi_floor)


def assemble_floor(nodes, quads, phi, stiffness_exponent=2.0):
    N = len(nodes); nd = 3 * N
    K = lil_matrix((nd, nd)); M = lil_matrix((nd, nd))
    for qi, q in enumerate(quads):
        coords = nodes[q]
        Ke, Me = quad_mindlin_sri_element(coords, D, nu, Gs, rho, h, phi=phi[qi],
                                           stiffness_exponent=stiffness_exponent)
        dofs = []
        for n in q: dofs += [n * 3, n * 3 + 1, n * 3 + 2]
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K[di, dj] += Ke[i, j]
                M[di, dj] += Me[i, j]
    return csr_matrix(K), csr_matrix(M)


def clamped_free_dofs(nodes):
    nd = 3 * len(nodes); tol = 1e-10
    xmn, xmx = nodes[:, 0].min(), nodes[:, 0].max()
    ymn, ymx = nodes[:, 1].min(), nodes[:, 1].max()
    bdry = ((nodes[:, 0] <= xmn+tol) | (nodes[:, 0] >= xmx-tol) |
            (nodes[:, 1] <= ymn+tol) | (nodes[:, 1] >= ymx-tol))
    con = []
    for i in np.where(bdry)[0]: con += [i*3, i*3+1, i*3+2]
    return np.setdiff1d(np.arange(nd), con)


def solve_modes(K, M, free, n_modes=6):
    Kf = K[np.ix_(free, free)]; Mf = M[np.ix_(free, free)]
    k = min(n_modes, len(free) - 2)
    sigma = max(Kf.diagonal().max() * 1e-4, 1e-20)
    vals, vecs = eigsh(Kf, k=k, M=Mf, sigma=sigma, which='LM', tol=1e-6, maxiter=50000)
    pos = vals > 1e-6 * np.abs(vals).max()
    vals = vals[pos]
    return np.sort(np.sqrt(np.abs(vals)) / (2 * np.pi))


# Use the same n=8, 80%-coverage radius derived earlier in this session
holes = debruijn_quasicrystal_points(N_FOLD, Lx, Ly, offset_seed=SEED)
nodes, quads = build_mesh(Lx, Ly, NX, NX)

# find the 80%-matched radius once (bisection), then hold geometry fixed
def cov_at(r, floor):
    phi = element_coverage_fractions_floor(nodes, quads, holes, r, phi_floor=floor)
    return phi.mean() * 100

r_lo, r_hi = 0.3e-6, 10.0e-6
target = 80.0
for _ in range(40):
    mid = 0.5 * (r_lo + r_hi)
    c = cov_at(mid, 1e-3)
    if abs(c - target) <= 0.4:
        break
    if c > target:
        r_lo = mid
    else:
        r_hi = mid
radius = mid
print(f"Fixed geometry: n_fold={N_FOLD}, seed={SEED}, radius={radius*1e6:.4f} um, coverage(floor=1e-3)={cov_at(radius,1e-3):.2f}%")

print("\n=== phi-floor sensitivity, fixed geometry, quadratic exponent ===")
floors = [1e-2, 1e-3, 1e-4, 1e-5]
results = {}
for floor in floors:
    phi = element_coverage_fractions_floor(nodes, quads, holes, radius, phi_floor=floor)
    cov = phi.mean() * 100
    n_at_floor = int((phi <= floor * 1.0001).sum())
    K, M = assemble_floor(nodes, quads, phi, stiffness_exponent=2.0)
    free = clamped_free_dofs(nodes)
    freqs = solve_modes(K, M, free)
    results[floor] = freqs[0]
    print(f"  phi_floor={floor:.0e}  coverage={cov:.3f}%  elements_at_floor={n_at_floor}  f1={freqs[0]/1e6:.6f} MHz")

ref = results[1e-3]
print(f"\n=== Relative to the paper's chosen floor (1e-3) ===")
for floor in floors:
    pct = (results[floor] - ref) / ref * 100
    print(f"  phi_floor={floor:.0e}  f1 deviation from 1e-3 baseline: {pct:+.4f}%")
