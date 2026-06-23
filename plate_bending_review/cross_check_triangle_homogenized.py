"""
Cross-validation of the corrected (homogenized, quadratic-exponent) FEM
results using a SECOND, independently-coded element: a 3-node Mindlin-
Reissner triangle (linear shape functions, physical/unfitted shear
stiffness), on a Delaunay-triangulated mesh -- different topology, different
shape functions, separately-written stiffness/mass assembly from the
4-node quad+SRI element used for the paper's main results.

This supersedes cross_check_triangle_element.py, which validated the
triangle element against the now-superseded BINARY-REMOVAL hole
representation. That comparison is no longer meaningful: binary removal is
known to produce non-converging, disconnected meshes (see
fem_plate_bending_homogenized.py docstring), so agreement or disagreement
with it doesn't say anything about the corrected methodology.

This script applies the SAME area-fraction homogenization and quadratic
stiffness-penalization exponent (Ke = phi^2 * (Kb+Ks), Me = phi*Me) used in
the corrected quad-element analysis, just re-implemented independently for
triangular elements via sub-sampling. The geometry (hole radii) is
re-derived via the same coverage-matching bisection used for the paper's
main results, so this script does not depend on copy-pasting numbers from
the quad-element run.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay

E, nu, rho, h = 170e9, 0.28, 2330.0, 100e-9
G = E / (2 * (1 + nu))
kap = 5.0 / 6.0
D = E * h ** 3 / (12 * (1 - nu ** 2))
Gs = kap * G * h

Lx = Ly = 100e-6


def build_tri_mesh(Lx, Ly, n_grid):
    xs = np.linspace(0, Lx, n_grid)
    ys = np.linspace(0, Ly, n_grid)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    tri = Delaunay(nodes)
    return nodes, tri.simplices


def element_coverage_fractions_tri(nodes, tris, holes, radii, sub_n=12):
    """Area fraction NOT covered by any hole, per triangle, via barycentric
    sub-sampling (same sub_n density as the quad element's sub-grid)."""
    if not hasattr(radii, "__len__"):
        radii = np.full(len(holes), radii)

    p1 = nodes[tris[:, 0]]; p2 = nodes[tris[:, 1]]; p3 = nodes[tris[:, 2]]

    # barycentric sample points covering the triangle interior
    samples = []
    for i in range(sub_n):
        for j in range(sub_n - i):
            u = (i + 1.0 / 3.0) / sub_n
            v = (j + 1.0 / 3.0) / sub_n
            if u + v < 1.0:
                samples.append((u, v))
    samples = np.array(samples)
    su = samples[:, 0]; sv = samples[:, 1]; sw = 1.0 - su - sv

    n_tri = len(tris)
    phi = np.ones(n_tri)
    for ti in range(n_tri):
        px = sw * p1[ti, 0] + su * p2[ti, 0] + sv * p3[ti, 0]
        py = sw * p1[ti, 1] + su * p2[ti, 1] + sv * p3[ti, 1]
        covered = np.zeros(len(px), dtype=bool)
        for (hx, hy), r in zip(holes, radii):
            covered |= (px - hx) ** 2 + (py - hy) ** 2 < r ** 2
        phi[ti] = 1.0 - covered.mean()
    phi = np.maximum(phi, 1e-3)
    return phi


def mindlin_triangle_element(p1, p2, p3, D_b, nu_m, Gs_m, rho_m, h_m, phi=1.0, stiffness_exponent=1.0):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = abs(A2) / 2.0
    if A < 1e-24:
        return np.zeros((9, 9)), np.zeros((9, 9))

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    dNdx = np.array([b1, b2, b3]) / A2
    dNdy = np.array([c1, c2, c3]) / A2

    Db = D_b * np.array([[1, nu_m, 0], [nu_m, 1, 0], [0, 0, (1 - nu_m) / 2]])
    Bb = np.zeros((3, 9))
    for i in range(3):
        Bb[0, i * 3 + 1] = dNdx[i]
        Bb[1, i * 3 + 2] = dNdy[i]
        Bb[2, i * 3 + 1] = dNdy[i]
        Bb[2, i * 3 + 2] = dNdx[i]
    Kb = A * (Bb.T @ Db @ Bb)

    Ds = Gs_m * np.eye(2)
    Ni_c = 1.0 / 3.0
    Bs = np.zeros((2, 9))
    for i in range(3):
        Bs[0, i * 3] = dNdx[i]
        Bs[0, i * 3 + 2] = -Ni_c
        Bs[1, i * 3] = dNdy[i]
        Bs[1, i * 3 + 1] = Ni_c
    Ks = A * (Bs.T @ Ds @ Bs)

    Me = np.zeros((9, 9))
    mt = rho_m * h_m * A
    mr = rho_m * h_m ** 3 / 12.0 * A
    for i in range(3):
        for j in range(3):
            fac = (2.0 if i == j else 1.0) / 12.0
            Me[i * 3, j * 3] = mt * fac
            Me[i * 3 + 1, j * 3 + 1] = mr * fac
            Me[i * 3 + 2, j * 3 + 2] = mr * fac

    Ke = (phi ** stiffness_exponent) * (Kb + Ks)
    Me = phi * Me
    return Ke, Me


def assemble(nodes, tris, phi=None, stiffness_exponent=1.0):
    N = len(nodes); nd = 3 * N
    K = lil_matrix((nd, nd)); M = lil_matrix((nd, nd))
    if phi is None:
        phi = np.ones(len(tris))
    for ti, tri in enumerate(tris):
        n0, n1, n2 = tri
        Ke, Me = mindlin_triangle_element(nodes[n0], nodes[n1], nodes[n2], D, nu, Gs, rho, h,
                                           phi=phi[ti], stiffness_exponent=stiffness_exponent)
        dofs = [n0*3, n0*3+1, n0*3+2, n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
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
    freqs = np.sqrt(np.abs(vals)) / (2 * np.pi)
    return np.sort(freqs)


def debruijn_quasicrystal_points(n_fold, domain_x, domain_y, offset_seed=42):
    rng = np.random.default_rng(offset_seed)
    gammas = rng.uniform(0, 1, n_fold)
    dirs = [np.array([np.cos(k*np.pi/n_fold), np.sin(k*np.pi/n_fold)]) for k in range(n_fold)]
    scale = min(domain_x, domain_y) * 0.45
    pts = []
    for i in range(n_fold):
        for j in range(i+1, n_fold):
            d0, d1 = dirs[i], dirs[j]
            cr = d0[0]*d1[1] - d0[1]*d1[0]
            if abs(cr) < 1e-10: continue
            for ni in range(-8, 9):
                for nj in range(-8, 9):
                    r = (ni + gammas[i] - gammas[j]) / cr
                    px = (ni + gammas[i])*d0[0] + r*d1[0]
                    py = (ni + gammas[i])*d0[1] + r*d1[1]
                    px_m = px*scale + domain_x/2
                    py_m = py*scale + domain_y/2
                    if 0 < px_m < domain_x and 0 < py_m < domain_y:
                        pts.append([px_m, py_m])
    if not pts: return np.empty((0, 2))
    pts = np.array(pts)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]: continue
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
        keep[i+1:][d < scale*0.05] = False
    return pts[keep]


def run_case(n_fold, hole_radius, n_grid, seed=42, sub_n=12, stiffness_exponent=2.0):
    holes = debruijn_quasicrystal_points(n_fold, Lx, Ly, offset_seed=seed)
    nodes, tris = build_tri_mesh(Lx, Ly, n_grid)
    phi = element_coverage_fractions_tri(nodes, tris, holes, hole_radius, sub_n=sub_n)
    cov = phi.mean() * 100
    K, M = assemble(nodes, tris, phi=phi, stiffness_exponent=stiffness_exponent)
    free = clamped_free_dofs(nodes)
    freqs = solve_modes(K, M, free)
    return freqs, cov


# --- coverage-matching bisection, mirroring the quad element's protocol ---
def find_radius_for_coverage(n_fold, target_cov, n_grid, seed=42, sub_n=12,
                              r_lo=0.3e-6, r_hi=4.0e-6, tol=0.4, max_iter=40):
    holes = debruijn_quasicrystal_points(n_fold, Lx, Ly, offset_seed=seed)
    nodes, tris = build_tri_mesh(Lx, Ly, n_grid)

    def cov_at(r):
        phi = element_coverage_fractions_tri(nodes, tris, holes, r, sub_n=sub_n)
        return phi.mean() * 100

    lo, hi = r_lo, r_hi
    cov_lo, cov_hi = cov_at(lo), cov_at(hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cov_mid = cov_at(mid)
        if abs(cov_mid - target_cov) <= tol:
            return mid, cov_mid
        # coverage decreases as radius increases
        if cov_mid > target_cov:
            lo = mid
        else:
            hi = mid
    return mid, cov_mid


if __name__ == "__main__":
    print("=== Step 1: Leissa benchmark for the HOMOGENIZED triangle element (phi=1 everywhere) ===")
    f_anal = 35.99 / (2*np.pi*Lx**2) * np.sqrt(D/(rho*h))
    print(f"Analytical f1: {f_anal/1e6:.4f} MHz")
    for ng in [16, 22, 30, 40]:
        nodes, tris = build_tri_mesh(Lx, Ly, ng)
        K, M = assemble(nodes, tris)
        free = clamped_free_dofs(nodes)
        freqs = solve_modes(K, M, free)
        print(f"  n_grid={ng:3d}  nodes={len(nodes):5d}  f1={freqs[0]/1e6:.5f} MHz  ratio={freqs[0]/f_anal:.4f}")

    NX = 26  # closest triangle-mesh density to the quad element's nx=28 working resolution
    TARGET_COV = 80.0
    n_folds = [3, 6, 8, 12]

    print(f"\n=== Step 2: coverage-matching bisection (target {TARGET_COV}%, n_grid={NX}, triangle element) ===")
    matched = {}
    for nf in n_folds:
        r, cov = find_radius_for_coverage(nf, TARGET_COV, NX, seed=42)
        matched[nf] = r
        print(f"  n_fold={nf:2d}  r={r*1e6:.4f} um  achieved coverage={cov:.2f}%")

    print(f"\n=== Step 3: symmetry comparison at matched {TARGET_COV}% coverage, quadratic exponent, 3 seeds ===")
    seeds = [42, 7, 123]
    per_nf_means = {}
    for nf in n_folds:
        r = matched[nf]
        f1s = []
        for s in seeds:
            freqs, cov = run_case(nf, r, NX, seed=s, stiffness_exponent=2.0)
            f1s.append(freqs[0])
            print(f"  n_fold={nf:2d}  seed={s:4d}  cov={cov:.2f}%  f1={freqs[0]/1e6:.5f} MHz")
        per_nf_means[nf] = np.mean(f1s)

    vals = np.array(list(per_nf_means.values()))
    spread = (vals.max() - vals.min()) / vals.mean() * 100
    print(f"\nPer-n_fold means (MHz): " + ", ".join(f"n={nf}: {per_nf_means[nf]/1e6:.5f}" for nf in n_folds))
    print(f"Cross-validation (triangle element, homogenized, quadratic exponent) spread across n_fold: {spread:.2f}%")
