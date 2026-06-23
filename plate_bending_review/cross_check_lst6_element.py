"""
Real (locking-resistant) independent cross-validation element: a 6-node
quadratic Mindlin-Reissner triangle (LST topology: 3 corner + 3 mid-edge
nodes), re-derived independently from the paper's main 4-node quad element
-- different shape functions, different (Delaunay-based, with generated
mid-edge nodes) mesh topology, separately-written stiffness/mass assembly.

INTEGRATION SCHEME (and a real dead-end worth recording): the first attempt
copied the quad element's SRI *strategy* literally -- full 3-point Gauss for
bending/mass, single-centroid-point reduced integration for shear. That
produced a rank-deficient stiffness matrix (negative/near-zero eigenvalues
even with all boundary DOFs constrained) -- spurious zero-energy
("hourglass") modes. The bilinear quad's 1-point shear reduction works
because its shear field is naturally degree-1; this element's shear field
is degree-2 (same order as its bending field), so 1 point is genuinely
under-integrated, not merely "selectively reduced". Fixed by using the same
3-point Gauss rule for both bending and shear -- full/consistent
integration for this element, which converges cleanly (ratio 1.28x at
n_grid=10 down to 1.02x at n_grid=24 on the Leissa benchmark) without the
severe, non-converging locking seen in the 3-node linear triangle.

WHY NOT THE EARLIER 3-NODE LINEAR TRIANGLE (cross_check_triangle_element.py
/ cross_check_triangle_homogenized.py): direct testing showed it is locked
by 19.7x at n_grid=16 and still 2.5x at n_grid=120 (14,400 nodes) on the
plain Leissa benchmark -- it does not converge in any practical mesh-size
regime at this plate's h/Lx ~ 1e-3 thickness ratio, because its constant
(single-point) shear strain field has no relief mechanism at all. That
element is unsuitable for cross-validation and is not used for the paper's
cross-validation claim; this file replaces it.

Homogenization: identical philosophy to the quad element -- per-element
area-fraction phi (NOT covered by holes) via sub-sampling, with
Ke = phi^stiffness_exponent * (Kb+Ks), Me = phi*Me (mass always linear,
since mass is extensive).
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


# ─────────────────────────────────────────────────────────────────────────
# Mesh: linear Delaunay triangulation + generated mid-edge nodes -> 6-node
# (LST-topology) elements. Genuinely different mesh-generation path from
# both the quad element (structured quad grid) and the old linear triangle
# (same Delaunay corners, but no mid-edge nodes).
# ─────────────────────────────────────────────────────────────────────────
def build_lst6_mesh(Lx, Ly, n_grid):
    xs = np.linspace(0, Lx, n_grid)
    ys = np.linspace(0, Ly, n_grid)
    xx, yy = np.meshgrid(xs, ys)
    corner_nodes = np.column_stack([xx.ravel(), yy.ravel()])
    tri = Delaunay(corner_nodes)
    corners = tri.simplices  # (n_tri, 3) corner-node indices

    edge_mid_id = {}
    mid_coords = []
    n_corner = len(corner_nodes)

    def mid_index(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in edge_mid_id:
            edge_mid_id[key] = n_corner + len(mid_coords)
            mid_coords.append(0.5 * (corner_nodes[a] + corner_nodes[b]))
        return edge_mid_id[key]

    elems6 = np.zeros((len(corners), 6), dtype=int)
    for ei, (n1, n2, n3) in enumerate(corners):
        m12 = mid_index(n1, n2)
        m23 = mid_index(n2, n3)
        m31 = mid_index(n3, n1)
        elems6[ei] = [n1, n2, n3, m12, m23, m31]

    nodes = np.vstack([corner_nodes, np.array(mid_coords)]) if mid_coords else corner_nodes
    return nodes, elems6, corners  # corners kept for geometry/coverage sampling


# ─────────────────────────────────────────────────────────────────────────
# Area-fraction homogenization (same sub-sampling philosophy as the quad
# element), sampled on the geometric (corner-node) triangle.
# ─────────────────────────────────────────────────────────────────────────
def element_coverage_fractions_lst6(nodes, corners, holes, radii, sub_n=12):
    if not hasattr(radii, "__len__"):
        radii = np.full(len(holes), radii)

    p1 = nodes[corners[:, 0]]; p2 = nodes[corners[:, 1]]; p3 = nodes[corners[:, 2]]

    samples = []
    for i in range(sub_n):
        for j in range(sub_n - i):
            u = (i + 1.0 / 3.0) / sub_n
            v = (j + 1.0 / 3.0) / sub_n
            if u + v < 1.0:
                samples.append((u, v))
    samples = np.array(samples)
    su = samples[:, 0]; sv = samples[:, 1]; sw = 1.0 - su - sv

    n_tri = len(corners)
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


# ─────────────────────────────────────────────────────────────────────────
# 6-node quadratic Mindlin triangle, SRI: full 3-point Gauss (degree-2
# exact) for bending/mass, single centroid point for shear.
# ─────────────────────────────────────────────────────────────────────────
_GAUSS3 = [(2/3, 1/6, 1/6, 1/3), (1/6, 2/3, 1/6, 1/3), (1/6, 1/6, 2/3, 1/3)]  # (L1,L2,L3,weight)
_CENTROID = (1/3, 1/3, 1/3, 1.0)


def _shape_lst6(L1, L2, L3, dL1, dL2, dL3):
    """Quadratic LST shape functions and their (x,y) derivatives, given the
    (constant, since the triangle is straight-sided) area-coordinate
    gradients dLi = (dLi/dx, dLi/dy)."""
    N = np.array([
        L1 * (2*L1 - 1),
        L2 * (2*L2 - 1),
        L3 * (2*L3 - 1),
        4 * L1 * L2,
        4 * L2 * L3,
        4 * L3 * L1,
    ])
    dN_dx = np.array([
        (4*L1 - 1) * dL1[0],
        (4*L2 - 1) * dL2[0],
        (4*L3 - 1) * dL3[0],
        4 * (L2 * dL1[0] + L1 * dL2[0]),
        4 * (L3 * dL2[0] + L2 * dL3[0]),
        4 * (L1 * dL3[0] + L3 * dL1[0]),
    ])
    dN_dy = np.array([
        (4*L1 - 1) * dL1[1],
        (4*L2 - 1) * dL2[1],
        (4*L3 - 1) * dL3[1],
        4 * (L2 * dL1[1] + L1 * dL2[1]),
        4 * (L3 * dL2[1] + L2 * dL3[1]),
        4 * (L1 * dL3[1] + L3 * dL1[1]),
    ])
    return N, dN_dx, dN_dy


def lst6_mindlin_sri_element(coords, D_b, nu_m, Gs_m, rho_m, h_m, phi=1.0, stiffness_exponent=1.0):
    """coords: (6,2) array, corners first [p1,p2,p3], then mid-edges [m12,m23,m31].

    Integration: 3-point Gauss (degree-2 exact) for BOTH bending and shear.
    A single-centroid-point shear reduction (the SRI trick that works for
    4-node bilinear quads) was tried first and produced a rank-deficient,
    spuriously-zero-energy stiffness matrix here (negative/near-zero
    eigenvalues even after constraining all boundary DOFs) -- this
    quadratic element's shear strain field is naturally degree-2 (same
    order as bending), so 1 point is genuinely under-integrated relative
    to its DOF count, not merely "selectively reduced". 3-point Gauss for
    both fields is therefore full/consistent integration for this element,
    not partial locking remediation -- the quadratic (not constant) shear
    field is expected to lock far less severely than the 3-node linear
    triangle's constant shear field did."""
    p1, p2, p3 = coords[0], coords[1], coords[2]
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = abs(A2) / 2.0

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    dL1 = (b1 / A2, c1 / A2)
    dL2 = (b2 / A2, c2 / A2)
    dL3 = (b3 / A2, c3 / A2)

    Db = D_b * np.array([[1, nu_m, 0], [nu_m, 1, 0], [0, 0, (1 - nu_m) / 2]])
    Ds = Gs_m * np.eye(2)

    nd = 18  # 6 nodes x 3 dof
    Kb = np.zeros((nd, nd)); Ks = np.zeros((nd, nd)); Me = np.zeros((nd, nd))

    for (L1, L2, L3, w) in _GAUSS3:
        N, dNdx, dNdy = _shape_lst6(L1, L2, L3, dL1, dL2, dL3)
        wA = w * A

        Bb = np.zeros((3, nd))
        Bs = np.zeros((2, nd))
        for i in range(6):
            Bb[0, i*3 + 1] = dNdx[i]
            Bb[1, i*3 + 2] = dNdy[i]
            Bb[2, i*3 + 1] = dNdy[i]
            Bb[2, i*3 + 2] = dNdx[i]
            Bs[0, i*3] = dNdx[i]
            Bs[0, i*3 + 1] = -N[i]
            Bs[1, i*3] = dNdy[i]
            Bs[1, i*3 + 2] = -N[i]
        Kb += (Bb.T @ Db @ Bb) * wA
        Ks += (Bs.T @ Ds @ Bs) * wA

        mt = rho_m * h_m; mr = rho_m * h_m ** 3 / 12.0
        for i in range(6):
            for j in range(6):
                Me[i*3, j*3] += mt * N[i] * N[j] * wA
                Me[i*3+1, j*3+1] += mr * N[i] * N[j] * wA
                Me[i*3+2, j*3+2] += mr * N[i] * N[j] * wA

    Ke = (phi ** stiffness_exponent) * (Kb + Ks)
    Me = phi * Me
    return Ke, Me


def assemble(nodes, elems6, phi=None, stiffness_exponent=1.0):
    N = len(nodes); nd = 3 * N
    K = lil_matrix((nd, nd)); M = lil_matrix((nd, nd))
    if phi is None:
        phi = np.ones(len(elems6))
    for ei, el in enumerate(elems6):
        coords = nodes[el]
        Ke, Me = lst6_mindlin_sri_element(coords, D, nu, Gs, rho, h,
                                           phi=phi[ei], stiffness_exponent=stiffness_exponent)
        dofs = []
        for n in el: dofs += [n*3, n*3+1, n*3+2]
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


def debruijn_quasicrystal_points(n_fold, domain_x, domain_y, offset_seed=42, nrange=4):
    rng = np.random.default_rng(offset_seed)
    gammas = rng.uniform(0, 1, n_fold)
    dirs = [np.array([np.cos(k*np.pi/n_fold), np.sin(k*np.pi/n_fold)]) for k in range(n_fold)]
    scale = min(domain_x, domain_y) * 0.45
    pts = []
    for i in range(n_fold):
        for j in range(i+1, n_fold):
            d_i, d_j = dirs[i], dirs[j]
            n_hat_i = np.array([-d_i[1], d_i[0]])
            n_hat_j = np.array([-d_j[1], d_j[0]])
            A = np.array([n_hat_i, n_hat_j])
            det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
            if abs(det) < 1e-10: continue
            for ni in range(-nrange, nrange+1):
                for nj in range(-nrange, nrange+1):
                    b = np.array([ni+gammas[i], nj+gammas[j]])
                    P = np.linalg.solve(A, b)
                    px_m = P[0]*scale + domain_x/2
                    py_m = P[1]*scale + domain_y/2
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
    nodes, elems6, corners = build_lst6_mesh(Lx, Ly, n_grid)
    phi = element_coverage_fractions_lst6(nodes, corners, holes, hole_radius, sub_n=sub_n)
    cov = phi.mean() * 100
    K, M = assemble(nodes, elems6, phi=phi, stiffness_exponent=stiffness_exponent)
    free = clamped_free_dofs(nodes)
    freqs = solve_modes(K, M, free)
    return freqs, cov


def find_radius_for_coverage(n_fold, target_cov, n_grid, seed=42, sub_n=12,
                              r_lo=0.3e-6, r_hi=10.0e-6, tol=0.4, max_iter=40):
    holes = debruijn_quasicrystal_points(n_fold, Lx, Ly, offset_seed=seed)
    nodes, elems6, corners = build_lst6_mesh(Lx, Ly, n_grid)

    def cov_at(r):
        phi = element_coverage_fractions_lst6(nodes, corners, holes, r, sub_n=sub_n)
        return phi.mean() * 100

    lo, hi = r_lo, r_hi
    cov_hi_val = cov_at(hi)
    if cov_hi_val > target_cov + tol:
        return hi, cov_hi_val  # cannot reach target even at r_hi -- caller should notice
    mid, cov_mid = hi, cov_hi_val
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


if __name__ == "__main__":
    print("=== Step 1: Leissa benchmark, 6-node quadratic Mindlin triangle, SRI (phi=1) ===")
    f_anal = 35.99 / (2*np.pi*Lx**2) * np.sqrt(D/(rho*h))
    print(f"Analytical f1: {f_anal/1e6:.4f} MHz")
    for ng in [10, 14, 18, 24]:
        nodes, elems6, corners = build_lst6_mesh(Lx, Ly, ng)
        K, M = assemble(nodes, elems6)
        free = clamped_free_dofs(nodes)
        freqs = solve_modes(K, M, free)
        print(f"  n_grid={ng:3d}  corner_nodes={ng*ng:5d}  total_nodes={len(nodes):5d}  "
              f"elems={len(elems6):5d}  f1={freqs[0]/1e6:.5f} MHz  ratio={freqs[0]/f_anal:.4f}")

    NX = 18  # 3.67% off analytical at phi=1 -- reasonable working resolution given runtime cost
    TARGET_COV = 80.0
    n_folds = [3, 6, 8, 12]

    print(f"\n=== Step 2: coverage-matching bisection (target {TARGET_COV}%, n_grid={NX}, LST6 element) ===")
    matched = {}
    for nf in n_folds:
        r, cov = find_radius_for_coverage(nf, TARGET_COV, NX, seed=42)
        matched[nf] = r
        flag = "  *** could not reach target even at r_hi ***" if abs(cov - TARGET_COV) > 0.4 else ""
        print(f"  n_fold={nf:2d}  r={r*1e6:.4f} um  achieved coverage={cov:.2f}%{flag}")

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
    print(f"Cross-validation (LST6 element, homogenized, quadratic exponent) spread across n_fold: {spread:.2f}%")
