"""
Independent cross-check of the quad-SRI plate-bending FEM (fem_plate_bending_2d_v2.py)
using a SECOND, differently-formulated element: a 3-node Mindlin-Reissner
triangle, with the physical (unfitted) shear stiffness Gs = kappa*G*h --
same physical material constant as the quad element, but a different shape
function family, different mesh topology (triangles, not quads), and a
separately-written stiffness/mass assembly.

This is NOT expected to match the quad element's absolute frequencies
exactly -- 3-node Mindlin triangles with linear shape functions are known
to be more prone to shear locking than the quad+SRI formulation, even with
the physical (not fitted) shear stiffness. The purpose here is narrower and
honest: does an independently-coded element, on the same perforated
geometry, reproduce the same QUALITATIVE conclusions (density matters,
symmetry doesn't, once coverage is controlled)? Agreement in trend from a
structurally different element is real corroborating evidence; agreement
in exact absolute value is not expected and not claimed.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay

E, nu, rho, h = 170e9, 0.28, 2330.0, 100e-9
G = E / (2 * (1 + nu))
kap = 5.0 / 6.0
D = E * h ** 3 / (12 * (1 - nu ** 2))
Gs = kap * G * h  # physical shear stiffness, same constant as the quad element

Lx = Ly = 100e-6


def build_tri_mesh(Lx, Ly, n_grid):
    """Delaunay triangulation of a structured point grid -- a different mesh
    topology than the quad element's structured quad grid."""
    xs = np.linspace(0, Lx, n_grid)
    ys = np.linspace(0, Ly, n_grid)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    tri = Delaunay(nodes)
    return nodes, tri.simplices


def remove_holes(nodes, tris, holes, radius):
    c = nodes[tris].mean(axis=1)
    ok = np.ones(len(tris), dtype=bool)
    for hc in holes:
        ok &= np.linalg.norm(c - hc, axis=1) > radius
    ta = tris[ok]
    used = np.zeros(len(nodes), dtype=bool)
    used[ta.ravel()] = True
    return ta, used


def mindlin_triangle_element(p1, p2, p3, D_b, nu_m, Gs_m, rho_m, h_m):
    """3-node Mindlin triangle, physical (unfitted) shear stiffness.
    DOF: [w1,tx1,ty1, w2,tx2,ty2, w3,tx3,ty3]. Linear shape functions ->
    constant strain/curvature fields (both Bb and Bs are single-point
    quantities for this element; that is a structural property of linear
    triangles, not a chosen integration scheme)."""
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

    Ke = Kb + Ks

    Me = np.zeros((9, 9))
    mt = rho_m * h_m * A
    mr = rho_m * h_m ** 3 / 12.0 * A
    for i in range(3):
        for j in range(3):
            fac = (2.0 if i == j else 1.0) / 12.0
            Me[i * 3, j * 3] = mt * fac
            Me[i * 3 + 1, j * 3 + 1] = mr * fac
            Me[i * 3 + 2, j * 3 + 2] = mr * fac
    return Ke, Me


def assemble(nodes, tris):
    N = len(nodes); nd = 3 * N
    K = lil_matrix((nd, nd)); M = lil_matrix((nd, nd))
    for tri in tris:
        n0, n1, n2 = tri
        Ke, Me = mindlin_triangle_element(nodes[n0], nodes[n1], nodes[n2], D, nu, Gs, rho, h)
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


def run_case(n_fold, hole_radius, n_grid, seed=42):
    holes = debruijn_quasicrystal_points(n_fold, Lx, Ly, offset_seed=seed)
    nodes, tris = build_tri_mesh(Lx, Ly, n_grid)
    ta, used = remove_holes(nodes, tris, holes, hole_radius)
    nmap = np.full(len(nodes), -1)
    act = np.where(used)[0]
    nmap[act] = np.arange(len(act))
    nr = nodes[act]; tr = nmap[ta]
    if len(tr) == 0: return None
    K, M = assemble(nr, tr)
    free = clamped_free_dofs(nr)
    freqs = solve_modes(K, M, free)
    return freqs


if __name__ == "__main__":
    print("=== Cross-check element: Leissa benchmark (expect some locking vs quad-SRI) ===")
    f_anal = 35.99 / (2*np.pi*Lx**2) * np.sqrt(D/(rho*h))
    for ng in [16, 22, 30, 40]:
        nodes, tris = build_tri_mesh(Lx, Ly, ng)
        K, M = assemble(nodes, tris)
        free = clamped_free_dofs(nodes)
        freqs = solve_modes(K, M, free)
        print(f"  n_grid={ng:3d}  nodes={len(nodes):5d}  f1={freqs[0]/1e6:.5f} MHz  ratio={freqs[0]/f_anal:.4f}")

    print("\n=== Cross-check: density sweep (n_fold=8, n_grid=26) ===")
    for r in [0.5e-6, 1.5e-6, 2.5e-6, 3.5e-6]:
        freqs = run_case(8, r, 26)
        if freqs is None or len(freqs) == 0:
            print(f"  r={r*1e6:.1f}um -> no modes"); continue
        print(f"  r={r*1e6:.1f}um  f1={freqs[0]/1e6:.4f} MHz")

    print("\n=== Cross-check: coverage-matched-radius symmetry sweep (n_grid=26) ===")
    # reuse the quad element's already-found coverage-matched radii (not re-deriving
    # bisection here -- the point is to check if a DIFFERENT element agrees on the
    # SAME geometry, not to re-run the bisection search itself)
    matched = {3: 3.000e-6, 6: 1.819e-6, 8: 1.45e-6, 12: 0.975e-6}
    results = {}
    for nf, r in matched.items():
        freqs = run_case(nf, r, 26)
        if freqs is None or len(freqs) == 0:
            print(f"  n_fold={nf} -> no modes"); continue
        results[nf] = freqs[0]
        print(f"  n_fold={nf:2d}  r={r*1e6:.3f}um  f1={freqs[0]/1e6:.4f} MHz")
    if len(results) >= 2:
        vals = np.array(list(results.values()))
        spread = (vals.max()-vals.min())/vals.mean()*100
        print(f"\n  cross-check element spread across n_fold: {spread:.2f}%")
