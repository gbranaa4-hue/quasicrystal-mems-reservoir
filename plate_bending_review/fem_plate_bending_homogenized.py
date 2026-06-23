"""
Real fix for the hole-geometry construction problem found across this whole
investigation: every prior version used BINARY element removal (an element
either fully exists or is fully voided, decided by a single centroid-in-hole
test). That construction is what produced disconnected mesh fragments,
non-converging under refinement, at every coverage level tested -- including
the supposedly "clean" 98% / r=1.5um case, just at low enough severity to be
missed until checked carefully.

THE FIX: area-fraction homogenization. For each element, compute the actual
fraction of its physical area NOT covered by any hole (via sub-sampling, not
a single point), and scale that element's stiffness and mass matrices by
that fraction. This is the same principle as SIMP-style penalization in
topology optimization -- material properties degrade continuously with
coverage, never discontinuously to exactly zero unless an element is fully
inside a hole. There is therefore no possible mesh fragmentation: every
element remains structurally present (even if very compliant), so global
connectivity is preserved by construction, not by post-hoc repair.

This trades one approximation (binary include/exclude, wrong) for another,
honestly-labeled approximation (homogenized/smeared material properties,
standard and defensible) -- it does NOT claim to exactly resolve discrete
hole boundaries at the sub-element scale, which would require true
conformal (constrained Delaunay) meshing. For the coverage levels and mesh
resolutions of interest here, it removes the catastrophic problem (mesh
disconnection) at the cost of a much milder, well-understood one (some
sub-element-scale hole geometry detail is smeared rather than sharp).
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

E, nu, rho, h = 170e9, 0.28, 2330.0, 100e-9
G = E / (2 * (1 + nu))
kap = 5.0 / 6.0
D = E * h ** 3 / (12 * (1 - nu ** 2))
Gs = kap * G * h

Lx = Ly = 100e-6


def build_mesh(Lx, Ly, nx, ny):
    xs = np.linspace(0, Lx, nx)
    ys = np.linspace(0, Ly, ny)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    quads = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i; n1 = n0 + 1; n2 = n0 + nx + 1; n3 = n0 + nx
            quads.append([n0, n1, n2, n3])
    return nodes, np.array(quads)


def element_coverage_fractions(nodes, quads, holes, radii, sub_n=12):
    """For each quad element, the fraction of its area NOT covered by any
    hole, via an sub_n x sub_n regular sub-sampling grid within the element
    (elements here are axis-aligned squares from the structured mesh, so
    this is just a bounding-box sub-grid test, no isoparametric mapping
    needed for the sampling itself)."""
    if not hasattr(radii, "__len__"):
        radii = np.full(len(holes), radii)

    coords = nodes[quads]  # (n_quads, 4, 2)
    xmin = coords[:, :, 0].min(axis=1)
    xmax = coords[:, :, 0].max(axis=1)
    ymin = coords[:, :, 1].min(axis=1)
    ymax = coords[:, :, 1].max(axis=1)

    su, sv = np.meshgrid(
        (np.arange(sub_n) + 0.5) / sub_n,
        (np.arange(sub_n) + 0.5) / sub_n,
    )
    su = su.ravel(); sv = sv.ravel()  # (sub_n^2,)

    n_quads = len(quads)
    phi = np.ones(n_quads)
    for qi in range(n_quads):
        px = xmin[qi] + su * (xmax[qi] - xmin[qi])
        py = ymin[qi] + sv * (ymax[qi] - ymin[qi])
        covered = np.zeros(len(px), dtype=bool)
        for (hx, hy), r in zip(holes, radii):
            covered |= (px - hx) ** 2 + (py - hy) ** 2 < r ** 2
        phi[qi] = 1.0 - covered.mean()
    # standard SIMP-style regularization: floor phi at a small positive value
    # rather than exactly 0, to avoid a singular (uninvertible) mass matrix
    # for nodes whose surrounding elements are all (near-)fully covered.
    # This is a numerical-conditioning safeguard, not a physical claim --
    # phi_min=1e-3 contributes negligible stiffness/mass while keeping the
    # eigensolver well-posed.
    phi = np.maximum(phi, 1e-3)
    return phi


def quad_mindlin_sri_element(coords, D_b, nu_m, Gs_m, rho_m, h_m, phi=1.0, stiffness_exponent=1.0):
    coords = np.asarray(coords)
    Db = D_b * np.array([[1, nu_m, 0], [nu_m, 1, 0], [0, 0, (1 - nu_m) / 2]])
    Ds = Gs_m * np.eye(2)

    def shape_derivs(xi, eta):
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                              (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        J = np.zeros((2, 2))
        J[0, 0] = dN_dxi @ coords[:, 0]; J[0, 1] = dN_dxi @ coords[:, 1]
        J[1, 0] = dN_deta @ coords[:, 0]; J[1, 1] = dN_deta @ coords[:, 1]
        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)
        dN = np.vstack([dN_dxi, dN_deta])
        dN_xy = Jinv @ dN
        return N, dN_xy[0], dN_xy[1], detJ

    nd = 12
    Kb = np.zeros((nd, nd)); Ks = np.zeros((nd, nd)); Me = np.zeros((nd, nd))

    gp = 1.0 / np.sqrt(3)
    for (xi, eta) in [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]:
        N, dNdx, dNdy, detJ = shape_derivs(xi, eta)
        Bb = np.zeros((3, nd))
        for i in range(4):
            Bb[0, i * 3 + 1] = dNdx[i]
            Bb[1, i * 3 + 2] = dNdy[i]
            Bb[2, i * 3 + 1] = dNdy[i]
            Bb[2, i * 3 + 2] = dNdx[i]
        Kb += (Bb.T @ Db @ Bb) * detJ

        mt = rho_m * h_m; mr = rho_m * h_m ** 3 / 12.0
        for i in range(4):
            for j in range(4):
                Me[i * 3, j * 3] += mt * N[i] * N[j] * detJ
                Me[i * 3 + 1, j * 3 + 1] += mr * N[i] * N[j] * detJ
                Me[i * 3 + 2, j * 3 + 2] += mr * N[i] * N[j] * detJ

    N, dNdx, dNdy, detJ = shape_derivs(0.0, 0.0)
    Bs = np.zeros((2, nd))
    for i in range(4):
        Bs[0, i * 3] = dNdx[i]
        Bs[0, i * 3 + 1] = -N[i]
        Bs[1, i * 3] = dNdy[i]
        Bs[1, i * 3 + 2] = -N[i]
    Ks += (Bs.T @ Ds @ Bs) * detJ * 4.0

    # homogenization: mass scales linearly with surviving material
    # (physically required -- mass is extensive). Stiffness uses a
    # separately adjustable exponent (default 1.0, i.e. also linear) --
    # see reviewer-requested sensitivity check using stiffness_exponent=2.0.
    Ke = (phi ** stiffness_exponent) * (Kb + Ks)
    Me = phi * Me
    return Ke, Me


def assemble(nodes, quads, phi=None, stiffness_exponent=1.0):
    N = len(nodes); nd = 3 * N
    K = lil_matrix((nd, nd)); M = lil_matrix((nd, nd))
    if phi is None:
        phi = np.ones(len(quads))
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
    bdry = ((nodes[:, 0] <= xmn + tol) | (nodes[:, 0] >= xmx - tol) |
            (nodes[:, 1] <= ymn + tol) | (nodes[:, 1] >= ymx - tol))
    con = []
    for i in np.where(bdry)[0]: con += [i * 3, i * 3 + 1, i * 3 + 2]
    return np.setdiff1d(np.arange(nd), con)


def solve_modes(K, M, free, n_modes=10):
    Kf = K[np.ix_(free, free)]; Mf = M[np.ix_(free, free)]
    k = min(n_modes, len(free) - 2)
    sigma = max(Kf.diagonal().max() * 1e-4, 1e-20)
    vals, vecs = eigsh(Kf, k=k, M=Mf, sigma=sigma, which='LM', tol=1e-6, maxiter=50000)
    pos = vals > 1e-6 * np.abs(vals).max()
    vals = vals[pos]
    freqs = np.sqrt(np.abs(vals)) / (2 * np.pi)
    return np.sort(freqs)


def debruijn_quasicrystal_points(n_fold, domain_x, domain_y, offset_seed=42, nrange=4):
    """Genuine de Bruijn multigrid line-intersection construction: n_fold
    families of parallel lines (one family per direction theta_k = k*pi/n_fold),
    each family's lines spaced unit distance apart along its own perpendicular
    and offset by an independent random phase gamma_k, with quasicrystal
    points placed at every pairwise intersection of lines from two different
    families -- line m of family k is {P : P . n_hat_k = m + gamma_k}, where
    n_hat_k is the unit normal to direction k.

    NOTE ON A PRIOR BUG: an earlier version of this function looped both ni
    and nj (the line indices for the two families) but its position formula
    never actually used nj, so it produced only ONE point per ni (the same
    point regardless of nj) -- i.e. a single straight line of points per
    direction pair, not the full 2D grid of pairwise intersections a genuine
    multigrid construction requires. That bug was the root cause of several
    previously-unexplained low-candidate-point-count issues at low n_fold
    documented elsewhere in this study. Fixed here by solving the proper 2x2
    linear system for each (ni, nj) pair; verified directly (both line
    equations are satisfied by the solved point to floating-point precision)
    and by checking the resulting point count scales sensibly and saturates
    with nrange (confirming all in-domain intersections are captured)."""
    rng = np.random.default_rng(offset_seed)
    gammas = rng.uniform(0, 1, n_fold)
    dirs = [np.array([np.cos(k * np.pi / n_fold), np.sin(k * np.pi / n_fold)]) for k in range(n_fold)]
    scale = min(domain_x, domain_y) * 0.45
    pts = []
    for i in range(n_fold):
        for j in range(i + 1, n_fold):
            d_i, d_j = dirs[i], dirs[j]
            n_hat_i = np.array([-d_i[1], d_i[0]])
            n_hat_j = np.array([-d_j[1], d_j[0]])
            A = np.array([n_hat_i, n_hat_j])
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-10: continue
            for ni in range(-nrange, nrange + 1):
                for nj in range(-nrange, nrange + 1):
                    b = np.array([ni + gammas[i], nj + gammas[j]])
                    P = np.linalg.solve(A, b)
                    px_m = P[0] * scale + domain_x / 2
                    py_m = P[1] * scale + domain_y / 2
                    if 0 < px_m < domain_x and 0 < py_m < domain_y:
                        pts.append([px_m, py_m])
    if not pts: return np.empty((0, 2))
    pts = np.array(pts)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]: continue
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        keep[i + 1:][d < scale * 0.05] = False
    return pts[keep]


def run_case(n_fold, hole_radius, nx, ny, seed=42, sub_n=12, stiffness_exponent=1.0):
    holes = debruijn_quasicrystal_points(n_fold, Lx, Ly, offset_seed=seed)
    nodes, quads = build_mesh(Lx, Ly, nx, ny)
    phi = element_coverage_fractions(nodes, quads, holes, hole_radius, sub_n=sub_n)
    cov = phi.mean() * 100  # mean coverage fraction, area-weighted (uniform elements -> simple mean)
    K, M = assemble(nodes, quads, phi=phi, stiffness_exponent=stiffness_exponent)
    free = clamped_free_dofs(nodes)
    freqs = solve_modes(K, M, free)
    return freqs, nodes, quads, holes, cov, phi


if __name__ == "__main__":
    print("=== Verification: Leissa CCCC benchmark (phi=1 everywhere, unperforated) ===")
    f_anal = 35.99 / (2 * np.pi * Lx ** 2) * np.sqrt(D / (rho * h))
    print(f"Analytical f1: {f_anal/1e6:.4f} MHz")
    for nx in [10, 16, 22, 30]:
        nodes, quads = build_mesh(Lx, Ly, nx, nx)
        K, M = assemble(nodes, quads)
        free = clamped_free_dofs(nodes)
        freqs = solve_modes(K, M, free, n_modes=6)
        print(f"  nx={nx:3d}  f1={freqs[0]/1e6:.5f} MHz  ratio={freqs[0]/f_anal:.4f}")

    print("\n=== Density sweep (n_fold=8, nx=22), homogenized hole representation ===")
    for r in [0.5e-6, 1.5e-6, 2.5e-6, 3.5e-6]:
        freqs, nodes, quads, holes, cov, phi = run_case(8, r, 22, 22)
        print(f"  r={r*1e6:.1f}um  cov={cov:.1f}%  f1={freqs[0]/1e6:.4f} MHz")

    print("\n=== Mesh-resolution convergence check on the homogenized density sweep (r=3.5um) ===")
    for nx in [22, 32, 44, 60]:
        freqs, nodes, quads, holes, cov, phi = run_case(8, 3.5e-6, nx, nx)
        print(f"  nx={nx:3d}  cov={cov:.1f}%  f1={freqs[0]/1e6:.4f} MHz")
