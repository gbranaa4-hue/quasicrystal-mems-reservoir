"""
Independent re-run of the pasted "corrected" 2D plate-bending FEM
(4-node bilinear Mindlin SRI quad, physical shear stiffness), reconstructed
verbatim from the file contents shown, to verify the claimed Leissa
benchmark convergence and density/symmetry results on this machine,
independent of whatever environment originally produced them.
(Plotting/mode-shape export omitted -- matplotlib not available here;
not needed to verify the numerical claims.)
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import time

# ---- Material ----
E    = 170e9
nu   = 0.28
rho  = 2330.0
h    = 100e-9
G    = E / (2*(1+nu))
kap  = 5.0/6.0
D    = E*h**3 / (12*(1-nu**2))
Gs   = kap * G * h

Lx = 100e-6
Ly = 100e-6


def debruijn_quasicrystal_points(n_fold, domain_x, domain_y, offset_seed=42):
    rng = np.random.default_rng(offset_seed)
    gammas = rng.uniform(0, 1, n_fold)
    dirs   = [np.array([np.cos(k*np.pi/n_fold), np.sin(k*np.pi/n_fold)])
              for k in range(n_fold)]
    scale  = min(domain_x, domain_y) * 0.45
    pts = []
    for i in range(n_fold):
        for j in range(i+1, n_fold):
            d0, d1 = dirs[i], dirs[j]
            cr = d0[0]*d1[1] - d0[1]*d1[0]
            if abs(cr) < 1e-10: continue
            for ni in range(-8, 9):
                for nj in range(-8, 9):
                    r  = (ni + gammas[i] - gammas[j]) / cr
                    px = (ni + gammas[i])*d0[0] + r*d1[0]
                    py = (ni + gammas[i])*d0[1] + r*d1[1]
                    px_m = px*scale + domain_x/2
                    py_m = py*scale + domain_y/2
                    if 0 < px_m < domain_x and 0 < py_m < domain_y:
                        pts.append([px_m, py_m])
    if not pts: return np.empty((0,2))
    pts = np.array(pts)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]: continue
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
        keep[i+1:][d < scale*0.05] = False
    return pts[keep]


def build_mesh(Lx, Ly, nx, ny):
    xs = np.linspace(0, Lx, nx)
    ys = np.linspace(0, Ly, ny)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    quads = []
    for j in range(ny-1):
        for i in range(nx-1):
            n0 = j*nx+i; n1 = n0+1; n2 = n0+nx+1; n3 = n0+nx
            quads.append([n0, n1, n2, n3])
    return nodes, np.array(quads)


def remove_holes(nodes, quads, holes, radius):
    c = nodes[quads].mean(axis=1)
    ok = np.ones(len(quads), dtype=bool)
    for hc in holes:
        ok &= np.linalg.norm(c - hc, axis=1) > radius
    qa = quads[ok]
    used = np.zeros(len(nodes), dtype=bool)
    used[qa.ravel()] = True
    return qa, used


def _largest_connected_component(nodes, quads, boundary_mask=None):
    """Same fix as used in the earlier in-plane (CST) investigation: naive
    hole-cutting on a structured grid can leave disconnected slivers/islands,
    which the eigensolver treats as nearly-unconstrained free bodies,
    producing spurious near-zero-frequency 'modes' -- exactly the failure
    seen in the n_fold=8, 80%-coverage run during the M2 retest. Keep only
    the component containing the most boundary (clamped) nodes, since that's
    the one that can actually be a valid clamped-plate problem."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(nodes)
    rows, cols = [], []
    for q in quads:
        n0, n1, n2, n3 = q
        for a, b in [(n0, n1), (n1, n2), (n2, n3), (n3, n0)]:
            rows += [a, b]; cols += [b, a]
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp <= 1:
        return quads, n_comp, n, n

    if boundary_mask is not None and boundary_mask.any():
        counts = np.bincount(labels[boundary_mask], minlength=n_comp)
        main_label = np.argmax(counts)
    else:
        counts = np.bincount(labels, minlength=n_comp)
        main_label = np.argmax(counts)
    keep_nodes = np.where(labels == main_label)[0]
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[keep_nodes] = True
    quad_keep = keep_mask[quads].all(axis=1)
    return quads[quad_keep], n_comp, len(keep_nodes), n


def remove_holes_safe(nodes, quads, holes, radius, domain_size):
    """remove_holes + connectivity repair. Use this instead of remove_holes
    directly whenever hole density is high enough that fragmentation is a
    real risk (in practice: coverage below ~85-90% with this quasicrystal
    construction)."""
    qa, used = remove_holes(nodes, quads, holes, radius)
    tol = domain_size / 50  # generous; only used to identify boundary nodes
    xmn, xmx = nodes[:, 0].min(), nodes[:, 0].max()
    ymn, ymx = nodes[:, 1].min(), nodes[:, 1].max()
    boundary_mask = ((nodes[:, 0] <= xmn+tol) | (nodes[:, 0] >= xmx-tol) |
                      (nodes[:, 1] <= ymn+tol) | (nodes[:, 1] >= ymx-tol))
    qa_fixed, n_comp, kept_n, total_n = _largest_connected_component(nodes, qa, boundary_mask)
    used = np.zeros(len(nodes), dtype=bool)
    used[qa_fixed.ravel()] = True
    return qa_fixed, used, n_comp


def quad_mindlin_sri_element(coords, D_b, nu_m, Gs_m, rho_m, h_m):
    coords = np.asarray(coords)
    Db = D_b * np.array([[1, nu_m, 0], [nu_m, 1, 0], [0, 0, (1-nu_m)/2]])
    Ds = Gs_m * np.eye(2)

    def shape_derivs(xi, eta):
        dN_dxi  = 0.25*np.array([-(1-eta),  (1-eta), (1+eta), -(1+eta)])
        dN_deta = 0.25*np.array([-(1-xi),  -(1+xi),  (1+xi),  (1-xi)])
        N = 0.25*np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                            (1+xi)*(1+eta), (1-xi)*(1+eta)])
        J = np.zeros((2,2))
        J[0,0] = dN_dxi @ coords[:,0];  J[0,1] = dN_dxi @ coords[:,1]
        J[1,0] = dN_deta @ coords[:,0]; J[1,1] = dN_deta @ coords[:,1]
        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)
        dN = np.vstack([dN_dxi, dN_deta])
        dN_xy = Jinv @ dN
        return N, dN_xy[0], dN_xy[1], detJ

    nd = 12
    Kb = np.zeros((nd, nd))
    Ks = np.zeros((nd, nd))
    Me = np.zeros((nd, nd))

    gp = 1.0/np.sqrt(3)
    for (xi, eta) in [(-gp,-gp), (gp,-gp), (gp,gp), (-gp,gp)]:
        N, dNdx, dNdy, detJ = shape_derivs(xi, eta)
        Bb = np.zeros((3, nd))
        for i in range(4):
            Bb[0, i*3+1] = dNdx[i]
            Bb[1, i*3+2] = dNdy[i]
            Bb[2, i*3+1] = dNdy[i]
            Bb[2, i*3+2] = dNdx[i]
        Kb += (Bb.T @ Db @ Bb) * detJ

        mt = rho_m*h_m
        mr = rho_m*h_m**3/12.0
        for i in range(4):
            for j in range(4):
                Me[i*3,   j*3  ] += mt*N[i]*N[j]*detJ
                Me[i*3+1, j*3+1] += mr*N[i]*N[j]*detJ
                Me[i*3+2, j*3+2] += mr*N[i]*N[j]*detJ

    N, dNdx, dNdy, detJ = shape_derivs(0.0, 0.0)
    Bs = np.zeros((2, nd))
    for i in range(4):
        Bs[0, i*3]   = dNdx[i]
        Bs[0, i*3+1] = -N[i]
        Bs[1, i*3]   = dNdy[i]
        Bs[1, i*3+2] = -N[i]
    Ks += (Bs.T @ Ds @ Bs) * detJ * 4.0

    Ke = Kb + Ks
    return Ke, Me


def assemble(nodes, quads):
    N  = len(nodes); nd = 3*N
    K  = lil_matrix((nd,nd))
    M  = lil_matrix((nd,nd))
    for q in quads:
        coords = nodes[q]
        Ke, Me = quad_mindlin_sri_element(coords, D, nu, Gs, rho, h)
        dofs = []
        for n in q: dofs += [n*3, n*3+1, n*3+2]
        for i,di in enumerate(dofs):
            for j,dj in enumerate(dofs):
                K[di,dj] += Ke[i,j]
                M[di,dj] += Me[i,j]
    return csr_matrix(K), csr_matrix(M)


def clamped_free_dofs(nodes):
    nd = 3*len(nodes)
    tol = 1e-10
    xmn,xmx = nodes[:,0].min(), nodes[:,0].max()
    ymn,ymx = nodes[:,1].min(), nodes[:,1].max()
    bdry = ((nodes[:,0]<=xmn+tol)|(nodes[:,0]>=xmx-tol)|
            (nodes[:,1]<=ymn+tol)|(nodes[:,1]>=ymx-tol))
    con = []
    for i in np.where(bdry)[0]: con += [i*3,i*3+1,i*3+2]
    return np.setdiff1d(np.arange(nd), con)


def solve_modes(K, M, free, n_modes=10):
    Kf = K[np.ix_(free,free)]
    Mf = M[np.ix_(free,free)]
    k  = min(n_modes, len(free)-2)
    if k < 1: return np.array([]), np.zeros((len(free),0))
    try:
        sigma = max(Kf.diagonal().max()*1e-4, 1e-20)
        vals, vecs = eigsh(Kf, k=k, M=Mf, sigma=sigma, which='LM',
                           tol=1e-6, maxiter=50000)
    except Exception:
        vals, vecs = eigsh(Kf, k=k, M=Mf, which='SM', tol=1e-5, maxiter=50000)
    pos = vals > 1e-6*np.abs(vals).max()
    vals=vals[pos]; vecs=vecs[:,pos]
    if len(vals)==0: return np.array([]), np.zeros((len(free),0))
    freqs = np.sqrt(np.abs(vals))/(2*np.pi)
    idx = np.argsort(freqs)
    return freqs[idx], vecs[:,idx]


def run_case(n_fold, hole_radius, nx, ny, seed=42, safe=True):
    holes = debruijn_quasicrystal_points(n_fold, Lx, Ly, offset_seed=seed)
    nodes, quads = build_mesh(Lx, Ly, nx, ny)
    if safe:
        qa, used, n_comp = remove_holes_safe(nodes, quads, holes, hole_radius, Lx)
        if n_comp > 1:
            print(f"    [mesh] n_fold={n_fold} r={hole_radius*1e6:.2f}um: "
                  f"{n_comp} components found, repaired to main component")
    else:
        qa, used = remove_holes(nodes, quads, holes, hole_radius)
    nmap = np.full(len(nodes),-1)
    act  = np.where(used)[0]
    nmap[act] = np.arange(len(act))
    nr = nodes[act]; qr = nmap[qa]
    if len(qr)==0: return None
    cov  = len(qr)/len(quads)*100
    K,M  = assemble(nr, qr)
    free = clamped_free_dofs(nr)
    freqs, vecs = solve_modes(K, M, free)
    return freqs, vecs, nr, qr, holes, cov, free


if __name__ == "__main__":
    print("=== Independent re-verification: Leissa CCCC benchmark ===")
    f_anal = 35.99/(2*np.pi*Lx**2)*np.sqrt(D/(rho*h))
    print(f"Analytical f1 (Leissa CCCC): {f_anal/1e6:.4f} MHz\n")
    for nx in [10, 16, 22, 30]:
        nodes, quads = build_mesh(Lx, Ly, nx, nx)
        K, M = assemble(nodes, quads)
        free = clamped_free_dofs(nodes)
        t0 = time.time()
        freqs, _ = solve_modes(K, M, free, n_modes=6)
        f1 = freqs[0] if len(freqs) else float('nan')
        print(f"nx={nx:3d}  nodes={len(nodes):5d}  f1={f1/1e6:.5f} MHz  "
              f"ratio={f1/f_anal:.4f}  ({time.time()-t0:.1f}s)")
