"""
Real (if minimal) FEM: 2D plane-stress linear elasticity, constant-strain
triangle (CST) elements, generalized eigenvalue solve for natural
frequencies and mode shapes of a clamped membrane with a quasicrystal hole
pattern actually cut out of the mesh (not "softened material" like the
earlier scalar toy model).

This is genuinely vector FEM: each node has 2 DOF (ux, uy), not 1. It is
still a simplification relative to a full plate/shell FEM (which would also
capture out-of-plane bending and shear) -- this solves in-plane elasticity,
which is the right model for in-plane vibration modes and is enough to get
real eigenfrequencies and check for literal spectral gaps, which the
scalar toy model could not do.

Material: silicon (matches the silicon_die_8mm case used earlier).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import connected_components

from phononic_symmetry_grading_sim import debruijn_quasicrystal_points

# ---- material: silicon, plane stress ----
E = 170e9         # Pa, Young's modulus
NU = 0.28          # Poisson's ratio
RHO = 2330.0        # kg/m^3
THICKNESS = 1.0      # m -- cancels in the eigenfrequency (K and M both scale linearly with t)

D_PLANE_STRESS = E / (1 - NU ** 2) * np.array([
    [1, NU, 0],
    [NU, 1, 0],
    [0, 0, (1 - NU) / 2],
])


def _triangle_area(nodes, simplices):
    p1, p2, p3 = nodes[simplices[:, 0]], nodes[simplices[:, 1]], nodes[simplices[:, 2]]
    return 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                         (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))


def _largest_connected_component(nodes, simplices, boundary_mask=None):
    """Keep only the largest connected component of the mesh graph -- the
    naive hole-cutting in this module (remove-by-centroid on a structured
    grid, not a true conformal/boundary-fitted mesh) can leave disconnected
    slivers and isolated islands. Those produce spurious near-zero-frequency
    'modes' (weakly/never connected to the clamped boundary) and, at higher
    hole density, a fully singular stiffness matrix. This is the real fix,
    not a numerical workaround: a structural problem in the mesh, fixed
    structurally by discarding everything not part of the main connected
    body."""
    n = len(nodes)
    rows, cols = [], []
    for tri in simplices:
        i, j, k = tri
        rows += [i, j, k]
        cols += [j, k, i]
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)

    if boundary_mask is not None and boundary_mask.any():
        # the clamped body must include the outer boundary ring -- pick the
        # component with the most boundary-touching nodes, not just the
        # most total nodes (those can disagree if hole-cutting disconnects
        # part of the boundary ring itself from the main interior mass)
        boundary_counts = np.bincount(labels[boundary_mask], minlength=n_comp)
        main_label = np.argmax(boundary_counts)
    else:
        counts = np.bincount(labels, minlength=n_comp)
        main_label = np.argmax(counts)
    keep_nodes = np.where(labels == main_label)[0]

    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[keep_nodes] = True
    tri_keep = keep_mask[simplices].all(axis=1)
    simplices_kept = simplices[tri_keep]

    used = np.unique(simplices_kept)
    remap = -np.ones(n, dtype=int)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[simplices_kept], n_comp, len(keep_nodes), n


def _prune_pendant_nodes(nodes, simplices, min_triangle_degree=2, max_passes=10):
    """Iteratively remove nodes touched by very few triangles (dangling
    slivers attached to the main body by a thin sliver of material) -- these
    create unrealistically low-stiffness local paths and spurious very-low
    frequency modes even when the mesh is technically one connected piece."""
    for _ in range(max_passes):
        n = len(nodes)
        tri_count = np.zeros(n, dtype=int)
        for tri in simplices:
            tri_count[tri] += 1
        bad = tri_count < min_triangle_degree
        if not bad.any():
            break
        keep_mask = ~bad
        tri_keep = keep_mask[simplices].all(axis=1)
        simplices = simplices[tri_keep]
        used = np.unique(simplices)
        remap = -np.ones(n, dtype=int)
        remap[used] = np.arange(len(used))
        nodes = nodes[used]
        simplices = remap[simplices]
    return nodes, simplices


def build_mesh(domain_size, n_grid, hole_centers, hole_radius, min_area_frac=0.05):
    """Structured grid of points -> Delaunay triangulation -> drop any
    triangle whose centroid falls inside a hole -> discard degenerate
    slivers -> keep only the largest connected component -> prune dangling
    pendant nodes. Returns nodes (M,2), triangles (T,3) as node indices, and
    the set of boundary node indices (outer edge of the square domain, used
    for clamped BC)."""
    xs = np.linspace(0, domain_size, n_grid)
    ys = np.linspace(0, domain_size, n_grid)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.stack([xx.ravel(), yy.ravel()], axis=1)

    tri = Delaunay(nodes)
    simplices = tri.simplices

    if len(hole_centers) > 0:
        centroids = nodes[simplices].mean(axis=1)
        radii = hole_radius if hasattr(hole_radius, "__len__") else [hole_radius] * len(hole_centers)
        keep = np.ones(len(simplices), dtype=bool)
        for (hx, hy), r in zip(hole_centers, radii):
            d2 = (centroids[:, 0] - hx) ** 2 + (centroids[:, 1] - hy) ** 2
            keep &= d2 > r ** 2
        simplices = simplices[keep]

    # drop degenerate/sliver triangles (tiny area relative to nominal grid cell)
    nominal_area = (domain_size / (n_grid - 1)) ** 2 / 2
    areas = _triangle_area(nodes, simplices)
    simplices = simplices[areas > min_area_frac * nominal_area]

    # drop nodes no longer referenced by any kept triangle, remap indices
    used = np.unique(simplices)
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    nodes_new = nodes[used]
    simplices_new = remap[simplices]

    tol = domain_size / (n_grid - 1) * 0.5
    pre_boundary_mask = (
        (nodes_new[:, 0] < tol) | (nodes_new[:, 0] > domain_size - tol) |
        (nodes_new[:, 1] < tol) | (nodes_new[:, 1] > domain_size - tol)
    )

    nodes_new, simplices_new, n_comp, kept_n, total_n = _largest_connected_component(
        nodes_new, simplices_new, boundary_mask=pre_boundary_mask)
    if n_comp > 1:
        print(f"    [mesh] {n_comp} disconnected components found after hole-cutting; "
              f"kept main component ({kept_n}/{total_n} nodes), discarded the rest.")

    nodes_new, simplices_new = _prune_pendant_nodes(nodes_new, simplices_new)

    tol = domain_size / (n_grid - 1) * 0.5
    boundary = np.where(
        (nodes_new[:, 0] < tol) | (nodes_new[:, 0] > domain_size - tol) |
        (nodes_new[:, 1] < tol) | (nodes_new[:, 1] > domain_size - tol)
    )[0]

    return nodes_new, simplices_new, boundary


def build_strip_mesh(length, width, n_grid_x, n_grid_y, hole_centers, hole_radius,
                      min_area_frac=0.05):
    """Like build_mesh, but a rectangular strip where only the top/bottom
    (long) edges are clamped -- the two ends (x=0, x=length) are left free
    so a wave can actually propagate down the strip instead of being
    confined to a closed eigenvalue problem. This is the geometry rainbow
    trapping needs: a propagating-wave problem, not a standing-wave one."""
    xs = np.linspace(0, length, n_grid_x)
    ys = np.linspace(0, width, n_grid_y)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.stack([xx.ravel(), yy.ravel()], axis=1)

    tri = Delaunay(nodes)
    simplices = tri.simplices

    if len(hole_centers) > 0:
        centroids = nodes[simplices].mean(axis=1)
        radii = hole_radius if hasattr(hole_radius, "__len__") else [hole_radius] * len(hole_centers)
        keep = np.ones(len(simplices), dtype=bool)
        for (hx, hy), r in zip(hole_centers, radii):
            d2 = (centroids[:, 0] - hx) ** 2 + (centroids[:, 1] - hy) ** 2
            keep &= d2 > r ** 2
        simplices = simplices[keep]

    nominal_area = (width / (n_grid_y - 1)) * (length / (n_grid_x - 1)) / 2
    areas = _triangle_area(nodes, simplices)
    simplices = simplices[areas > min_area_frac * nominal_area]

    used = np.unique(simplices)
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    nodes_new = nodes[used]
    simplices_new = remap[simplices]

    tol_y = width / (n_grid_y - 1) * 0.5
    pre_clamp_mask = (nodes_new[:, 1] < tol_y) | (nodes_new[:, 1] > width - tol_y)

    nodes_new, simplices_new, n_comp, kept_n, total_n = _largest_connected_component(
        nodes_new, simplices_new, boundary_mask=pre_clamp_mask)
    if n_comp > 1:
        print(f"    [mesh] {n_comp} disconnected components found after hole-cutting; "
              f"kept main component ({kept_n}/{total_n} nodes), discarded the rest.")

    nodes_new, simplices_new = _prune_pendant_nodes(nodes_new, simplices_new)

    tol_y = width / (n_grid_y - 1) * 0.5
    clamped = np.where((nodes_new[:, 1] < tol_y) | (nodes_new[:, 1] > width - tol_y))[0]

    return nodes_new, simplices_new, clamped


def cst_matrices(p1, p2, p3):
    """Constant-strain-triangle stiffness and consistent mass matrix
    (6x6 each, DOF order ux1,uy1,ux2,uy2,ux3,uy3). Standard linear FEM."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = abs(A2) / 2.0
    if A < 1e-18:
        return np.zeros((6, 6)), np.zeros((6, 6)), 0.0

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    B = (1 / A2) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3],
    ])

    Ke = THICKNESS * A * (B.T @ D_PLANE_STRESS @ B)

    Me = RHO * THICKNESS * A / 12.0 * np.array([
        [2, 0, 1, 0, 1, 0],
        [0, 2, 0, 1, 0, 1],
        [1, 0, 2, 0, 1, 0],
        [0, 1, 0, 2, 0, 1],
        [1, 0, 1, 0, 2, 0],
        [0, 1, 0, 1, 0, 2],
    ])
    return Ke, Me, A


def assemble(nodes, triangles):
    n_dof = 2 * len(nodes)
    K = sp.lil_matrix((n_dof, n_dof))
    M = sp.lil_matrix((n_dof, n_dof))
    total_area = 0.0

    for tri in triangles:
        i, j, k = tri
        Ke, Me, A = cst_matrices(nodes[i], nodes[j], nodes[k])
        total_area += A
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1]
        for a in range(6):
            for b in range(6):
                K[dofs[a], dofs[b]] += Ke[a, b]
                M[dofs[a], dofs[b]] += Me[a, b]

    return K.tocsr(), M.tocsr(), total_area


def solve_eigenmodes(K, M, boundary_nodes, n_modes=20, return_vectors=False, n_nodes=None):
    """Clamp the boundary (Dirichlet, u=0), solve the generalized
    eigenproblem K v = omega^2 M v on the remaining free DOFs."""
    n_dof = K.shape[0]
    fixed_dofs = np.concatenate([2 * boundary_nodes, 2 * boundary_nodes + 1])
    free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

    Kff = K[free_dofs][:, free_dofs]
    Mff = M[free_dofs][:, free_dofs]

    # shift-invert around a small positive sigma to get the lowest modes
    eigvals, eigvecs = spla.eigsh(Kff, k=n_modes, M=Mff, sigma=1e-3, which="LM")
    eigvals = np.clip(eigvals, 0, None)
    freqs_hz = np.sqrt(eigvals) / (2 * np.pi)
    order = np.argsort(freqs_hz)

    if not return_vectors:
        return freqs_hz[order]

    # expand each free-DOF eigenvector back to full (clamped) node displacement field
    full_modes = np.zeros((n_dof, len(order)))
    full_modes[free_dofs, :] = eigvecs[:, order]
    return freqs_hz[order], full_modes


def run_case(n_fold, domain_size=100e-6, n_grid=34, hole_radius=2.5e-6,
             window_radius=6.0, grid_index_range=2, n_modes=20):
    print(f"\n=== n_fold={n_fold} ===")
    pts = debruijn_quasicrystal_points(n_fold, window_radius, grid_index_range=grid_index_range)
    # scale+center the quasicrystal point set into the physical domain
    scale = domain_size / (2 * window_radius) * 1.3
    centers = pts * scale + domain_size / 2

    nodes, triangles, boundary = build_mesh(domain_size, n_grid, centers, hole_radius)
    print(f"  mesh: {len(nodes)} nodes, {len(triangles)} elements, "
          f"{len(centers)} holes, {len(boundary)} clamped boundary nodes")
    if len(triangles) < 10:
        print("  WARNING: too few elements survived hole-cutting -- hole density too high for this mesh.")
        return None

    K, M, total_area = assemble(nodes, triangles)
    print(f"  remaining membrane area: {total_area*1e12:.1f} um^2 "
          f"({100*total_area/domain_size**2:.1f}% of domain)")

    try:
        freqs = solve_eigenmodes(K, M, boundary, n_modes=n_modes)
    except Exception as e:
        print(f"  eigensolve failed: {e}")
        return None

    print(f"  first {len(freqs)} eigenfrequencies (Hz):")
    print("   " + ", ".join(f"{f:,.0f}" for f in freqs))

    gaps = np.diff(freqs)
    if len(gaps) > 0:
        biggest_gap_idx = np.argmax(gaps)
        print(f"  largest spectral gap: {gaps[biggest_gap_idx]:,.0f} Hz, "
              f"between mode {biggest_gap_idx} ({freqs[biggest_gap_idx]:,.0f} Hz) "
              f"and mode {biggest_gap_idx+1} ({freqs[biggest_gap_idx+1]:,.0f} Hz)")
    return freqs


def density_sweep(n_fold=8, hole_radii=(1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6, 5.0e-6)):
    """Controlled test: fix symmetry order and hole *positions*, vary only
    hole_radius (hence coverage fraction). Isolates density as the single
    variable, unlike the four n_fold cases above (which didn't have matched
    coverage fractions)."""
    print(f"\n=== controlled density sweep, n_fold={n_fold} fixed ===")
    rows = []
    for hr in hole_radii:
        freqs = run_case(n_fold, hole_radius=hr)
        rows.append((hr, freqs))
    print("\n  --- density sweep summary ---")
    for hr, freqs in rows:
        if freqs is None or len(freqs) < 4:
            print(f"  hole_radius={hr*1e6:.1f}um: failed or too few modes")
            continue
        gaps = np.diff(freqs)
        gap_idx = np.argmax(gaps)
        print(f"  hole_radius={hr*1e6:.1f}um: f_min={freqs[0]/1e6:.2f}MHz, "
              f"gap at {freqs[gap_idx]/1e6:.2f}-{freqs[gap_idx+1]/1e6:.2f}MHz "
              f"(width {gaps[gap_idx]/1e6:.2f}MHz)")
    return rows


def graded_density_membrane(n_fold=8, domain_size=100e-6, n_grid=34,
                             r_min=1.0e-6, r_max=4.0e-6,
                             window_radius=6.0, grid_index_range=2, n_modes=24):
    """Single membrane, hole positions fixed (same n_fold pattern as the
    density sweep), but hole RADIUS varies linearly with x: small holes
    (high coverage, high local frequency per the sweep above) at x=0,
    large holes (low coverage, low local frequency) at x=domain_size.

    Real test of the revised hypothesis: does each eigenmode's vibration
    concentrate (spatially localize) on the side of the membrane whose
    local density 'belongs' to that mode's frequency -- i.e. does this
    produce an FEM-verified rainbow-trapping-like spatial sorting, not
    just a sweep of separate fixed-density samples?"""
    print(f"\n=== graded-density membrane: hole radius {r_min*1e6:.1f}um -> {r_max*1e6:.1f}um across x ===")
    pts = debruijn_quasicrystal_points(n_fold, window_radius, grid_index_range=grid_index_range)
    scale = domain_size / (2 * window_radius) * 1.3
    centers = pts * scale + domain_size / 2
    frac_x = np.clip(centers[:, 0] / domain_size, 0, 1)
    radii = r_min + (r_max - r_min) * frac_x

    nodes, triangles, boundary = build_mesh(domain_size, n_grid, centers, radii)
    print(f"  mesh: {len(nodes)} nodes, {len(triangles)} elements, {len(centers)} holes")
    if len(triangles) < 10:
        print("  WARNING: too few elements survived -- density gradient too aggressive.")
        return

    K, M, total_area = assemble(nodes, triangles)
    print(f"  remaining membrane area: {100*total_area/domain_size**2:.1f}% of domain")

    try:
        freqs, modes = solve_eigenmodes(K, M, boundary, n_modes=n_modes, return_vectors=True)
    except Exception as e:
        print(f"  eigensolve failed: {e}")
        return

    print(f"\n  mode # | freq (MHz) | displacement x-centroid (um, 0=dense/small-hole end, "
          f"{domain_size*1e6:.0f}=sparse/large-hole end)")
    centroids = []
    for m_idx in range(len(freqs)):
        disp = modes[:, m_idx]
        ux = disp[0::2]
        uy = disp[1::2]
        amp2 = ux ** 2 + uy ** 2
        if amp2.sum() < 1e-30:
            continue
        x_centroid = np.sum(nodes[:, 0] * amp2) / np.sum(amp2)
        centroids.append((freqs[m_idx], x_centroid))
        print(f"  {m_idx:6d} | {freqs[m_idx]/1e6:9.3f}  | {x_centroid*1e6:8.2f}")

    if len(centroids) >= 3:
        fs = np.array([c[0] for c in centroids])
        xs = np.array([c[1] for c in centroids])
        corr = np.corrcoef(fs, xs)[0, 1]
        print(f"\n  correlation(frequency, x-centroid) = {corr:.3f}")
        print("  (a real rainbow-trapping-like effect would show a clear NEGATIVE")
        print("   correlation -- higher-frequency modes concentrated toward the dense/")
        print("   small-hole end (x=0), lower-frequency modes toward the sparse end --")
        print("   matching the density sweep's finding that less coverage -> lower frequency.")
        print("   A correlation near 0 means modes are NOT spatially sorting by frequency.)")


def transient_strip_experiment(length=300e-6, width=60e-6, n_grid_x=120, n_grid_y=25,
                                n_fold=8, r_min=1.0e-6, r_max=4.0e-6,
                                window_radius=8.0, grid_index_range=2,
                                n_probes=6, cfl=0.3, n_steps=4000, seed=42):
    """Real vector-elastic transient analog of the earlier scalar FDTD
    experiment: a density-graded strip (small holes/high coverage at x=0 ->
    large holes/low coverage at x=length), clamped top/bottom, FREE at both
    x-ends so a wave actually propagates, explicit central-difference time
    integration, broadband pulse injected at x=0, displacement probed at
    several x positions. If rainbow trapping is real here, probes further
    along x should show their dominant frequency shift (and amplitude drop)
    relative to probes near the source, tracking the density gradient --
    the thing the earlier closed-eigenmode test was structurally unable to
    test."""
    print(f"\n=== transient vector-FEM strip: {length*1e6:.0f}um x {width*1e6:.0f}um, "
          f"density-graded, n_fold={n_fold} ===")

    pts = debruijn_quasicrystal_points(n_fold, window_radius, grid_index_range=grid_index_range,
                                        seed=seed)
    scale_x = length / (2 * window_radius) * 1.3
    scale_y = width / (2 * window_radius) * 1.3
    centers = np.stack([pts[:, 0] * scale_x + length / 2,
                         pts[:, 1] * scale_y + width / 2], axis=1)
    in_domain = (centers[:, 0] > 0) & (centers[:, 0] < length) & \
                (centers[:, 1] > 0) & (centers[:, 1] < width)
    centers = centers[in_domain]
    frac_x = np.clip(centers[:, 0] / length, 0, 1)
    radii = r_min + (r_max - r_min) * frac_x

    nodes, triangles, clamped = build_strip_mesh(length, width, n_grid_x, n_grid_y,
                                                  centers, radii)
    print(f"  mesh: {len(nodes)} nodes, {len(triangles)} elements, {len(centers)} holes, "
          f"{len(clamped)} clamped (top/bottom) nodes")
    if len(triangles) < 10:
        print("  WARNING: too few elements survived -- aborting.")
        return

    K, M, total_area = assemble(nodes, triangles)
    M_lump = np.array(M.sum(axis=1)).flatten()
    M_lump = np.maximum(M_lump, 1e-30)  # guard isolated/near-zero-mass dof
    print(f"  remaining membrane area: {100*total_area/(length*width):.1f}% of domain")

    n_dof = 2 * len(nodes)
    clamped_dofs = np.concatenate([2 * clamped, 2 * clamped + 1])

    c_p = np.sqrt(E / (RHO * (1 - NU ** 2)))  # plate longitudinal wave speed
    dx_min = min(length / (n_grid_x - 1), width / (n_grid_y - 1))
    dt = cfl * dx_min / c_p
    print(f"  wave speed ~{c_p:,.0f} m/s, dt={dt:.3e}s, n_steps={n_steps} "
          f"(simulated time {n_steps*dt*1e9:.1f} ns, "
          f"~{length/c_p*1e9:.1f} ns to cross the strip once)")

    # source: free nodes near x=0, transverse (uy) force pulse
    source_nodes = np.where((nodes[:, 0] < 2 * dx_min) &
                             (~np.isin(np.arange(len(nodes)), clamped)))[0]
    if len(source_nodes) == 0:
        print("  WARNING: no free nodes found near x=0 for source injection -- aborting.")
        return
    pulse_len = 30

    # absorbing sponge at the x=length end only (where we WANT the wave to
    # exit cleanly, not reflect and contaminate the probe readings with a
    # reverberant standing-wave pattern -- the actual bug found in the
    # unfixed version, where 4000 steps covered ~10 round trips down a
    # purely free-ended strip). Leave x=0 undamped: that's where the source
    # injects, and the wave should only ever travel away from it here, so
    # there's nothing arriving from that end to absorb.
    sponge_width_frac = 0.18
    sponge_x0 = length * (1 - sponge_width_frac)
    node_sponge = np.ones(len(nodes))
    in_sponge = nodes[:, 0] > sponge_x0
    frac = np.clip((nodes[in_sponge, 0] - sponge_x0) / (length - sponge_x0), 0, 1)
    node_sponge[in_sponge] = 1.0 - 0.08 * frac ** 2
    dof_sponge = np.repeat(node_sponge, 2)

    # probes: evenly spaced along x, nearest free node to mid-width at each.
    # Kept clear of the sponge zone (starts at 82% of length) -- a probe
    # inside the absorbing region would read artificially damped amplitude/
    # frequency, contaminating the measurement the sponge was added to protect.
    probe_xs = np.linspace(length * 0.1, length * 0.78, n_probes)
    probe_nodes = []
    for px in probe_xs:
        d2 = (nodes[:, 0] - px) ** 2 + (nodes[:, 1] - width / 2) ** 2
        probe_nodes.append(np.argmin(d2))
    probe_nodes = np.array(probe_nodes)

    u_prev = np.zeros(n_dof)
    u_curr = np.zeros(n_dof)
    probe_series = np.zeros((n_probes, n_steps))

    K_csr = K.tocsr()
    for t in range(n_steps):
        F = np.zeros(n_dof)
        if t < pulse_len:
            tt = t - pulse_len / 2
            pulse_val = (1 - 2 * (0.3 * tt) ** 2) * np.exp(-(0.3 * tt) ** 2)
            F[2 * source_nodes + 1] = pulse_val * 1e-3  # uy-direction force

        accel = (F - K_csr.dot(u_curr)) / M_lump
        u_next = (2 * u_curr - u_prev + dt ** 2 * accel) * dof_sponge
        u_next[clamped_dofs] = 0.0

        u_prev, u_curr = u_curr, u_next
        probe_series[:, t] = u_curr[2 * probe_nodes + 1]  # uy at each probe

    print(f"\n  probe x (um) | max |uy| | dominant frequency (MHz)")
    results = []
    for p_idx in range(n_probes):
        series = probe_series[p_idx]
        max_amp = np.max(np.abs(series))
        spec = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(n_steps, d=dt)
        band_mask = freqs > 1e5  # skip DC/near-DC
        if spec[band_mask].max() > 0:
            peak_freq = freqs[band_mask][np.argmax(spec[band_mask])]
        else:
            peak_freq = float("nan")
        results.append((probe_xs[p_idx], max_amp, peak_freq))
        print(f"  {probe_xs[p_idx]*1e6:11.1f} | {max_amp:.4e} | {peak_freq/1e6:8.2f}")

    xs = np.array([r[0] for r in results])
    amps = np.array([r[1] for r in results])
    fpeaks = np.array([r[2] for r in results])
    valid = ~np.isnan(fpeaks)
    if valid.sum() >= 3:
        corr = np.corrcoef(xs[valid], fpeaks[valid])[0, 1]
        print(f"\n  correlation(probe x, dominant frequency) = {corr:.3f}")
        print("  (rainbow trapping / density-graded filtering would show dominant")
        print("   frequency dropping as x increases (negative correlation), tracking")
        print("   the coverage gradient -- plus monotonically decaying amplitude.)")
    if amps[0] > 0:
        print(f"  amplitude ratio (last probe / first probe) = {amps[-1]/amps[0]:.4f}")


if __name__ == "__main__":
    results = {}
    for n_fold in (3, 6, 8, 12):
        results[n_fold] = run_case(n_fold)

    print("\n=== summary: lowest eigenfrequency by symmetry order ===")
    for n_fold, freqs in results.items():
        if freqs is not None and len(freqs) > 0:
            print(f"  n_fold={n_fold:2d}: f_min={freqs[0]:,.0f} Hz, f_max(of computed)={freqs[-1]:,.0f} Hz")

    density_sweep()
    graded_density_membrane()
    transient_strip_experiment()
