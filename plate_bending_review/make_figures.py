"""Generate the mode-shape and convergence figures for the paper.

fig_convergence() uses the unperforated Leissa benchmark, which does not
depend on hole geometry and is unaffected by either the binary-removal vs.
homogenization correction or the de Bruijn construction bug fix -- it
continues to use fem_plate_bending_2d_v2 (equivalent for this purpose to
fem_plate_bending_homogenized.py, since with no holes phi=1 everywhere and
the two modules' element formulations are identical).

fig_mode_shapes() uses fem_plate_bending_homogenized.py (the corrected,
area-fraction-homogenized element, quadratic stiffness exponent) and the
corrected de Bruijn point-generation construction -- both required so this
figure represents the same methodology as every numerical result in
Section 3, rather than the superseded binary-removal geometry an earlier
version of this figure was generated from."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from fem_plate_bending_2d_v2 import (
    build_mesh, assemble, clamped_free_dofs, Lx, Ly, D, rho, h
)
from fem_plate_bending_homogenized import (
    debruijn_quasicrystal_points, element_coverage_fractions,
    assemble as assemble_homogenized,
)
from scipy.sparse.linalg import eigsh


def solve_modes_with_vectors(K, M, free, n_modes=6):
    Kf = K[np.ix_(free, free)]; Mf = M[np.ix_(free, free)]
    k = min(n_modes, len(free) - 2)
    sigma = max(Kf.diagonal().max() * 1e-4, 1e-20)
    vals, vecs = eigsh(Kf, k=k, M=Mf, sigma=sigma, which='LM', tol=1e-6, maxiter=50000)
    pos = vals > 1e-6 * np.abs(vals).max()
    vals = vals[pos]; vecs = vecs[:, pos]
    freqs = np.sqrt(np.abs(vals)) / (2*np.pi)
    order = np.argsort(freqs)
    return freqs[order], vecs[:, order]


def quads_to_tris(quads):
    tris = []
    for q in quads:
        n0, n1, n2, n3 = q
        tris += [[n0, n1, n2], [n0, n2, n3]]
    return np.array(tris)


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
        if cov_mid > target_cov: lo = mid
        else: hi = mid
    return mid, cov_mid


def fig_mode_shapes():
    """n=8, ~98% coverage, nx=28, quadratic exponent -- matching the
    methodology and a representative configuration from Section 3.4."""
    n_fold, seed, nx = 8, 42, 28
    r, cov = find_radius_for_coverage(n_fold, 98.0, nx, seed)
    holes = debruijn_quasicrystal_points(n_fold, Lx, Ly, offset_seed=seed)
    nodes, quads = build_mesh(Lx, Ly, nx, nx)
    phi = element_coverage_fractions(nodes, quads, holes, r, sub_n=12)
    cov = phi.mean() * 100

    K, M = assemble_homogenized(nodes, quads, phi=phi, stiffness_exponent=2.0)
    free = clamped_free_dofs(nodes)
    freqs, vecs = solve_modes_with_vectors(K, M, free, n_modes=6)

    nd = 3*len(nodes)
    full = np.zeros((nd, vecs.shape[1]))
    full[free, :] = vecs
    w_all = full[0::3, :]
    tr_plot = quads_to_tris(quads)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle(f"2D Plate-Bending FEM (Mindlin SRI quad, area-fraction homogenized, "
                 f"quadratic exponent) Mode Shapes\n"
                 f"De Bruijn 8-fold quasicrystal (corrected construction), r={r*1e6:.2f}um holes, "
                 f"Silicon h=100nm, {Lx*1e6:.0f}x{Ly*1e6:.0f}um, coverage={cov:.1f}%",
                 fontsize=10)
    for idx, ax in enumerate(axes.flat):
        w = w_all[:len(nodes), idx]
        tri2 = mtri.Triangulation(nodes[:, 0]*1e6, nodes[:, 1]*1e6, tr_plot)
        cf = ax.tricontourf(tri2, w, levels=20, cmap='RdBu_r')
        ax.triplot(tri2, 'k-', alpha=0.06, lw=0.3)
        for hc in holes:
            ax.add_patch(plt.Circle((hc[0]*1e6, hc[1]*1e6), r*1e6, color='gray', alpha=0.5))
        ax.set_title(f"Mode {idx+1}: {freqs[idx]/1e6:.4f} MHz", fontsize=9)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig("figure_mode_shapes.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure_mode_shapes.png (n_fold={n_fold}, r={r*1e6:.3f}um, coverage={cov:.2f}%, "
          f"{len(holes)} holes)")


def fig_convergence():
    """Leissa benchmark convergence -- independently re-run here for the figure
    (matches the exact values already verified earlier this session)."""
    f_anal = 35.99 / (2*np.pi*Lx**2) * np.sqrt(D/(rho*h))
    nxs = [10, 16, 22, 30]
    ratios = []
    for nx in nxs:
        nodes, quads = build_mesh(Lx, Ly, nx, nx)
        K, M = assemble(nodes, quads)
        free = clamped_free_dofs(nodes)
        freqs, _ = solve_modes_with_vectors(K, M, free, n_modes=1)
        ratios.append(freqs[0]/f_anal)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(nxs, ratios, 'o-', color='#2E5E8C', linewidth=2, markersize=7)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Analytical (Leissa 1969)')
    ax.set_xlabel("Mesh resolution (nx)")
    ax.set_ylabel("FEM f$_1$ / Analytical f$_1$")
    ax.set_title("Element verification: convergence to the\nLeissa clamped-plate benchmark")
    ax.legend()
    ax.grid(alpha=0.3)
    for x, y in zip(nxs, ratios):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("figure_convergence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figure_convergence.png")


if __name__ == "__main__":
    fig_convergence()
    fig_mode_shapes()
