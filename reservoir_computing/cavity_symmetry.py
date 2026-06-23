#!/usr/bin/env python3
"""
CAVITY SHAPE AS A SYMMETRY KNOB -- can the CONTAINER break the symmetry?

Our edge comes from breaking the substrate's point symmetry, which we have so
far done with the HOLE pattern. The "put it in a cavity" idea predicts the
BOUNDARY should work too. We test it with an ordinary PERIODIC hole array in
cavities of decreasing point symmetry:

    square cavity        -> D4   -> ~7/8 = 87.5% of modes silenced (baseline)
    rectangular cavity   -> C2v  -> ~3/4 = 75%   silenced
    parallelogram cavity -> C2   -> ~1/2 = 50%   silenced
    (quasicrystal, square, for reference) -> ~38% silenced

(The silenced fraction follows the same selection rule int(phi^3)=0 unless the
mode is totally symmetric; only A1 survives, and A1 is 1/|point group| of the
1-D content.) PREDICTION: the even-order task R^2 rises monotonically as the
silenced fraction falls -- and PERIODIC plates, by cavity shape alone, climb
toward quasicrystal performance with no aperiodicity at all. That would prove
the cavity boundary is a genuine symmetry-breaking design knob, and reinforce
that the effect is SYMMETRY, not quasicrystallinity.

Odd-task control included (must stay flat: cavity shape should not help linear
tasks). REAL FEM throughout; modeled nonlinearity, as always.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

FEM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "plate_bending_review")
sys.path.insert(0, FEM_DIR)
from fem_plate_bending_homogenized import (  # noqa: E402
    Lx as L0, build_mesh, element_coverage_fractions, assemble,
    debruijn_quasicrystal_points,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_rung4_modeshapes import ridge_fit, r2, nearest_elem  # noqa: E402
from reservoir_rung6_stresstest import run_uncoupled  # noqa: E402

N_MODES = 40
TARGET_COV = 85.0
OMEGA_LO, OMEGA_HI = 0.5, 2.5
INPUT_AMP = 1.0
L = 1800
WASHOUT = 200
N_TRAIN = 1100
SEEDS = [0, 1]
_drng = np.random.default_rng(9)
DRIVE_FRACS = [tuple(p) for p in _drng.uniform(0.28, 0.72, size=(5, 2))]


def clamped_free_dofs_topo(nx, ny):
    """Clamp the TOPOLOGICAL boundary (grid edges), so it works for sheared
    (parallelogram) meshes where coordinate min/max no longer define the edge."""
    Nn = nx * ny; con = []
    for j in range(ny):
        for i in range(nx):
            if i in (0, nx - 1) or j in (0, ny - 1):
                n = j * nx + i; con += [3 * n, 3 * n + 1, 3 * n + 2]
    return np.setdiff1d(np.arange(3 * Nn), con)


def periodic_rect(n_x, n_y, LX, LY, frac=0.84):
    xs = np.linspace(LX * (1 - frac) / 2, LX * (1 + frac) / 2, n_x)
    ys = np.linspace(LY * (1 - frac) / 2, LY * (1 + frac) / 2, n_y)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def quad_areas(coords):
    x = coords[:, :, 0]; y = coords[:, :, 1]
    x2 = np.roll(x, -1, axis=1); y2 = np.roll(y, -1, axis=1)
    return 0.5 * np.abs((x * y2 - x2 * y).sum(axis=1))


def coverage_radius(holes, LX, LY, nx, ny, shear, target=TARGET_COV):
    nodes, quads = build_mesh(LX, LY, nx, ny)
    if shear:
        nodes = nodes.copy(); nodes[:, 0] += shear * nodes[:, 1]
    lo, hi = 0.2e-6, 20e-6
    mid = hi
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        cov = element_coverage_fractions(nodes, quads, holes, mid, sub_n=10).mean() * 100
        if abs(cov - target) < 0.4:
            return mid
        if cov > target:
            lo = mid
        else:
            hi = mid
    return mid


def cavity_modes(holes_ref, LX, LY, nx, ny, shear):
    """Real FEM modes in a (possibly sheared, possibly rectangular) cavity.
    Returns freqs, element-center mode field (unit-RMS), area weights, centers,
    self-coefficients c2,c3, and the silenced fraction."""
    holes = holes_ref.copy()
    if shear:
        holes[:, 0] = holes[:, 0] + shear * holes[:, 1]
    radius = coverage_radius(holes, LX, LY, nx, ny, shear)
    nodes, quads = build_mesh(LX, LY, nx, ny)
    if shear:
        nodes = nodes.copy(); nodes[:, 0] += shear * nodes[:, 1]
    phi_cov = element_coverage_fractions(nodes, quads, holes, radius, sub_n=12)
    K, M = assemble(nodes, quads, phi=phi_cov, stiffness_exponent=2.0)
    free = clamped_free_dofs_topo(nx, ny)
    Kf = K[np.ix_(free, free)]; Mf = M[np.ix_(free, free)]
    k = min(N_MODES + 6, len(free) - 2)
    sigma = max(Kf.diagonal().max() * 1e-4, 1e-20)
    vals, vecs = eigsh(Kf, k=k, M=Mf, sigma=sigma, which="LM", tol=1e-6, maxiter=50000)
    keep = vals > 1e-6 * np.abs(vals).max()
    vals, vecs = vals[keep], vecs[:, keep]
    order = np.argsort(vals)
    vals, vecs = vals[order][:N_MODES], vecs[:, order][:, :N_MODES]
    freqs = np.sqrt(np.abs(vals)) / (2 * np.pi)

    Nn = len(nodes)
    coords = nodes[quads]; centers = coords.mean(axis=1)
    aw = quad_areas(coords); aw = aw / aw.sum()
    w_nodes = np.zeros((N_MODES, Nn))
    for m in range(N_MODES):
        full = np.zeros(3 * Nn); full[free] = vecs[:, m]
        w_nodes[m] = full[0::3]
    Wc = w_nodes[:, quads].mean(axis=2)
    rms = np.sqrt((aw[None, :] * Wc**2).sum(axis=1))
    Wc = Wc / rms[:, None]
    c2 = (aw[None, :] * Wc**3).sum(axis=1)
    c3 = (aw[None, :] * Wc**4).sum(axis=1)
    dead = float(np.mean(np.abs(c2) < 0.10 * np.abs(c2).max()))
    return freqs, Wc, aw, centers, c2, c3, dead


def eval_split(states, target):
    Xtr, Ytr = states[WASHOUT:WASHOUT + N_TRAIN], target[WASHOUT:WASHOUT + N_TRAIN]
    Xte, Yte = states[WASHOUT + N_TRAIN:], target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None], lam=1e-6)
    return r2(Yte, (Xte @ W)[:, 0])


def even_odd(freqs, Wc, c2, c3, centers):
    omega = OMEGA_LO + (OMEGA_HI - OMEGA_LO) * (freqs - freqs.min()) / (freqs.max() - freqs.min())
    ev, od = [], []
    for s in SEEDS:
        rng = np.random.default_rng(100 + s)
        u = rng.uniform(-INPUT_AMP, INPUT_AMP, L)
        et = np.zeros(L); et[2:] = u[1:L - 1] * u[0:L - 2]    # even: u[n-1]*u[n-2]
        ot = np.zeros(L); ot[1:] = u[:L - 1]                  # odd : u[n-1]
        for fr in DRIVE_FRACS:
            e = nearest_elem(centers, fr); w_in = Wc[:, e].copy()
            st = run_uncoupled(omega, c2, c3, w_in, u)
            ev.append(eval_split(st, et)); od.append(eval_split(st, ot))
    return np.mean(ev), np.std(ev), np.mean(od), np.std(od)


def main():
    print("=" * 88)
    print("CAVITY SHAPE AS A SYMMETRY KNOB -- does the container break the symmetry?")
    print("=" * 88)

    nx0 = 24
    substrates = [
        ("square periodic (D4)",      periodic_rect(9, 9, L0, L0),            L0, L0,       nx0, nx0, 0.0),
        ("rect cavity 1.3 (C2v)",     periodic_rect(9, 12, L0, 1.3 * L0),     L0, 1.3 * L0, nx0, 31,  0.0),
        ("rect cavity 1.6 (C2v)",     periodic_rect(9, 14, L0, 1.6 * L0),     L0, 1.6 * L0, nx0, 38,  0.0),
        ("parallelogram (C2)",        periodic_rect(9, 9, L0, L0),            L0, L0,       nx0, nx0, 0.35),
        ("quasicrystal sq (ref)",     debruijn_quasicrystal_points(8, L0, L0, offset_seed=42), L0, L0, nx0, nx0, 0.0),
    ]

    print(f"\n{'cavity':<26}{'silenced':>10}{'even R^2':>16}{'odd R^2':>14}")
    print("-" * 70)
    rows = []
    for name, holes, LX, LY, nx, ny, shear in substrates:
        fr, Wc, aw, ctr, c2, c3, dead = cavity_modes(holes, LX, LY, nx, ny, shear)
        em, es, om, osd = even_odd(fr, Wc, c2, c3, ctr)
        rows.append((name, dead, em, es, om, osd))
        print(f"{name:<26}{dead:>10.2f}{em:>10.3f}+/-{es:<4.3f}{om:>9.3f}+/-{osd:<4.3f}")

    # ---- monotonicity check ----
    order = sorted(range(len(rows)), key=lambda i: -rows[i][1])  # by decreasing silenced
    evens = [rows[i][2] for i in order]
    mono = all(evens[i] <= evens[i + 1] + 1e-9 for i in range(len(evens) - 1))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    deads = np.array([r[1] for r in rows])
    em = np.array([r[2] for r in rows]); es = np.array([r[3] for r in rows])
    colors = ["#C0392B", "#E67E22", "#E67E22", "#27AE60", "#2E5E8C"]
    ax.errorbar(deads, em, yerr=es, fmt="o", ms=10, capsize=4, color="k", zorder=3, lw=0)
    for (name, d, e, s, *_), c in zip(rows, colors):
        ax.scatter([d], [e], s=120, color=c, zorder=4)
        ax.annotate(name, (d, e), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("symmetry-silenced fraction  (higher = more symmetric)")
    ax.set_ylabel("even-order task R$^2$")
    ax.set_title("The cavity shape alone tunes the even-order edge\n"
                 "(periodic plates reach toward the quasicrystal by container shape)")
    ax.invert_xaxis(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cavity_symmetry.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved {out}")

    # ---- verdict ----
    sq = next(r for r in rows if "square periodic" in r[0])
    qc = next(r for r in rows if "quasicrystal" in r[0])
    par = next(r for r in rows if "parallelogram" in r[0])
    odd_spread = max(r[4] for r in rows) - min(r[4] for r in rows)
    print("\n" + "=" * 88)
    print("VERDICT -- is the cavity a symmetry knob?")
    print("=" * 88)
    print(f"  even-order R^2 vs silenced fraction is monotonic: {mono}")
    print(f"  square periodic (D4):   silenced {sq[1]:.2f}, even R^2 {sq[2]:.3f}")
    print(f"  parallelogram periodic: silenced {par[1]:.2f}, even R^2 {par[2]:.3f}  "
          f"(periodic, NO quasicrystal)")
    print(f"  quasicrystal (ref):     silenced {qc[1]:.2f}, even R^2 {qc[2]:.3f}")
    print(f"  odd-task spread across all cavities: {odd_spread:.3f} (should be ~0)")
    captured = (par[2] - sq[2]) / (qc[2] - sq[2] + 1e-9)
    if mono and par[2] > sq[2] + 0.03 and odd_spread < 0.05:
        print(f"\n  => CONFIRMED. The cavity BOUNDARY is a genuine symmetry-breaking knob.")
        print(f"     A PERIODIC plate, by container shape alone (square->rect->parallelogram),")
        print(f"     climbs the same even-order curve and recovers {captured*100:.0f}% of the")
        print(f"     quasicrystal's edge with NO aperiodicity. The odd-task control stays")
        print(f"     flat, as the selection rule demands. This makes 'cavity engineering'")
        print(f"     a rigorous design knob, and reinforces: the effect is SYMMETRY.")
    else:
        print("\n  => Mixed -- inspect the table (mesh resolution / coverage approximation")
        print("     on the sheared cavity can blur the parallelogram point).")
    print("=" * 88)


if __name__ == "__main__":
    main()
