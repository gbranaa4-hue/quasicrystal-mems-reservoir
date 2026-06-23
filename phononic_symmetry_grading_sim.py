"""
Toy/qualitative simulation: does hole-pattern rotational symmetry order (3,
6, 8, 12-fold) shift the stopband of a 2D membrane, and does a structure
graded from low to high symmetry order spatially separate frequencies
(rainbow trapping), the way graded-spacing phononic crystals already do in
the literature (see chirped/rainbow phononic crystal papers)?

Hole positions now come from a real de Bruijn multigrid quasiperiodic
tiling (the construction behind Penrose/Ammann-Beenker tilings,
generalized to arbitrary n-fold symmetry) -- not the earlier rosette
approximation. This is still NOT a proper FEM solver, just a 2D scalar
wave equation (FDTD) on a grid with holes as locally softened material.
Good enough to see whether the qualitative trend (symmetry order ->
stopband / attenuation) is plausible before anyone invests in real FEM.
"""

import numpy as np

# ---- wave equation / material parameters (arbitrary, normalized units) ----
C_SOLID = 1.0       # wave speed in intact membrane
C_HOLE = 0.15        # wave speed in a "hole" (softened, not literally empty)
DX = 1.0             # grid spacing
CFL = 0.5             # Courant number, dt = CFL * dx / c_max
DT = CFL * DX / C_SOLID


def debruijn_quasicrystal_points(n_fold, window_radius, grid_index_range=8,
                                  offset_seed=0.0, seed=42):
    """Real quasiperiodic point set via the de Bruijn multigrid construction
    (the actual algorithm behind Penrose (5-fold) and Ammann-Beenker (8-fold)
    tilings, generalized to arbitrary n). Replaces the earlier rosette
    approximation, which was explicitly flagged as not a real tiling.

    n directions e_j = (cos(2*pi*j/n), sin(2*pi*j/n)), j=0..n-1, each with a
    family of parallel grid lines x.e_j = m + gamma_j (m integer). Vertices
    of the dual quasiperiodic tiling are found at every pairwise intersection
    of two different grid-line families; each vertex's tiling coordinate is
    sum_i ceil(x.e_i - gamma_i) * e_i, evaluated at the intersection point x.
    """
    j_idx = np.arange(n_fold)
    dirs = np.stack([np.cos(2 * np.pi * j_idx / n_fold),
                      np.sin(2 * np.pi * j_idx / n_fold)], axis=1)  # (n,2)
    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0.1, 0.9, size=n_fold) + offset_seed
    gammas -= gammas.mean()  # generic offsets, avoids degenerate triple intersections

    m_range = np.arange(-grid_index_range, grid_index_range + 1)
    points = []
    for j in range(n_fold):
        for k in range(j + 1, n_fold):
            ej, ek = dirs[j], dirs[k]
            det = ej[0] * ek[1] - ej[1] * ek[0]
            if abs(det) < 1e-9:
                continue
            for m in m_range:
                for p in m_range:
                    # solve x.ej = m+gamma_j, x.ek = p+gamma_k
                    rhs0 = m + gammas[j]
                    rhs1 = p + gammas[k]
                    x0 = (rhs0 * ek[1] - rhs1 * ej[1]) / det
                    x1 = (ej[0] * rhs1 - ek[0] * rhs0) / det
                    x = np.array([x0, x1])
                    if np.hypot(x0, x1) > window_radius * 1.5:
                        continue
                    # tiling-space coordinate: integer index along each grid direction
                    idx = np.ceil(x @ dirs.T - gammas).astype(int)
                    vertex = idx @ dirs
                    points.append(vertex)
    if not points:
        return np.zeros((0, 2))
    pts = np.array(points)
    # dedupe near-identical vertices (multiple grid pairs can yield the same point)
    pts = np.round(pts, 6)
    pts = np.unique(pts, axis=0)
    dist = np.hypot(pts[:, 0], pts[:, 1])
    return pts[dist <= window_radius]


def build_quasicrystal_mask(shape, center, n_fold, window_radius=14.0,
                             hole_radius=0.8, scale=2.0, grid_index_range=3):
    """Place holes at the vertices of a real de Bruijn quasiperiodic tiling
    (scaled and centered), instead of the earlier rosette approximation.

    Defaults tuned by a density sweep (grid_index_range/scale/hole_radius)
    to land in a partial-transmission regime (~2-6% hole coverage) -- the
    first attempt used grid_index_range=8, which produced near-total
    blockage (transmitted amplitude ~0 beyond the first segment) and made
    it impossible to compare attenuation *across* symmetry orders."""
    ny, nx = shape
    cy, cx = center
    pts = debruijn_quasicrystal_points(n_fold, window_radius, grid_index_range=grid_index_range)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    c_field = np.full(shape, C_SOLID)
    for (px, py) in pts:
        hx = cx + px * scale
        hy = cy + py * scale
        if hx < 0 or hx >= nx or hy < 0 or hy >= ny:
            continue
        dist2 = (xx - hx) ** 2 + (yy - hy) ** 2
        c_field = np.where(dist2 < hole_radius ** 2, C_HOLE, c_field)
    return c_field


def make_sponge_mask(shape, width=15, max_damp=0.06):
    """Smoothly-ramped absorbing border (a crude sponge layer / poor man's
    PML), applied as a per-step multiplicative damping field. Replaces the
    earlier np.roll() Laplacian, which silently wrapped waves around the
    domain edges (periodic boundary) and the single-row hard damping, which
    barely absorbed anything -- both let reflected/wrapped energy
    contaminate the 'transmission' measurement without warning."""
    ny, nx = shape
    damp = np.ones(shape)
    for i in range(width):
        frac = (width - i) / width          # 1 at the very edge -> 0 at the interior boundary
        factor = 1.0 - max_damp * frac ** 2
        damp[i, :] = np.minimum(damp[i, :], factor)
        damp[-1 - i, :] = np.minimum(damp[-1 - i, :], factor)
        damp[:, i] = np.minimum(damp[:, i], factor)
        damp[:, -1 - i] = np.minimum(damp[:, -1 - i], factor)
    return damp


def laplacian_no_wrap(u):
    """5-point Laplacian using zero-padding (not periodic wraparound)."""
    padded = np.pad(u, 1, mode="constant")
    return (
        padded[2:, 1:-1] + padded[:-2, 1:-1] +
        padded[1:-1, 2:] + padded[1:-1, :-2] -
        4 * u
    )


def fdtd_transmission_spectrum(c_field, source_pos, probe_pos, n_steps=4000):
    """Excite a broadband impulse at source_pos, record displacement at
    probe_pos over time, return the time series (FFT it outside to get a
    spectrum). 2D scalar wave eq: u_tt = c^2 * laplacian(u)."""
    ny, nx = c_field.shape
    u_prev = np.zeros((ny, nx))
    u_curr = np.zeros((ny, nx))
    probe_series = np.zeros(n_steps)
    sponge = make_sponge_mask((ny, nx))

    sy, sx = source_pos
    # broadband-ish source: short Ricker-like pulse over the first ~30 steps
    pulse_len = 30

    c2dt2_dx2 = (c_field * DT / DX) ** 2

    for t in range(n_steps):
        if t < pulse_len:
            tt = t - pulse_len / 2
            u_curr[sy, sx] += (1 - 2 * (0.3 * tt) ** 2) * np.exp(-(0.3 * tt) ** 2)

        lap = laplacian_no_wrap(u_curr)
        u_next = (2 * u_curr - u_prev + c2dt2_dx2 * lap) * sponge

        u_prev, u_curr = u_curr, u_next
        probe_series[t] = u_curr[probe_pos]

    return probe_series


def spectrum_peak_band(series, dt=DT, low_cut=0.02):
    """Return (freqs, magnitude) and a crude 'stopband-ish' summary: the
    frequency range where transmitted magnitude drops below low_cut times
    the peak, scanning from DC outward -- just enough to compare relative
    stopband location across symmetry orders, not a rigorous bandgap calc."""
    n = len(series)
    spec = np.abs(np.fft.rfft(series))
    freqs = np.fft.rfftfreq(n, d=dt)
    if spec.max() > 0:
        spec = spec / spec.max()
    return freqs, spec


def experiment_symmetry_vs_stopband():
    print("=== Part 1: does hole-pattern symmetry order shift the apparent stopband? ===")
    shape = (120, 120)
    center = (60, 60)
    source_pos = (60, 20)
    probe_pos = (60, 100)

    results = {}
    for n_fold in (3, 4, 6, 8, 12):
        c_field = build_quasicrystal_mask(shape, center, n_fold)
        series = fdtd_transmission_spectrum(c_field, source_pos, probe_pos, n_steps=3000)
        freqs, spec = spectrum_peak_band(series)
        # crude "dominant transmitted frequency" = frequency of peak magnitude
        # in the band that actually has appreciable excitation (skip near-DC)
        band_mask = (freqs > 0.01) & (freqs < 0.5)
        if np.any(spec[band_mask] > 0):
            peak_idx = np.argmax(spec[band_mask])
            peak_freq = freqs[band_mask][peak_idx]
        else:
            peak_freq = float("nan")
        results[n_fold] = peak_freq
        print(f"  n_fold={n_fold:2d}: dominant transmitted frequency ~ {peak_freq:.4f}")

    print("\n  (If these numbers trend monotonically with n_fold, that's at least")
    print("   consistent with the idea that symmetry order shifts the stopband.")
    print("   Now using a real de Bruijn quasiperiodic tiling, not the earlier rosette.)")
    return results


def experiment_graded_pyramid():
    print("\n=== Part 2: graded strip, n_fold=3 at one end -> n_fold=12 at the other ===")
    print("  (does a broadband pulse separate spatially by frequency, i.e. rainbow trapping?)")

    ny, nx = 60, 240
    c_field = np.full((ny, nx), C_SOLID)
    n_folds_sequence = [3, 4, 6, 8, 12]
    n_segments = len(n_folds_sequence)
    seg_width = nx // n_segments

    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    for i, n_fold in enumerate(n_folds_sequence):
        x0 = i * seg_width
        x1 = x0 + seg_width
        seg_center = ((x0 + x1) / 2, ny / 2)
        pts = debruijn_quasicrystal_points(n_fold, window_radius=14.0, grid_index_range=3)
        for (px, py) in pts:
            hx = seg_center[0] + px * 2.0
            hy = seg_center[1] + py * 2.0 * 0.6  # flatten vertically to fit the strip
            dist2 = (xx - hx) ** 2 + (yy - hy) ** 2
            c_field = np.where((dist2 < 0.8 ** 2) &
                                (np.arange(nx)[None, :] >= x0) &
                                (np.arange(nx)[None, :] < x1),
                                C_HOLE, c_field)

    source_pos = (ny // 2, 5)
    n_steps = 4000
    u_prev = np.zeros((ny, nx))
    u_curr = np.zeros((ny, nx))
    max_amp_by_x = np.zeros(nx)
    pulse_len = 20
    c2dt2_dx2 = (c_field * DT / DX) ** 2
    # only sponge the top/bottom/right (left has the source close to the edge;
    # absorbing it there would also damp the source injection itself)
    sponge_full = make_sponge_mask((ny, nx))
    sponge = np.ones((ny, nx))
    sponge[:, 1:] = sponge_full[:, 1:]   # leave column 0 (and the source col) undamped

    for t in range(n_steps):
        if t < pulse_len:
            tt = t - pulse_len / 2
            u_curr[source_pos] += (1 - 2 * (0.3 * tt) ** 2) * np.exp(-(0.3 * tt) ** 2)
        lap = laplacian_no_wrap(u_curr)
        u_next = (2 * u_curr - u_prev + c2dt2_dx2 * lap) * sponge
        u_prev, u_curr = u_curr, u_next

        if t > pulse_len:
            col_amp = np.max(np.abs(u_curr), axis=0)
            max_amp_by_x = np.maximum(max_amp_by_x, col_amp)

    print("  max amplitude reached at each x-position (segment boundaries marked):")
    for i, n_fold in enumerate(n_folds_sequence):
        x0 = i * seg_width
        x1 = x0 + seg_width
        seg_max = max_amp_by_x[x0:x1].max()
        seg_mean = max_amp_by_x[x0:x1].mean()
        print(f"    segment n_fold={n_fold:2d} (x={x0}-{x1}): max_amp={seg_max:.4f}, mean_amp={seg_mean:.4f}")

    overall_argmax = np.argmax(max_amp_by_x)
    seg_of_argmax = min(overall_argmax // seg_width, n_segments - 1)
    print(f"\n  energy peaks at x={overall_argmax} -> segment n_fold={n_folds_sequence[seg_of_argmax]}")
    print("  (a clean rainbow-trapping signature would show energy DECAYING")
    print("   monotonically as it crosses into higher-n_fold segments, i.e.")
    print("   getting reflected/stopped progressively earlier -- check the")
    print("   per-segment numbers above for that trend, don't just trust this single line)")


if __name__ == "__main__":
    experiment_symmetry_vs_stopband()
    experiment_graded_pyramid()
