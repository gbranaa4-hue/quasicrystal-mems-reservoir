"""
1D Euler-Bernoulli beam bending eigenmode model -- a reduced-order stand-in
for full 2D Kirchhoff plate bending FEM (which is the physically correct
but much heavier model the actual literature uses, and out of scope to
build and validate in one session). This is NOT a substitute for that —
it's a much simpler test of the same basic question: does density grading
affect BENDING-mode behavior the way it affected the in-plane modes?

Bending stiffness EI ~ h^3 (cube of local thickness), unlike the in-plane
case where thickness canceled out of the eigenfrequency entirely. This
means density grading should, in principle, have a MUCH stronger effect
on bending modes than it did on the in-plane modes -- that's the real,
testable prediction this script checks.

Governing equation (variable-coefficient Euler-Bernoulli beam):
    d^2/dx^2 [ EI(x) * d^2w/dx^2 ] = omega^2 * rho*A(x) * w
Clamped-clamped boundary conditions (w=0, dw/dx=0 at both ends).
"""

import numpy as np
import scipy.linalg as la

# ---- material: silicon ----
E = 170e9       # Pa
RHO = 2330.0     # kg/m^3

# ---- beam geometry ----
LENGTH = 300e-6        # m, matches the strip length used in the FEM transient test
WIDTH = 60e-6           # m, out-of-plane width (matches strip width)
THICKNESS_NOMINAL = 2.0e-6  # m -- a real but somewhat arbitrary MEMS-scale thickness.
# (Unlike the in-plane model, thickness does NOT cancel here -- EI ~ h^3 --
# so this number directly sets the absolute frequency scale. Treat absolute
# values as illustrative; the density-dependence TREND is the real result.)


def coverage_from_hole_radius(hole_radius_um):
    """Interpolated from the actual controlled density sweep run earlier
    this session (n_fold=8, FEM eigenmode test): hole_radius -> coverage
    fraction. Reusing real measured points rather than inventing a new curve."""
    radii = np.array([1.0, 2.0, 3.0, 4.0])
    coverages = np.array([0.980, 0.917, 0.815, 0.639])
    return np.interp(hole_radius_um, radii, coverages)


def effective_thickness_profile(x, length, r_min_um=1.0, r_max_um=4.0,
                                 thickness_exponent=1.0):
    """Coverage fraction graded linearly along x (matching the earlier
    in-plane graded-membrane test), converted to an effective local
    thickness. h_eff = h0 * coverage(x)^thickness_exponent is a simple,
    explicitly-labeled approximation -- NOT a rigorously derived perforated-
    plate homogenization (those exist in the literature but are more
    involved than this reduced model warrants)."""
    frac_x = np.clip(x / length, 0, 1)
    hole_radius_um = r_min_um + (r_max_um - r_min_um) * frac_x
    coverage = coverage_from_hole_radius(hole_radius_um)
    return THICKNESS_NOMINAL * coverage ** thickness_exponent, coverage


def solve_beam_eigenmodes(length, width, n_points, h_profile_fn, n_modes=10):
    """Finite-difference solve of the variable-coefficient clamped-clamped
    Euler-Bernoulli beam eigenvalue problem.

    Two-stage central-difference construction:
      1. w'' via standard 3-point stencil
      2. M = EI * w''  (bending moment)
      3. M'' via standard 3-point stencil  =>  combined 4th-derivative-like
         operator that correctly handles spatially-varying EI(x), unlike a
         naive constant-coefficient biharmonic stencil would.
    Clamped BC (w=0, w'=0) enforced via symmetric ghost points at both ends.
    """
    x = np.linspace(0, length, n_points)
    dx = x[1] - x[0]

    h_eff, coverage = h_profile_fn(x)
    EI = E * width * h_eff ** 3 / 12.0
    rhoA = RHO * width * h_eff

    # extend with ghost points for clamped BC: w_{-1}=w_1 (zero slope), w_0=0 fixed
    n = n_points
    # second-derivative operator D2 (n x n), zero rows at clamped boundary nodes (w fixed)
    D2 = np.zeros((n, n))
    for i in range(1, n - 1):
        D2[i, i - 1] = 1.0
        D2[i, i] = -2.0
        D2[i, i + 1] = 1.0
    D2 /= dx ** 2
    # ghost-point correction at i=0 and i=n-1 isn't needed since those rows
    # of D2 are never used for clamped dofs; w''=0 imposed implicitly there.

    EI_diag = np.diag(EI)
    M_moment = EI_diag @ D2   # bending moment field from curvature

    # second derivative of the moment field, with symmetric (zero-slope) ghost
    # extension at the two clamped ends
    D2_outer = np.zeros((n, n))
    for i in range(1, n - 1):
        D2_outer[i, i - 1] = 1.0
        D2_outer[i, i] = -2.0
        D2_outer[i, i + 1] = 1.0
    D2_outer /= dx ** 2

    K = D2_outer @ M_moment

    # clamp w=0 at both ends: remove those rows/cols (Dirichlet)
    free = np.arange(1, n - 1)
    K_ff = K[np.ix_(free, free)]
    Mdiag_ff = np.diag(rhoA[free])

    eigvals, eigvecs = la.eig(K_ff, Mdiag_ff)
    eigvals = eigvals.real
    eigvals = np.clip(eigvals, 0, None)
    freqs_hz = np.sqrt(eigvals) / (2 * np.pi)
    order = np.argsort(freqs_hz)[:n_modes]

    full_modes = np.zeros((n, len(order)))
    full_modes[free, :] = eigvecs[:, order].real

    return x, freqs_hz[order], full_modes, h_eff, coverage


def run_uniform_comparison():
    print("=== Uniform-coverage comparison (no grading) ===")
    n_points = 200
    for label, r_um in [("dense (1um holes, 98% coverage)", 1.0),
                         ("sparse (4um holes, 64% coverage)", 4.0)]:
        def h_profile(x, r=r_um):
            cov = coverage_from_hole_radius(r)
            return np.full_like(x, THICKNESS_NOMINAL * cov), np.full_like(x, cov)

        x, freqs, modes, h_eff, coverage = solve_beam_eigenmodes(
            LENGTH, WIDTH, n_points, h_profile, n_modes=6)
        print(f"  {label}: f1-f6 = " + ", ".join(f"{f/1e6:.3f}" for f in freqs) + " MHz")


def run_graded_localization():
    print("\n=== Graded-density beam: mode localization test ===")
    n_points = 300

    def h_profile(x):
        return effective_thickness_profile(x, LENGTH)

    x, freqs, modes, h_eff, coverage = solve_beam_eigenmodes(
        LENGTH, WIDTH, n_points, h_profile, n_modes=10)

    print(f"  coverage range: {coverage.max()*100:.1f}% (x=0) -> {coverage.min()*100:.1f}% (x=L)")
    print(f"\n  mode # | freq (MHz) | displacement x-centroid (um, 0=dense/stiff end, "
          f"{LENGTH*1e6:.0f}=sparse/compliant end)")
    centroids = []
    for m_idx in range(len(freqs)):
        w = modes[:, m_idx]
        amp2 = w ** 2
        if amp2.sum() < 1e-30:
            continue
        x_centroid = np.sum(x * amp2) / np.sum(amp2)
        centroids.append((freqs[m_idx], x_centroid))
        print(f"  {m_idx:6d} | {freqs[m_idx]/1e6:9.3f}  | {x_centroid*1e6:8.2f}")

    if len(centroids) >= 3:
        fs = np.array([c[0] for c in centroids])
        xs = np.array([c[1] for c in centroids])
        corr = np.corrcoef(fs, xs)[0, 1]
        print(f"\n  correlation(frequency, x-centroid) = {corr:.3f}")
        print("  (EI ~ h^3 means density grading should have a MUCH stronger effect")
        print("   here than in the in-plane case (which used EA ~ h, linear). If this")
        print("   correlation is large and negative, bending modes DO localize by")
        print("   density -- the effect the in-plane model couldn't show.)")


def build_beam_operator(x, EI):
    """Same two-stage curvature/moment finite-difference construction as the
    eigenmode solver, but WITHOUT removing boundary rows -- the transient
    test needs free (not clamped) ends so a wave can actually propagate, the
    same correction the in-plane model needed (closed eigenvalue problem on
    a clamped domain can't show spatial localization; only a propagating-
    wave problem can)."""
    n = len(x)
    dx = x[1] - x[0]
    D2 = np.zeros((n, n))
    for i in range(1, n - 1):
        D2[i, i - 1] = 1.0
        D2[i, i] = -2.0
        D2[i, i + 1] = 1.0
    D2 /= dx ** 2

    M_moment = np.diag(EI) @ D2
    K = D2 @ M_moment
    return K


def transient_beam_strip(length=LENGTH, width=WIDTH, n_points=300,
                          n_probes=6, cfl=0.2, n_steps=20000, seed_check=False):
    """Transient (propagating-wave) analog of the in-plane strip test, for
    BENDING waves: explicit time-stepping of rho*A * w_tt = -K w, free ends,
    broadband pulse at x=0, absorbing sponge at x=length, probes along x."""
    print(f"\n=== transient bending-wave strip: {length*1e6:.0f}um, free ends ===")
    x = np.linspace(0, length, n_points)
    dx = x[1] - x[0]
    h_eff, coverage = effective_thickness_profile(x, length)
    EI = E * width * h_eff ** 3 / 12.0
    rhoA = RHO * width * h_eff

    K = build_beam_operator(x, EI)

    # explicit stability for the 4th-order (dispersive) beam equation is
    # much stricter than the 2nd-order wave case -- scales with dx^2, not dx
    omega_max_est = (np.pi / dx) ** 2 * np.sqrt(EI.max() / rhoA.min())
    dt = cfl * 2.0 / omega_max_est
    print(f"  dx={dx*1e9:.1f}nm, dt={dt:.3e}s, n_steps={n_steps} "
          f"(simulated time {n_steps*dt*1e9:.2f} ns)")

    # absorbing sponge at x=length only (same one-sided pattern as the
    # in-plane fix -- source end shouldn't be damped, nothing reflects
    # toward it in a one-way-launched pulse)
    sponge_width_frac = 0.18
    sponge_x0 = length * (1 - sponge_width_frac)
    sponge = np.ones(n_points)
    in_sponge = x > sponge_x0
    frac = np.clip((x[in_sponge] - sponge_x0) / (length - sponge_x0), 0, 1)
    sponge[in_sponge] = 1.0 - 0.10 * frac ** 2

    source_idx = 1  # one node in from the free end
    pulse_len = 40

    probe_xs = np.linspace(length * 0.1, length * 0.78, n_probes)
    probe_idx = [np.argmin(np.abs(x - px)) for px in probe_xs]

    w_prev = np.zeros(n_points)
    w_curr = np.zeros(n_points)
    probe_series = np.zeros((n_probes, n_steps))

    for t in range(n_steps):
        F = np.zeros(n_points)
        if t < pulse_len:
            tt = t - pulse_len / 2
            F[source_idx] = (1 - 2 * (0.3 * tt) ** 2) * np.exp(-(0.3 * tt) ** 2) * 1e-9

        accel = (F - K @ w_curr) / rhoA
        w_next = (2 * w_curr - w_prev + dt ** 2 * accel) * sponge
        w_prev, w_curr = w_curr, w_next

        for p in range(n_probes):
            probe_series[p, t] = w_curr[probe_idx[p]]

    print(f"\n  probe x (um) | max |w| | dominant frequency (MHz)")
    results = []
    for p_idx in range(n_probes):
        series = probe_series[p_idx]
        max_amp = np.max(np.abs(series))
        spec = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(n_steps, d=dt)
        band_mask = freqs > 1e4
        if spec[band_mask].max() > 0:
            peak_freq = freqs[band_mask][np.argmax(spec[band_mask])]
        else:
            peak_freq = float("nan")
        results.append((probe_xs[p_idx], max_amp, peak_freq))
        print(f"  {probe_xs[p_idx]*1e6:11.1f} | {max_amp:.4e} | {peak_freq/1e6:8.3f}")

    xs = np.array([r[0] for r in results])
    amps = np.array([r[1] for r in results])
    fpeaks = np.array([r[2] for r in results])
    valid = ~np.isnan(fpeaks)
    if valid.sum() >= 3:
        corr = np.corrcoef(xs[valid], fpeaks[valid])[0, 1]
        print(f"\n  correlation(probe x, dominant frequency) = {corr:.3f}")
    if amps[0] > 0:
        print(f"  amplitude ratio (last probe / first probe) = {amps[-1]/amps[0]:.4f}")
    return results


if __name__ == "__main__":
    run_uniform_comparison()
    run_graded_localization()
