#!/usr/bin/env python3
"""
PHYSICAL RESERVOIR COMPUTING -- rung 1 feasibility test.

QUESTION (narrow and honest): does a network of NONLINEAR oscillators show
the dynamics needed for reservoir computing -- i.e., (a) fading memory of
recent inputs, and (b) the ability to compute a NONLINEAR function of past
inputs that a linear system provably cannot?

WHY THIS IS THE RIGHT FIRST TEST: reservoir computing only works if the
PHYSICAL SYSTEM does the hard nonlinear transformation, and only a SIMPLE
LINEAR readout is trained on top. So the decisive experiment is a task that
REQUIRES nonlinearity -- here, reproducing a product of two delayed inputs,
y[n] = u[n-1]*u[n-2]. A linear readout on a LINEAR reservoir (or on the raw
input) literally cannot produce a product term. A NONLINEAR reservoir can.
We run all three and compare:
    (1) nonlinear oscillator reservoir   -> expected: works
    (2) same reservoir, nonlinearity OFF -> expected: fails (control)
    (3) no reservoir, linear readout on raw delayed input -> expected: fails

If only (1) works, the NONLINEAR PHYSICS is doing the computation -- the core
reservoir-computing property -- not the readout. That is the rung-1 result.

WHAT THIS IS NOT: this uses GENERIC nonlinear oscillators (Duffing type), not
a quasicrystal-specific model. It tests whether oscillator dynamics CAN do
reservoir computing at all -- a necessary precondition for a quasicrystal
MEMS reservoir, not a demonstration of one. A real quasicrystal version would
need the plate's actual (nonlinear, multi-mode, coupled) dynamics, which the
linear modal FEM in this project does not yet model. This is rung 1: the
principle, not the device.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)

# ---- reservoir configuration ----
# Tuned so the nonlinearity is actually ENGAGED: oscillator amplitudes reach
# ~O(1), where the quadratic/cubic terms are comparable to the linear restoring
# force. (A first attempt drove too weakly -- amplitudes ~0.05 -- so the
# nonlinear terms were ~0.1% of linear and the reservoir behaved linearly.)
N = 50              # number of oscillators
ZETA = 0.20         # damping ratio (sets fading-memory length)
BETA2 = 0.6         # QUADRATIC nonlinearity -> generates 2nd-order products (e.g. u[n-1]*u[n-2])
BETA3 = 1.0         # CUBIC (hardening) nonlinearity -> bounds the amplitude, keeps it stable
TAU_IN = 1.0        # input held constant for this long, then state is sampled
DT = 0.02           # integration sub-step (TAU_IN/DT = 50 sub-steps per input sample)
INPUT_AMP = 1.5     # input drawn uniform in [-INPUT_AMP, +INPUT_AMP] -- hard enough to wake the nonlinearity

L = 2200            # total input samples
WASHOUT = 200       # discard initial transient (reservoir must forget its start)
N_TRAIN = 1300
# remainder is test


def build_reservoir(seed=7):
    rng = np.random.default_rng(seed)
    omega = rng.uniform(0.5, 2.5, N)           # spread of natural frequencies (lower => larger response => nonlinearity engages)
    w_in = rng.uniform(-1.0, 1.0, N)           # input injection weights
    # sparse random coupling between oscillators (diffusive), kept small for stability
    C = rng.uniform(-1.0, 1.0, (N, N))
    mask = rng.random((N, N)) < 0.2            # ~20% connectivity
    C = C * mask
    np.fill_diagonal(C, 0.0)
    C *= 0.08                                  # scale down for stability
    return dict(omega=omega, zeta=ZETA, w_in=w_in, C=C)


def run_reservoir(params, u_series, beta2, beta3):
    """Drive the oscillator network with the input series; return the state
    matrix (one row per input sample): features = [x_1..x_N, v_1..v_N, bias].

    Nonlinearity: -beta2*x^2 (quadratic -> 2nd-order products) - beta3*x^3
    (cubic hardening -> stays bounded). beta2=beta3=0 gives the LINEAR control."""
    omega = params["omega"]; zeta = params["zeta"]
    w_in = params["w_in"]; C = params["C"]
    rowsum = C.sum(axis=1)                      # for diffusive coupling term
    x = np.zeros(N); v = np.zeros(N)
    n_sub = int(round(TAU_IN / DT))
    feats = np.empty((len(u_series), 2 * N + 1))
    for n, u in enumerate(u_series):
        for _ in range(n_sub):
            coupling = C @ x - rowsum * x       # sum_j C_ij (x_j - x_i)
            accel = (-(omega**2) * x - 2 * zeta * omega * v
                     - beta2 * x**2 - beta3 * x**3 + w_in * u + coupling)
            v = v + accel * DT                  # semi-implicit (symplectic) Euler
            x = x + v * DT
        feats[n, :N] = x
        feats[n, N:2*N] = v
        feats[n, -1] = 1.0                      # bias
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"reservoir blew up at sample {n} (reduce BETA/INPUT_AMP/DT)")
    return feats


def ridge_fit(X, Y, lam=1e-6):
    """Linear (ridge) readout: W = (X^T X + lam I)^-1 X^T Y."""
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def eval_task(states, target):
    """Train linear readout on train split, report R^2 on held-out test split."""
    Xtr = states[WASHOUT:WASHOUT + N_TRAIN]
    Ytr = target[WASHOUT:WASHOUT + N_TRAIN]
    Xte = states[WASHOUT + N_TRAIN:]
    Yte = target[WASHOUT + N_TRAIN:]
    W = ridge_fit(Xtr, Ytr[:, None])
    pred = (Xte @ W)[:, 0]
    return r2_score(Yte, pred), pred, Yte


def main():
    print("=" * 70)
    print("PHYSICAL RESERVOIR COMPUTING -- rung 1")
    print(f"  {N} nonlinear coupled oscillators, linear readout")
    print("=" * 70)

    u = RNG.uniform(-INPUT_AMP, INPUT_AMP, L)

    params = build_reservoir()
    print("\nRunning reservoirs (nonlinear, and linear control)...")
    states_nl = run_reservoir(params, u, beta2=BETA2, beta3=BETA3)   # nonlinear reservoir
    states_lin = run_reservoir(params, u, beta2=0.0, beta3=0.0)      # SAME reservoir, nonlinearity OFF
    print(f"  (nonlinear reservoir state amplitude: mean|x| = {np.mean(np.abs(states_nl[:, :N])):.3f} "
          f"-- needs to be ~O(1) for the nonlinearity to be engaged)")

    # raw-input baseline: linear readout on a delay line of the raw input
    maxd = 8
    raw = np.zeros((L, maxd + 2))
    for d in range(maxd + 1):
        raw[d:, d] = u[:L - d]
    raw[:, -1] = 1.0

    # ---------- TEST A: nonlinear task  y[n] = u[n-1]*u[n-2] ----------
    print("\n" + "-" * 70)
    print("TEST A -- nonlinear task:  y[n] = u[n-1] * u[n-2]")
    print("  (requires BOTH memory AND nonlinearity; a linear system cannot do it)")
    print("-" * 70)
    yA = np.zeros(L)
    yA[2:] = u[1:L-1] * u[0:L-2]
    r2_nl, pred_nl, yte = eval_task(states_nl, yA)
    r2_lin, pred_lin, _ = eval_task(states_lin, yA)
    r2_raw, _, _ = eval_task(raw, yA)
    print(f"  (1) NONLINEAR reservoir         R^2 = {r2_nl:6.3f}")
    print(f"  (2) LINEAR reservoir (control)  R^2 = {r2_lin:6.3f}")
    print(f"  (3) no reservoir, raw input     R^2 = {r2_raw:6.3f}")
    verdict = ("PASS -- only the nonlinear physics solves it"
               if (r2_nl > 0.6 and r2_lin < 0.3 and r2_raw < 0.3)
               else "inconclusive -- see numbers / tune params")
    print(f"\n  VERDICT (nonlinear computation): {verdict}")

    # ---------- TEST B: fading memory capacity ----------
    print("\n" + "-" * 70)
    print("TEST B -- short-term memory capacity (fading memory)")
    print("  reproduce u[n-k] for k=1..15; sum of test R^2 = memory capacity")
    print("-" * 70)
    Ks = range(1, 16)
    mc_curve = []
    for k in Ks:
        yk = np.zeros(L)
        yk[k:] = u[:L - k]
        r2k, _, _ = eval_task(states_nl, yk)
        mc_curve.append(max(0.0, r2k))
    MC = sum(mc_curve)
    print(f"  memory capacity (sum of R^2 over delays 1..15) = {MC:.2f}")
    print(f"  per-delay R^2: " + " ".join(f"{v:.2f}" for v in mc_curve))
    print("  (nonzero, decaying-with-delay capacity = fading memory present)")

    # ---------- plots ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    seg = slice(0, 120)
    ax1.plot(yte[seg], "k-", lw=2, label="target  u[n-1]*u[n-2]")
    ax1.plot(pred_nl[seg], "-", color="#2E5E8C", lw=1.5, label=f"nonlinear reservoir (R^2={r2_nl:.2f})")
    ax1.plot(pred_lin[seg], "--", color="#C0392B", lw=1.2, label=f"linear reservoir (R^2={r2_lin:.2f})")
    ax1.set_title("Nonlinear task: only the nonlinear reservoir tracks it")
    ax1.set_xlabel("test timestep"); ax1.set_ylabel("output")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.bar(list(Ks), mc_curve, color="#2E5E8C")
    ax2.set_title(f"Fading memory: capacity = {MC:.2f}")
    ax2.set_xlabel("delay k (how many steps back)"); ax2.set_ylabel("reconstruction R^2 of u[n-k]")
    ax2.set_ylim(0, 1.05); ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Physical reservoir computing -- rung 1 (GENERIC nonlinear oscillators, "
                 "not a quasicrystal device)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reservoir_rung1_results.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")

    # ---------- honest verdict ----------
    print("\n" + "=" * 70)
    print("HONEST VERDICT (rung 1)")
    print("=" * 70)
    if r2_nl > 0.6 and r2_lin < 0.3:
        print("  The nonlinear oscillator network DOES show reservoir-computing dynamics:")
        print("  it computes a nonlinear function of past inputs that the SAME network")
        print("  with nonlinearity switched off cannot -- so the PHYSICS, not the readout,")
        print("  is doing the work. It also shows fading memory (Test B). Rung 1: cleared.")
    else:
        print("  Mixed/negative result -- the configured dynamics didn't cleanly separate")
        print("  from the linear control. That's still an honest finding (needs tuning).")
    print("  NOT shown: that a QUASICRYSTAL MEMS plate specifically does this. This is")
    print("  generic oscillators -- a precondition test. The quasicrystal version needs")
    print("  the plate's real nonlinear multi-mode dynamics (a future rung), not the")
    print("  linear modal FEM used elsewhere in this project.")
    print("=" * 70)


if __name__ == "__main__":
    main()
