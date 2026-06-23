================================================================
Physical Reservoir Computing -- a new ladder
================================================================

THE BIG QUESTION (top of the ladder, NOT answered yet):
  Can a quasicrystal-perforated MEMS plate act as a physical reservoir
  computer -- and does its aperiodic, rich mode structure make it a
  BETTER reservoir than ordinary periodic / random structures?

THE LADDER (each rung de-risks the next; honesty about where we are):

  RUNG 1  [DONE -- see reservoir_rung1.py]
    Do nonlinear oscillator networks do reservoir computing AT ALL,
    in our own code, with proper controls?
    RESULT: YES.
      - Nonlinear task y[n]=u[n-1]*u[n-2]:  R^2 = 0.71
        (linear-reservoir control: -0.05;  raw-input baseline: -0.01)
        => only the NONLINEAR PHYSICS solves it, not the readout.
      - Fading memory capacity: 10.45 (clean decaying curve).
    HONEST SCOPE: generic Duffing-type oscillators, NOT a quasicrystal.
    This reproduces a KNOWN principle (oscillator reservoirs work) and
    gives us a working, validated testbed. It is NOT novel by itself.

  RUNG 2 & 3  [DONE -- NEGATIVE result; see FINDINGS.txt]
    Race the REAL quasicrystal vs periodic mode SPECTRA (FEM frequencies,
    matched coverage, normalized band) as reservoirs.
    RESULT: no advantage. The quasicrystal spectrum is genuinely richer
    (3 vs 18 near-degenerate mode pairs) but COMPUTATIONALLY INERT --
    nonlinear-task R^2 and memory capacity tie within seed spread, at
    every reservoir size 6..40 (size sweep). The mode-frequency
    DISTRIBUTION does not matter for this kind of computing. A clean
    negative result from a pre-registered mechanism (not knob-fishing).
    Files: reservoir_rung2_3.py, reservoir_rung3_sizesweep.py

  RUNG 4  [DONE -- NEGATIVE result; see FINDINGS.txt]
    Used the plate's ACTUAL MODE SHAPES (FEM eigenvectors): point-drive
    input weight phi_i(x0) and a nonlinear coupling network from the real
    spatial overlap of mode shapes (modal projection of a pointwise
    nonlinearity) -- coupling set by physics, not random.
    RESULT: the pre-registered prediction HELD hugely -- the quasicrystal
    coupling network is ~2.4x denser (triple-overlap density 0.835 vs
    0.344, because periodic symmetry makes many overlaps vanish). And it
    was COMPUTATIONALLY INERT: nonlinear task and memory capacity both tie
    within noise. Two independent structural advantages (spectrum AND
    coupling), both inert.
    File: reservoir_rung4_modeshapes.py

  RUNG 5 / 5b / 5c  [DONE -- mechanism CONFIRMED + first real QC edge found]
    Dialed coupling 0->full (5), then chased down a surprise.
    - 5  : strong coupling -> plates TIE (coupling inert, confirmed); but
           UNCOUPLED, the quasicrystal wins ~1.6-1.9x -- first real QC edge.
    - 5b : that edge is NOT effective dimensionality (3.1 vs 3.2, equal).
    - 5c : it IS a QUADRATIC SYMMETRY SELECTION RULE. Periodic D4 symmetry
           zeroes the product-generating term c2_i=integral(phi_i^3) in 88%
           of modes; the quasicrystal keeps it alive (38% dead). Ablation:
           equalizing c2 erases the gap, equalizing frequencies does not.
           A MODEST, real, mechanism-level edge -- weak-coupling regime,
           even-order tasks only.
    Files: reservoir_rung5_saturation.py, reservoir_rung5b_why.py,
           reservoir_rung5c_mechanism.py

  RUNG 6  [DONE -- stress test PASSED; the finding holds up]
    Tried to break the rung-5c finding three ways:
    - TASK GENERALITY: QC wins ALL 5 even-order tasks (+1.4..+2.9x), TIES
      ALL 5 odd-order tasks -- a textbook even/odd dichotomy, the exact
      signature of the c2=integral(phi^3) selection rule.
    - GEOMETRY: edge persists across coverage (78/85/92%) and quasicrystal
      symmetry (5/7/8/12-fold) -- 100% of configs keep it.
    - PHYSICAL NONLINEARITY: survives an electrostatic-style even term.
    => Real symmetry selection rule, not an artifact. Modest, weakly-coupled,
       even-order regime. File: reservoir_rung6_stresstest.py

----------------------------------------------------------------
FILES
  reservoir_rung1.py            Run it. Builds the oscillator reservoir,
                                runs the nonlinear + memory tasks with
                                controls, prints results, saves the plot.
  reservoir_rung1_results.png   Left: nonlinear reservoir tracks the
                                product task, linear control doesn't.
                                Right: fading-memory capacity curve.

  Requirements: Python 3 + numpy + matplotlib.

----------------------------------------------------------------
THE HONEST BOTTOM LINE
  Rung 1: oscillator reservoirs compute (known principle, reproduced).
  Rungs 2-4: the quasicrystal's structural richness -- in BOTH its mode
  spectrum AND its mode-shape coupling network -- is real but COMPUTATIONALLY
  INERT (no advantage at any size or coupling strength).
  Rungs 5-6: the ONE real quasicrystal advantage is a QUADRATIC SYMMETRY
  SELECTION RULE. A periodic plate's D4 symmetry zeroes the product-
  generating even-order nonlinearity (c2=integral(phi^3)) in ~88% of its
  modes; the quasicrystal's broken symmetry keeps it alive. This gives a
  modest, robust edge -- stress-tested across every even-order task (and
  correctly absent on odd ones), across coverage and quasicrystal symmetry,
  and under a physical electrostatic nonlinearity. Confined to the weakly-
  coupled, even-order regime, but real and mechanism-backed.
  Nothing here claims a fabricated quasicrystal device computes. See
  FINDINGS.txt for the full running record, results and all.
================================================================
