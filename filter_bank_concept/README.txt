================================================================
Quasicrystal Filter-Bank Concept -- behavioral simulation
================================================================

WHAT THIS IS
  A proof-of-concept that wires the REAL finite-element center
  frequencies from the MEMS paper into a bank of bandpass filters,
  and plots the frequency response -- the "filter bank" idea, made
  concrete and runnable.

  Each channel is a separate quasicrystal resonator tuned to a
  different frequency BY HOLE COVERAGE -- the exact knob the paper
  found to be dominant. So the bank is built on the paper's own
  central result.

FILES
  filter_bank_sim.py        The basic concept. Computes real FEM center
                            frequencies (live), applies a behavioral
                            bandpass model, prints a channel table,
                            and saves the response plot.
  filter_bank_response.png  Output: 5-channel filter-bank response.

  filter_bank_realism.py    The "does it work?" test. Derives Q from a
                            real loss model (thermoelastic damping),
                            then MEASURES adjacent-channel isolation vs.
                            Q and runs a two-tone selection demo.
  filter_bank_isolation_vs_Q.png  Output: isolation vs. Q.

  KEY RESULT (filter_bank_realism.py): the channel selection works for
  Q >= ~700; a conservative real MEMS Q (1e3) clears that, and Li et
  al.'s measured 1e7 clears it hugely. Thermoelastic Q came out ~5e8
  (not the bottleneck), so Q is not the binding constraint in any
  realistic regime. The *function* is sound. This is NOT a
  fabricated-device guarantee -- anchor-loss Q, transduction/insertion
  loss, inter-resonator coupling, and real fabrication are still not
  modeled.

RUN IT
  python filter_bank_sim.py
  (Needs Python 3 + numpy + scipy + matplotlib. It imports the FEM from
   ../plate_bending_review/fem_plate_bending_homogenized.py, so keep
   this folder next to that one.)

----------------------------------------------------------------
WHAT IS REAL vs. WHAT IS ASSUMED  (read before quoting any number)
----------------------------------------------------------------
REAL (from the same FEM as every result in the paper):
  - The CENTER FREQUENCY of each channel.
  - The fact that coverage tunes those frequencies (the paper's result).

ASSUMED / behavioral (NOT derived from the plate physics):
  - Q (sharpness / bandwidth). Set to 1000 in the script. The FEM is
    undamped and has no loss model, so it cannot predict Q. The paper
    explicitly does not compute Q. Change ASSUMED_Q to see how the
    bank's selectivity scales -- but it is an input, not a result.

NOT MODELED AT ALL (would be needed for a real performance prediction):
  - Energy loss mechanisms -> the actual Q.
  - Insertion loss, electrical impedance, electromechanical transduction.
  - Inter-resonator coupling (each channel is treated independently).

----------------------------------------------------------------
THE HONEST BOTTOM LINE
----------------------------------------------------------------
This answers: "what shape does a filter bank take, given these REAL
center frequencies and an ASSUMED Q?"  It is illustrative.

It does NOT answer: "what performance would a fabricated quasicrystal
filter actually achieve?"  That needs the loss/transduction/coupling
modeling listed above, plus real fabrication -- a separate, much larger
effort. Treat this as a concept demo, not a device prediction.
