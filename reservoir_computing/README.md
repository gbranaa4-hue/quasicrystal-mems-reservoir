# Symmetry-Activated Even-Order Computation in Physical Reservoirs

**A point-group selection rule for physical reservoir computing — and an honest
map of where it does and doesn't help.**

Author: Gavin Branaa (Independent Researcher)

---

## The one-line result

On a resonator with the square's **D4 symmetry**, the integral `∫φ³` over each
mode — the strength of the even-order (product-making) nonlinearity — is forced
to **exactly zero** for every mode except the totally symmetric ones (7/8 of
modes are silenced). **Breaking the symmetry** — with an aperiodic hole pattern,
or simply a lower-symmetry cavity — switches them back on. Proved by group
theory, confirmed on real finite-element modes to **1 part in 10⁹**.

## What's actually claimed (and what isn't)

- **Proven, exactly:** the selection rule above. It's a theorem, not a curve fit.
- **Real but bounded:** on *shallow* even-order tasks in the weakly-coupled
  regime, a symmetry-broken substrate beats a symmetric one (cross-validated,
  p ≈ 10⁻¹³). The advantage tracks the *degree of symmetry breaking* — a
  **periodic but low-symmetry** plate captures ~half of it, so the principle is
  **symmetry, not aperiodicity**.
- **Disproven:** the "richer structure computes better" intuition — spectral and
  coupling-network richness are computationally **inert**.
- **NOT claimed:** any advantage on deep-memory tasks or NARMA-10 (walled off by
  the conserved memory–nonlinearity budget), and nothing about a fabricated
  device — this is simulation with a modeled nonlinearity throughout.

## How it was reached (the rungs)

Each script is a self-contained, controlled experiment. The honest running log
is in [`FINDINGS.txt`](FINDINGS.txt); the analytic proof is in
[`PROOF_selection_rule.txt`](PROOF_selection_rule.txt).

| step | question | script |
|---|---|---|
| 1 | do oscillator reservoirs compute at all? | `reservoir_rung1.py` |
| 2–3 | does the quasicrystal *spectrum* help? (no) | `reservoir_rung2_3.py`, `reservoir_rung3_sizesweep.py` |
| 4 | does the mode-shape *coupling* help? (no) | `reservoir_rung4_modeshapes.py` |
| 5 | where does the edge actually live? | `reservoir_rung5_saturation.py`, `_5b_why.py`, `_5c_mechanism.py` |
| 6 | does it generalize? (even/odd, geometry, nonlinearity) | `reservoir_rung6_stresstest.py` |
| 7 | NARMA-10 + the memory wall | `reservoir_rung7_narma.py`, `_7b_depth.py` |
| — | the theorem, verified mode-by-mode | `verify_selection_rule.py` |
| — | peer review / cross-validation | `peer_review_crossval.py` |
| — | symmetry from the *cavity*, not just holes | `cavity_symmetry.py`, `peer_review_cavity.py` |
| — | would an optomechanical experiment see it? | `full_sim_optomech.py` |

## Run it

```bash
pip install numpy scipy matplotlib
python reservoir_rung1.py          # the validated testbed
python verify_selection_rule.py    # the theorem, confirmed on FEM modes
python peer_review_crossval.py     # the statistics
```

The plate finite-element solver lives one folder up
(`../plate_bending_review/fem_plate_bending_homogenized.py`); the reservoir
scripts import it.

## Paper

`paper_draft.tex` + `paper_supplement.tex` (compile on Overleaf). Plain-language
explainer of the theorem: `Selection_Rule_Explained.docx`.

## License

Code and text released under CC-BY-4.0 — reuse freely with attribution.
