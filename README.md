# Quasicrystal MEMS & Physical Reservoir Computing

Research by **Gavin Branaa** (Independent Researcher) on perforated-plate MEMS
resonators and what their mode structure can — and cannot — compute.

There are two connected threads here, both built on the same finite-element
plate model.

## 1. The symmetry selection rule (the main result) → [`reservoir_computing/`](reservoir_computing/)

A point-group selection rule for **even-order computation** in modal physical
reservoirs. On a resonator with the square's D4 symmetry, the cubic mode
overlap `∫φ³` — the strength of the product-making nonlinearity — vanishes
exactly for every mode except the totally symmetric ones (7/8 are silenced).
Breaking the symmetry (aperiodic perforation, or just a lower-symmetry cavity)
revives them. Proved by group theory, confirmed on finite-element modes to
1 part in 10⁹, and shown to give a real but **bounded** reservoir-computing
advantage. See [`reservoir_computing/README.md`](reservoir_computing/README.md)
for the full ladder of experiments, the proof, and the paper.

**Preprint:** (Zenodo DOI — add link once published)

## 2. The MEMS resonator study → top-level files + [`plate_bending_review/`](plate_bending_review/)

The underlying device study: a quasicrystal-perforated micro-plate resonator,
its mode frequencies, and how hole coverage tunes them — plus the homogenized
finite-element solver (`plate_bending_review/fem_plate_bending_homogenized.py`)
that every result above imports. The [`filter_bank_concept/`](filter_bank_concept/)
folder explores using the real mode frequencies as an RF filter bank.

## Honest scope

All results are from **simulation** with finite-element mode shapes and
modeled nonlinearities. No device has been fabricated; claims about real
hardware performance are explicitly out of scope and flagged as such in the
individual write-ups.

## Running it

```bash
pip install numpy scipy matplotlib
# the FEM engine:
python plate_bending_review/fem_plate_bending_homogenized.py
# the headline result, confirmed on real modes:
python reservoir_computing/verify_selection_rule.py
```

## License

Code: MIT. Manuscripts/figures/written research: CC-BY-4.0. See `LICENSE`.
