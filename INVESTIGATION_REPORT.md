# Investigation Report: Density-Graded Quasicrystal Phononic Membranes
## A Computational Study of Vibration Filtering and Broadband Absorption

**Date:** June 20, 2026
**Researcher:** Gavin Branaa
**Status:** Complete
**Key Finding:** Hole density, not symmetry order, controls in-plane bandgap location in phononic membranes

---

## Executive Summary

This investigation set out to test whether a density-graded quasicrystal membrane could produce frequency-position sorting ("rainbow trapping") in vector-elastic FEM. Through four rounds of debugging and a critical robustness check, the following conclusions were reached:

| Claim | Status | Confidence |
|-------|--------|------------|
| Hole density controls in-plane bandgap location | ✅ Validated | High (controlled sweep, monotonic, reproducible) |
| Symmetry order (3/6/8/12-fold) does NOT affect in-plane bandgaps at fixed density | ✅ Validated | High |
| Density-graded structures attenuate waves monotonically with distance | ✅ Validated | Moderate (robust across 5 trials, but a generic effect) |
| Density grading → frequency-position rainbow trapping | ❌ Not supported | Failed robustness check (correlation ranged -0.765 to +0.298 across seeds) |
| FEM solver/mesh pipeline is internally consistent | ✅ Validated | Moderate (one mesh-refinement check only, not a full convergence study) |

**Important scope note:** this is a small, single-author computational exploration (one custom 2D in-plane FEM script, no commercial solver cross-check, no fabrication), not a peer-reviewed or externally validated study. Confidence levels above are about internal consistency of this model, not about whether the findings hold in a real device.

---

## 1. Theoretical Background

### 1.1 Quasicrystal Phononic Membranes
Quasicrystals are aperiodic but ordered structures with rotational symmetries (e.g., 5-fold, 8-fold, 12-fold) impossible in conventional periodic crystals. Published literature (Yu et al., arXiv:2604.07379, "Quasicrystal Architected Nanomechanical Resonators via Data-Driven Design") reports a 12-fold quasicrystal nanomechanical resonator achieving Q≈10⁷ with force sensitivity of 26.4 aN/√Hz, motivating interest in quasicrystal phononic structures for MEMS sensing. That paper used a 3D plate/bending-mode model and a data-driven optimized hole pattern — this investigation does not reproduce that result; it tests a related but distinct, simpler question (see 1.3) with a much cruder model.

### 1.2 Rainbow Trapping
"Rainbow trapping" refers to the spatial separation of different frequency components of a broadband wave as it propagates through a graded structure, published in graded/chirped phononic and photonic crystals (e.g. Nature Scientific Reports srep40004; Scientific Reports s41598-020-75977-8 for 3D phononic rainbow trapping). Prior published work grades a single structural parameter (lattice spacing, hole size) within one fixed symmetry family. This investigation tested a related but different and unpublished idea: grading hole density specifically within a quasicrystalline (not periodic) hole arrangement.

### 1.3 Research Question
**Can a density-graded quasicrystal membrane produce frequency-position sorting (rainbow trapping) in a simplified vector-elastic FEM model?**

### 1.4 Revised Hypothesis (Post-Debugging)
The original hypothesis (symmetry order controls bandgap location) was tested first and not supported. Based on that result, the hypothesis was revised:
> "Hole density (coverage fraction), not symmetry order, controls bandgap location. A spatial density gradient should therefore produce frequency-dependent spatial separation."

This revised hypothesis was tested and partially supported (density→bandgap: yes; density grading→frequency sorting: no, see Section 3.3).

---

## 2. Computational Methodology

### 2.1 Model Architecture — as actually implemented

| Component | Specification |
|-----------|---------------|
| **Geometry** | 100 µm × 100 µm square membrane (eigenmode tests) / 300 µm × **60 µm** strip (transient test) |
| **Material** | **Silicon** — E = 170 GPa, ν = 0.28, ρ = 2330 kg/m³ (matches the silicon_die case used earlier in the same investigation) |
| **Thickness** | Not physically modeled. Set to an arbitrary placeholder (1.0, unitless) — it cancels algebraically in the in-plane eigenfrequency calculation (stiffness and mass both scale linearly with thickness). **No real thickness value (e.g. a MEMS-typical ~100nm) was used or validated.** This matters because thickness would strongly affect the out-of-plane bending modes this model does not include (Section 5, limitations). |
| **Hole Pattern** | De Bruijn multigrid quasiperiodic tiling (real construction, generalized to n=3,6,8,12-fold; not a true Penrose/Ammann-Beenker tiling beyond n=5,8 but the same underlying algorithm) |
| **Density Range** | 64% – 98% coverage fraction (controlled via hole radius, fixed hole positions) |
| **FEM Solver** | Custom Python FEM, linear constant-strain triangle (CST) elements, 2D plane-stress |
| **Physics modeled** | **In-plane elasticity only.** Out-of-plane bending/shear (the actual dominant mode family in thin membrane resonators) is NOT modeled. |
| **Time integration (transient case)** | Explicit central-difference scheme with lumped (row-sum) mass matrix. (Not a general Newmark-beta scheme — central difference is a special case of Newmark-beta, but no other Newmark parameters were implemented or tested.) |

### 2.2 Key Parameters

| Parameter | Value Range Actually Tested | Purpose |
|-----------|-------------|---------|
| Hole radius | 1.0 – 5.0 µm (5.0µm failed: mesh over-fragmented) | Control coverage fraction |
| Symmetry order (n_fold) | 3, 6, 8, 12 | Test symmetry dependence |
| Mesh size | 2,817 – 5,593 nodes across two resolutions tested | One refinement comparison, not a full convergence study |
| Sponge layer width | 18% of strip length (one end only) | Absorbing boundary for the free propagation end |
| Simulation time | 4,000–5,500 steps (~327–337 ns simulated) | Transient propagation |
| Random seeds tested | 42 (two mesh resolutions), 1, 7, 99 | Robustness check on tiling realization |

### 2.3 Validation Pipeline

```
1. Scalar Toy Model (2D FDTD, point-source approximation)
   - Found apparent rainbow-trapping signal -- later shown to be a
     periodic-boundary artifact (wave wrapping around the domain edge)
   - Fixed: non-periodic boundary + sponge absorption
   - Fixed: replaced crude "rosette" hole pattern with real de Bruijn tiling
   - Re-tested: rainbow-trapping-like signal reappeared with correct boundaries
     (still in the simplified scalar/point-source model, not vector elasticity)

2. Vector FEM, eigenmode analysis (square membrane)
   - First attempt: near-zero eigenfrequencies, singular matrix for dense
     patterns -- traced to disconnected mesh fragments from naive
     hole-cutting (delete-by-centroid on a structured grid)
   - Fixed: largest-connected-component selection (boundary-aware) +
     pendant-node pruning
   - Result: physically sane MHz-range eigenfrequencies
   - Controlled density sweep (n_fold=8 fixed, hole radius varied):
     bandgap location shifts down monotonically with decreasing coverage
   - Symmetry comparison (n_fold=3,6,8,12, matched ~85-91% coverage):
     gap location varies <6% across symmetry orders -- not a meaningful effect

3. Vector FEM, transient propagation (strip geometry)
   - Density-graded strip, broadband pulse at one end, free other end
   - First attempt: reflections off the unabsorbed far end contaminated
     probe readings (~10 round trips within the simulated window)
   - Fixed: one-sided absorbing sponge layer
   - Result (seed=42): correlation(x, dominant frequency) = -0.765,
     amplitude decaying monotonically -- looked like a real effect

4. Robustness check (the critical step)
   - Re-ran with mesh refinement: correlation dropped to -0.451
   - Re-ran with 3 different random tiling seeds: correlations of
     -0.293, +0.298, +0.156 -- sign flipped in 2 of 5 trials
   - Conclusion: the frequency-position sorting result does NOT replicate.
     The original -0.765 was a single-trial artifact, not a real effect.
   - What DID replicate: amplitude attenuation with distance (robust
     across all 5 trials, ratio 0.20-0.35) -- a much more generic,
     less novel finding than rainbow trapping
```

---

## 3. Results

### 3.1 Density-Controlled Bandgap Location (Eigenmode Sweep)

**Experiment:** Fixed symmetry order (n_fold=8), varied hole radius (hence coverage fraction), holding hole *positions* constant.

| Hole Radius (µm) | Coverage Fraction | f_min (MHz) | Gap Location (MHz) | Gap Width (MHz) |
|------------------|-------------------|-------------|-------------------|-----------------|
| 1.0 | 98.0% | 50.33 | 60.8 – 74.1 | 13.3 |
| 2.0 | 91.7% | 48.17 | 58.3 – 71.2 | 12.9 |
| 3.0 | 81.5% | 43.76 | 52.6 – 64.9 | 12.3 |
| 4.0 | 63.9% | 34.12 | 35.3 – 42.2 | 6.9 |
| 5.0 | — | **mesh failed** (over-fragmented, spurious near-zero modes) |

**Finding:** As coverage fraction decreases, both the fundamental frequency and gap location shift downward monotonically across all 4 valid data points. Clean, controlled, reproducible within this model.

---

### 3.2 Symmetry Order Comparison (Eigenmode)

**Experiment:** Fixed coverage fraction (~85-91%, not perfectly matched across cases), varied symmetry order.

| Symmetry Order (n_fold) | f_min (MHz) | Gap Location (MHz) | Gap Width (MHz) |
|-------------------------|-------------|-------------------|-----------------|
| 3 | 47.6 | 57.9 – 70.4 | 12.5 |
| 6 | 44.9 | 54.6 – 66.1 | 11.5 |
| 8 | 46.1 | 56.3 – 69.3 | 12.9 |
| 12 | 44.8 | 55.6 – 67.7 | 12.1 |

**Finding:** Gap location varies by roughly 6% across all four symmetry orders — small enough that, in this in-plane model, symmetry order is not a significant lever compared to density (Section 3.1's ~50% f_min range).

---

### 3.3 Density-Graded Transient Propagation — Result Retracted After Robustness Check

**Experiment:** 300µm × 60µm strip, density graded from dense (1µm holes) at x=0 to sparse (4µm holes) at x=length, broadband pulse injected at x=0, absorbing boundary at x=length.

**Initial result (seed=42, original mesh, 2,850 nodes):**

| Probe x (µm) | max \|uy\| | Dominant Freq (MHz) |
|--------------|-----------|-------------------|
| 30.0 | 1.43e-15 | 474.5 |
| 70.8 | 1.22e-15 | 249.1 |
| 111.6 | 8.29e-16 | 415.2 |
| 152.4 | 4.97e-16 | 376.7 |
| 193.2 | 3.02e-16 | 225.4 |
| 234.0 | 2.85e-16 | 74.2 |

Correlation(x, frequency) = **-0.765**, amplitude ratio (last/first) = **0.199**. This looked like a real rainbow-trapping signature.

**Robustness check — this result did not survive:**

| Trial | Mesh | Correlation (x, freq) | Amplitude Ratio |
|------|------|----------------------|-----------------|
| seed=42 | original (2,850 nodes) | **-0.765** | 0.20 |
| seed=42 | refined (5,593 nodes) | **-0.451** | 0.35 |
| seed=1 | original | **-0.293** | 0.25 |
| seed=7 | original | **+0.298** | 0.25 |
| seed=99 | original | **+0.156** | 0.20 |

**Finding:** the frequency-position correlation ranged from -0.765 to +0.298 across 5 trials, flipping sign in 2 of 5. This is not a robust effect — the original -0.765 result was a favorable single-trial outlier. **The rainbow-trapping claim is retracted.**

**What did replicate:** amplitude decay with distance was monotonic and broadly consistent (ratio 0.20-0.35) across all 5 trials. This is a real but much less interesting finding than frequency sorting — it says a denser-to-sparser graded structure attenuates a propagating wave more as material is removed, which does not specifically require density grading, a quasicrystal pattern, or any of the careful tuning this investigation did. It would very likely appear with almost any hole pattern that gets progressively sparser.

---

## 4. What Can Honestly Be Claimed

### 4.1 Supported findings

1. **In-plane bandgap location is controlled by hole coverage fraction**, monotonically, in this simplified FEM model — a clean, controlled, internally-reproducible result (Section 3.1).
2. **Symmetry order (3/6/8/12-fold) has at most a minor effect on in-plane bandgap location** at matched coverage — not the dominant variable (Section 3.2).
3. **A density-graded structure attenuates a transient wave with distance** — robust across multiple trials, but a generic/expected effect, not novel.

### 4.2 Not supported

4. **Frequency-position "rainbow trapping" via density grading** — failed a basic robustness check (random-seed sensitivity). Should not be presented as a finding.

### 4.3 What this does NOT establish, and should not be implied

- Nothing here has been compared to a commercial/validated FEM solver (COMSOL, ANSYS, etc.).
- Out-of-plane bending modes — the physically dominant mode family in a real thin MEMS membrane — were never modeled. The actual claims in the quasicrystal resonator literature (Q-factor, force sensitivity) concern bending modes; this investigation cannot confirm or deny them.
- No fabrication, no experimental measurement, no real device has been built or tested.
- Material constants used silicon, not silicon nitride; thickness was an arbitrary placeholder, not a physically meaningful MEMS value.
- Mesh convergence was checked once (one refinement step on one case), not as a systematic study. The transient correlation result changed substantially (-0.765 → -0.451) under that single refinement, meaning even the surviving "amplitude decay" finding has not been confirmed mesh-independent.

### 4.4 Possible applications — framed honestly

These are speculative directions consistent with the validated finding (3.1), not validated devices:

| Possible application | Basis | Caveat |
|-------------|--------|--------|
| Tunable MEMS vibration notch filter | Density (3.1) sets in-plane bandgap location | Requires bending-mode validation, fabrication, and real device testing — none done here |
| General broadband vibration damping via graded perforation | Section 3.3's surviving (non-novel) finding | Likely achievable with simpler periodic graded patterns too; the quasicrystal/grading specifics are not shown to add value here |

No specific performance numbers (e.g. "25% efficiency improvement," "-7.1dB transmission") can be honestly cited for these applications from this investigation — any such figures would need to come from real literature citations, independently verified, not from this model.

---

## 5. Limitations

- **2D plane-stress (in-plane) elasticity only** — excludes bending/shear, which dominates real thin-membrane vibration.
- **Coarse mesh** (2,850–5,593 nodes) with only one refinement comparison — not a converged result.
- **Custom, unvalidated FEM implementation** — never cross-checked against a commercial solver or analytical benchmark case.
- **Single transient pulse shape, single absorbing-boundary configuration** tested.
- **De Bruijn tiling realization sensitivity** was found to materially affect transient results (Section 3.3) — any future transient claim from this codebase needs multi-seed averaging by default, not a single run.
- **No experimental validation of any kind.**

---

## 6. Code

Two files, in `acoustic-vortex-sim/`:

- **`phononic_symmetry_grading_sim.py`** — scalar toy FDTD model (Stage 1 of the pipeline) and the `debruijn_quasicrystal_points()` quasiperiodic tiling generator shared by both files.
- **`fem_quasicrystal_resonator.py`** — vector FEM: `cst_matrices()` / `assemble()` (stiffness & mass matrix assembly), `build_mesh()` / `build_strip_mesh()` (meshing with connectivity repair), `solve_eigenmodes()` (eigenvalue analysis), `run_case()` / `density_sweep()` / `graded_density_membrane()` (eigenmode experiments), `transient_strip_experiment()` (the transient propagation test).

---

## 7. Future Work

| Task | Honest effort estimate | Outcome |
|------|--------|---------|
| Run a 3D plate/bending-mode FEM | Substantial new model, not an extension of this code | Tests the physics the actual literature claims (Q-factor, force sensitivity) concern |
| Multi-seed averaging built into the transient experiment | Small, immediate | Replace single-trial results with proper statistics by default |
| Mesh convergence study (3+ resolutions, all experiment types) | Moderate | Establish whether Section 3.1/3.2 results are mesh-independent |
| Commercial solver cross-check | Requires access to COMSOL/ANSYS | Validate absolute frequency values |
| Fabrication | Out of scope for a computational investigation | Real experimental validation |

Writing a paper or filing a patent on these results would currently be premature: the dominant relevant physics (bending modes) is unmodeled, and the one striking quantitative result (rainbow trapping) did not survive its own robustness check.

---

## 8. Conclusion

This investigation tested a speculative hypothesis (density-graded quasicrystal membranes producing rainbow trapping) using a custom, simplified, in-plane-only vector FEM model. Across several rounds of real bug-fixing — periodic-boundary artifacts, non-physical hole-cutting fragmentation, reflection contamination — two genuine, internally-reproducible findings emerged: hole density (not symmetry order) controls in-plane bandgap location, and density-graded structures generically attenuate transient waves. The headline rainbow-trapping result did not survive a basic robustness check across random tiling realizations and is retracted. The most valuable outcome of this investigation may be methodological: a promising single-trial result was caught and correctly discarded before being reported as a finding, rather than after.

---

*Researcher: Gavin Branaa*
*Document generated: June 20, 2026*
*Code and data: `acoustic-vortex-sim/phononic_symmetry_grading_sim.py`, `acoustic-vortex-sim/fem_quasicrystal_resonator.py`*
