================================================================
Hole Coverage and Rotational Symmetry in Quasicrystal-Perforated
MEMS Plates: A Finite-Element Study
Gavin Branaa
================================================================

This archive contains the complete, corrected manuscript and all the
code that produced its results.

----------------------------------------------------------------
READ THE PAPER
----------------------------------------------------------------
  PAPER_DRAFT.docx   <- start here (Word, figures embedded)
  PAPER_DRAFT.md     <- same paper, plain-text/Markdown source
  paper.tex          <- same paper, LaTeX source (for arXiv submission;
                        compile with pdflatex, or paste into Overleaf)

  figure_convergence.png   <- Figure 1 (element benchmark convergence)
  figure_mode_shapes.png   <- Figure 2 (first six mode shapes)

----------------------------------------------------------------
THE CODE (folder: plate_bending_review/)
----------------------------------------------------------------
Main finite-element implementations:
  fem_plate_bending_homogenized.py
        The corrected, area-fraction-homogenized element used for
        EVERY result in the paper. Includes the corrected de Bruijn
        hole-pattern generator. Run directly to reproduce the
        benchmark + density-sweep sanity checks.
  fem_plate_bending_2d_v2.py
        The original binary-removal implementation, retained for
        record (superseded -- see the paper's Section 2.2).

Result-generating scripts (each reproduces one part of the paper):
  rerun_main_table_v2.py     -> Section 3.4 main results table
  noise_floor_rerun.py       -> Section 3.4 mesh-convergence noise floors
  intermediate_coverage.py   -> Section 3.4 intermediate coverage levels
  exponent_recheck.py        -> Section 3.3 exponent calibration
  phi_floor_sensitivity.py   -> Section 2.2 phi-floor sensitivity check
  bandgap_quadratic_rerun.py -> Section 4.3 bandgap-width test
  cross_check_lst6_element.py -> Section 2.7 independent cross-validation
  make_figures.py            -> regenerates the two figures

Requirements: Python 3 with numpy, scipy, matplotlib.

----------------------------------------------------------------
NOTE
----------------------------------------------------------------
This is a COMPUTATIONAL study. No physical device was fabricated or
measured. All results are finite-element simulations.

The build scripts md_to_docx.py and embed_figures.py (in the root)
convert the Markdown source to the Word document; they are included
for completeness but are not needed to read the paper.
