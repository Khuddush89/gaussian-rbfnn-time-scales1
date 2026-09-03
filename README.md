# Gaussian Radial Basis Function Neural Networks on Time Scales

Final audited revision bundle for *Mathematics* (MDPI). The manuscript uses the
MDPI Mathematics class included under `Definitions/`. All manuscript tables are
written directly in the main `.tex` files; no separate table `.tex` files are
used.

## Main submission files

| Path | Contents |
|---|---|
| `manuscript_clean.tex` / `.pdf` | Clean revised MDPI manuscript, 18 pages |
| `manuscript_highlighted.tex` / `.pdf` | Revised manuscript with substantive changes in blue, 18 pages |
| `Response_to_Reviewers.tex` / `.pdf` | Point-by-point response to the actual 5 Reviewer 1 comments and 4 Reviewer 2 comments, 4 pages |
| `REVIEWER_COMMENTS.txt` | Reviewer comments used to audit the response letter |
| `FINAL_AUDIT.md` | Final mathematical, numerical, LaTeX and PDF validation summary |
| `Definitions/` | MDPI class, bibliography styles, logos and journal assets |
| `figures/` | Eight figures, each supplied as a 600 dpi PNG and vector PDF |
| `data/` | 47 CSV result files and 3 JSON stage summaries |
| `code/` | Basis, solvers, experiments, figure generation and independent theory checks |
| `compile_logs/` | Final LaTeX logs |

## Reproduction

Install the tested numerical dependencies:

```bash
pip install -r requirements.txt
python code/reproduce_results.py
```

The master random seed is `20260901`. The final validation environment was
Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3 and Matplotlib 3.10.8.
Wall times are machine dependent.

The complete reproduction pipeline runs:

1. `stageA.py`: basis identities, admissibility, quantum problem, clustering
   problem and refinement study.
2. `stageB.py`: hybrid time scale and linear/nonlinear boundary value problems.
3. `stageC.py`: strict same-width-box basis ablation, generous-width classical
   sensitivity control, optimiser comparison, recurrence baseline, logistic
   model and parameter identification setup.
4. `redo_inverse.py`: final joint parameter-identification comparison.
5. `make_figures.py`: regenerates all eight PNG and vector-PDF figures.

`code/final_consistency_check.py` cross-checks the high-risk manuscript statements against the final CSV/JSON files, including the optimiser row, training-mode timings, hybrid collocation count, logistic certificate roundoff note and bibliography cleanup. Its final line is `ALL FINAL CONSISTENCY CHECKS PASSED`.

`code/verify_theory.py` is an independent numerical check of the activation
identity, envelope, continuous-limit estimate, unimodality, analytic width
Jacobian, quantum closed form, integer-grid worked example, hybrid identities,
corrected nonuniform-graininess second-order manufactured equation, and the
first-order manufactured equation. The final audit gives no failed checks.


## LaTeX and PDF checks

Compile with:

```bash
pdflatex manuscript_clean.tex
pdflatex manuscript_clean.tex
pdflatex manuscript_highlighted.tex
pdflatex manuscript_highlighted.tex
pdflatex Response_to_Reviewers.tex
pdflatex Response_to_Reviewers.tex
```

Final logs contain no LaTeX errors, undefined citations, undefined references,
overfull boxes or underfull boxes. PDF preflight confirms that the clean and
highlighted manuscripts each contain 18 pages and the response letter contains
4 pages. All three PDFs were rendered and visually inspected after compilation.

No Unicode em dash or en dash characters are used in the three LaTeX sources.
## Reference numbering

The manuscript bibliography is arranged in strict first-citation order (first come, first serve), not alphabetically. See `REFERENCE_ORDER.md` for the verified sequence.

