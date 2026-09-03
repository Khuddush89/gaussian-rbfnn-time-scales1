"""Cross-check manuscript statements against final CSV/JSON outputs.

This script is intentionally lightweight: it does not rerun optimisations. It
verifies the final numerical summaries and source wording that were identified
as high-risk during the revision audit.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = (ROOT / "manuscript_clean.tex").read_text()


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg)

opt = pd.read_csv(DATA / "optimizer_summary.csv").set_index("method")
lm = opt.loc["lm"]
need(abs(lm.median_max_error - 2.7755575615628914e-15) < 1e-30,
     "LM median error matches optimizer_summary.csv")
need(abs(lm.best_max_error - 2.220446049250313e-16) < 1e-30,
     "LM best error matches optimizer_summary.csv")
need(abs(lm.success_rate - 0.80) < 1e-15 and abs(lm.admissible_rate - 0.80) < 1e-15,
     "LM success/admissibility rates match final CSV")
need("Levenberg--Marquardt & $2.78\\times10^{-15}$ & $2.22\\times10^{-16}$" in SRC,
     "manuscript LM row uses final rounded CSV values")
need("when $\\|e\\|_\\infty<10^{-8}$ \\emph{and} its widths are admissible" in SRC,
     "optimizer success rule is explicit in manuscript")

modes = pd.read_csv(DATA / "quantum_training_modes.csv").set_index("mode")
expected = {"fd": 0.247, "analytic": 0.096, "varpro": 0.125, "hybrid": 0.135}
for mode, rounded in expected.items():
    need(round(float(modes.loc[mode, "median_wall_s"]), 3) == rounded,
         f"{mode} median wall time rounds to {rounded:.3f} s")
for val in ("0.247", "0.096", "0.125", "0.135"):
    need(val in SRC, f"manuscript contains final training-mode time {val} s")

stageb = json.loads((DATA / "stageB.json").read_text())["hybrid"]
need(stageb["sample_nodes"] == 33 and stageb["residual_equations"] == 32,
     "hybrid JSON distinguishes 33 sample nodes from 32 residual equations")
need(stageb["dense_residual_nodes"] == 30 and stageb["scattered_residual_nodes"] == 2,
     "hybrid JSON records 30 dense and 2 scattered residual nodes")
need("The sampled representation contains 33 nodes" in SRC and
     "the residual vector has 32 equations" in SRC,
     "hybrid sample/residual count is explicit in manuscript")

log = pd.read_csv(DATA / "logistic_solution.csv")
gap = (log.abs_error_population - log.certificate_population).max()
need(0 < gap < 5e-15, "logistic pointwise certificate discrepancy is roundoff-scale")
need(log.certificate_population.max() >= log.abs_error_population.max(),
     "logistic sup-norm certificate remains above sup-norm error")

need("the two cases in\nTable~\\ref{tab:baseline} marked ``no''" in SRC,
     "baseline wording says two cases, not three columns")
need("Yaslan2016" not in SRC, "unused Yaslan bibliography entry removed")
for key in ("GolubPereyra1973", "ORLeary2013", "Branch1999", "Virtanen2020"):
    need(f"\\cite{{{key}}}" in SRC or key in SRC.split("\\cite{")[-1],
         f"{key} is present in source and citation audit is handled")

print("ALL FINAL CONSISTENCY CHECKS PASSED")
