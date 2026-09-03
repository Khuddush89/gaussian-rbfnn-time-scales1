#!/usr/bin/env python3
"""Regenerate every table and figure in the manuscript.

Usage:  python code/reproduce_results.py [--quick]

Stages
  A  basis properties, quantum grid, clustering grid + refinement study
  B  hybrid time scale, linear and nonlinear boundary value problems
  C  matched basis ablation, optimiser study, recurrence baseline,
     logistic model, joint parameter identification
  F  all eight figures

Modules tsrbf.py (basis), solver.py (residuals, Jacobians, variable
projection) and hybrid.py (hybrid time scales) sit alongside this file.
Master seed 20260901; every reported number is reproducible from this script.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ['stageA.py', 'stageB.py', 'stageC.py', 'redo_inverse.py',
          'make_figures.py']

if __name__ == '__main__':
    for s in STAGES:
        p = HERE / s
        if not p.exists():
            print(f'[skip] {s} not found')
            continue
        print(f'\n===== running {s} =====', flush=True)
        r = subprocess.run([sys.executable, str(p)], cwd=str(HERE))
        if r.returncode != 0:
            sys.exit(f'{s} failed with code {r.returncode}')
    print('\nAll stages complete. Data in results/, figures in figures/.')
