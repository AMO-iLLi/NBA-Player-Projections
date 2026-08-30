"""Rebuild the entire project from raw CSVs.

Usage:  python src/run_all.py
"""

import subprocess
import sys
from pathlib import Path

STEPS = [
    ("Phase 0  base table",        "build_base.py"),
    ("Phase 1  model table",       "build_model_table.py"),
    ("Phase 2  aging + stability", "eda_aging_stability.py"),
    ("Phase 2  usage/team/extremes", "eda_phase2b.py"),
    ("Phase 3  feature engineering", "build_features.py"),
    ("Phase 4  modeling",          "train_models.py"),
    ("Phase 5  classification + projections", "phase5_analysis.py"),
    ("Phase 6  Tableau export",    "export_bi.py"),
]

here = Path(__file__).resolve().parent

for label, script in STEPS:
    print(f"\n{'='*60}\n{label}  ({script})\n{'='*60}")
    r = subprocess.run([sys.executable, str(here / script)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        sys.exit(f"FAILED at {script}")
    print(r.stdout[-1500:])

print("\nAll phases complete.")
