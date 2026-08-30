"""Single source of truth for every path in the project.

Every script imports from here rather than hardcoding, so moving the repo
or renaming a folder is a one-line change instead of nine.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"              # untouched Kaggle CSVs
PROCESSED = ROOT / "data" / "processed"  # everything this project generates
PLOTS = ROOT / "plots"
TABLEAU = ROOT / "tableau"

for _d in (PROCESSED, PLOTS, TABLEAU):
    _d.mkdir(parents=True, exist_ok=True)

DB = PROCESSED / "nba.db"
