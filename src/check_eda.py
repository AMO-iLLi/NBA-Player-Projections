"""
Checks your Phase 2.1 + 2.3 implementation.

Run:  python src/check_eda.py

It imports your functions, runs them, and compares against values computed
independently from the raw data. It tells you WHICH check failed and what it
expected, so a red result should point you at the bug rather than just
saying no.
"""

import sqlite3
import sys
import traceback
from pathlib import Path

from paths import PLOTS, PROCESSED as DATA, RAW, TABLEAU, DB

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))



PASS, FAIL, WARN = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m", "\033[93mWARN\033[0m"
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"\n         {detail}" if not cond and detail else ""))


def near(a, b, tol=0.02):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


# ---- expected values, computed independently from the parquet ------------
def expected():
    b = pd.read_parquet(DATA / "player_seasons.parquet")
    q = b[b.mp >= 1000]

    cross = (
        q.groupby("age")["per"].agg(["size", "mean"])
        .rename(columns={"size": "n", "mean": "avg_per"})
        .query("n >= 30").reset_index()
    )

    nxt = b[["season", "player_id", "per", "mp"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"per": "per_next", "mp": "mp_next"})
    pair = q.merge(nxt, on=["season", "player_id"])
    pair = pair[pair.mp_next >= 1000]
    pair["delta"] = pair.per_next - pair.per
    within = (
        pair.groupby("age")["delta"].agg(["size", "mean"])
        .rename(columns={"size": "n", "mean": "avg_delta"})
        .query("n >= 30").reset_index()
    )
    within["cum"] = within.avg_delta.cumsum()
    return cross, within


def main():
    try:
        import eda_aging_stability as m
    except Exception:
        print("Could not import src/eda_aging_stability.py:\n")
        traceback.print_exc()
        return 1

    conn = sqlite3.connect(DATA / "nba.db")
    base = pd.read_parquet(DATA / "player_seasons.parquet")
    exp_cross, exp_within = expected()

    # ---------------------------------------------------------- 2.1a
    print("\n2.1a  aging_cross_sectional")
    try:
        cross = m.aging_cross_sectional(conn)
        check("returns a DataFrame", isinstance(cross, pd.DataFrame))
        check("has columns age, n, avg_per",
              {"age", "n", "avg_per"} <= set(cross.columns),
              f"got {list(cross.columns)}")
        check("sorted by age", cross.age.is_monotonic_increasing)
        check("MIN_N filter applied", (cross.n >= 30).all(),
              "some age buckets have n < 30 — use HAVING COUNT(*) >= 30")
        check(f"row count == {len(exp_cross)}", len(cross) == len(exp_cross),
              f"got {len(cross)} — check the mp >= 1000 filter")
        peak = int(cross.loc[cross.avg_per.idxmax(), "age"])
        check("peak age == 26", peak == 26, f"got {peak}")
        check("avg_per at age 25 ~ 15.605",
              near(cross.loc[cross.age == 25, "avg_per"].iloc[0], 15.605),
              "value is off — is the mp filter right?")
    except Exception:
        traceback.print_exc()
        results.append(False)

    # ---------------------------------------------------------- 2.1b
    print("\n2.1b  aging_within_player")
    try:
        within = m.aging_within_player(conn)
        check("returns a DataFrame", isinstance(within, pd.DataFrame))
        check("has columns age, n, avg_delta, cum",
              {"age", "n", "avg_delta", "cum"} <= set(within.columns),
              f"got {list(within.columns)}")
        check("sorted by age", within.age.is_monotonic_increasing)
        check(f"row count == {len(exp_within)}", len(within) == len(exp_within),
              f"got {len(within)} — did you require mp >= 1000 on BOTH seasons?")
        check("n smaller than cross-sectional at same age",
              within.n.sum() < exp_cross.n.sum(),
              "pairs should be fewer than seasons — check the self-join")
        check("deltas positive when young",
              within.loc[within.age <= 22, "avg_delta"].gt(0).all())
        check("deltas negative when old",
              within.loc[within.age >= 30, "avg_delta"].lt(0).all())
        check("cum is a running total",
              near(within.cum.iloc[-1], within.avg_delta.sum(), 0.01))
        peak = int(within.loc[within.cum.idxmax(), "age"])
        check("within-player peak age == 24", peak == 24, f"got {peak}")
        check("avg_delta at age 30 ~ -0.946",
              near(within.loc[within.age == 30, "avg_delta"].iloc[0], -0.946))
    except Exception:
        traceback.print_exc()
        results.append(False)

    # ------------------------------------------------------------ 2.3
    print("\n2.3  stability")
    try:
        stab = m.stability(base, m.STATS)
        check("returns a DataFrame", isinstance(stab, pd.DataFrame))
        check("has columns stat, n, corr",
              {"stat", "n", "corr"} <= set(stab.columns),
              f"got {list(stab.columns)}")
        check("one row per stat", len(stab) == len(m.STATS),
              f"got {len(stab)} rows for {len(m.STATS)} stats")
        check("sorted by corr desc", stab["corr"].is_monotonic_decreasing)
        check("trb_percent is most stable",
              stab.iloc[0].stat == "trb_percent", f"got {stab.iloc[0].stat}")
        check("g is least stable",
              stab.iloc[-1].stat == "g", f"got {stab.iloc[-1].stat}")
        check("per corr ~ 0.777",
              near(stab.loc[stab.stat == "per", "corr"].iloc[0], 0.777))
        check("x3p_percent corr ~ 0.503",
              near(stab.loc[stab.stat == "x3p_percent", "corr"].iloc[0], 0.503))
        n_per = int(stab.loc[stab.stat == "per", "n"].iloc[0])
        n_3p = int(stab.loc[stab.stat == "x3p_percent", "n"].iloc[0])
        check("per-stat dropna, not global",
              n_per == 7088 and n_3p < n_per,
              f"per n={n_per} (want 7088), x3p_percent n={n_3p} (want < per). "
              "Dropping NaNs across the whole frame shrinks every column.")
    except NotImplementedError:
        print(f"  [{WARN}] stability() not implemented yet")
    except Exception:
        traceback.print_exc()
        results.append(False)

    # ---------------------------------------------------------- outputs
    print("\noutputs")
    for f in ["aging_curve.png", "stability.png"]:
        check(f"plots/{f} exists", (PLOTS / f).exists(),
              "run src/eda_aging_stability.py first")

    n_pass, n_tot = sum(results), len(results)
    print(f"\n{'-'*46}\n{n_pass}/{n_tot} checks passed")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())
