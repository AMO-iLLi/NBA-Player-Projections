"""
Phase 2.1 + 2.3 — aging curves and year-over-year stability.

Two questions:
  2.1  When do NBA players peak? Computed two ways, because the naive way
       is wrong in an instructive direction.
  2.3  Which stats persist year to year (skill) and which do not (noise)?
       This feeds directly into feature selection in Phase 3.

Run:  python src/eda_aging_stability.py
Then: python src/check_eda.py
"""

import sqlite3
from pathlib import Path

from paths import PLOTS, PROCESSED as DATA, RAW, TABLEAU, DB

import matplotlib
matplotlib.use("Agg")  # write PNGs without a display
import matplotlib.pyplot as plt
import pandas as pd


MIN_MP = 1000
MIN_N = 30  # drop age buckets with too few players to be meaningful


# ---------------------------------------------------------------- 2.1a
def aging_cross_sectional(conn):
    """Mean PER at each age, across all players.

    This is the NAIVE view. It is contaminated by survivorship: the only
    38-year-olds left in the league are the ones good enough to still be
    employed, so the curve looks far too flat at the top end. We compute it
    anyway so we can contrast it with 2.1b.

    HAVING rather than WHERE, because WHERE cannot see COUNT(*).
    """
    query = f"""
        SELECT
            age,
            COUNT(*)  AS n,
            AVG(per)  AS avg_per
        FROM player_seasons
        WHERE mp >= {MIN_MP}
        GROUP BY age
        HAVING COUNT(*) >= {MIN_N}
        ORDER BY age
    """
    return pd.read_sql(query, conn)


# ---------------------------------------------------------------- 2.1b
def aging_within_player(conn):
    """How much does the SAME player change from age X to age X+1?

    Controls for survivorship, because every row compares a player to
    himself. `cum` reconstructs the curve by accumulating the deltas.

    The minutes filter applies to BOTH seasons here. That is the opposite
    of what the model table does, and deliberately so: this is descriptive,
    so conditioning on both seasons is fine. In prediction it would not be,
    because season N+1 has not happened yet.
    """
    query = f"""
        SELECT
            a.age                 AS age,
            COUNT(*)              AS n,
            AVG(b.per - a.per)    AS avg_delta
        FROM player_seasons a
        JOIN player_seasons b
          ON a.player_id = b.player_id
         AND b.season    = a.season + 1
        WHERE a.mp >= {MIN_MP}
          AND b.mp >= {MIN_MP}
        GROUP BY a.age
        HAVING COUNT(*) >= {MIN_N}
        ORDER BY a.age
    """
    df = pd.read_sql(query, conn)
    df["cum"] = df["avg_delta"].cumsum()
    return df


# ---------------------------------------------------------------- 2.3
def stability(base: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Correlation between each stat in season N and the same stat in N+1.

    High correlation means the stat is a repeatable skill. Low correlation
    means it is mostly noise, and a model that leans on it will chase
    randomness.

    NaNs are dropped per stat, not across the whole frame. x3p_percent is
    missing for players who never attempted a three; a global dropna would
    let that one sparse column shrink the sample for every other stat.
    """
    nxt = base[["season", "player_id"] + stats].copy()
    nxt["season"] -= 1  # align season N+1 onto its season N row
    nxt = nxt.rename(columns={s: f"{s}_next" for s in stats})

    pair = base[base.mp >= MIN_MP].merge(nxt, on=["season", "player_id"])

    rows = []
    for s in stats:
        d = pair[[s, f"{s}_next"]].dropna()
        rows.append({"stat": s, "n": len(d), "corr": d[s].corr(d[f"{s}_next"])})

    return (
        pd.DataFrame(rows)
        .sort_values("corr", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------- plots
def plot_aging(cross: pd.DataFrame, within: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(cross.age, cross.avg_per, marker="o")
    ax[0].axhline(15, ls="--", lw=1, color="grey")
    ax[0].set_title("Cross-sectional (survivorship-biased)")
    ax[0].set_xlabel("Age")
    ax[0].set_ylabel("Mean PER")

    ax[1].plot(within.age, within.cum, marker="o", color="darkorange")
    ax[1].axhline(0, ls="--", lw=1, color="grey")
    ax[1].set_title("Within-player (survivorship-controlled)")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel("Cumulative PER change")

    fig.suptitle("NBA aging curve, two ways", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "aging_curve.png", dpi=140)
    plt.close(fig)


def plot_stability(stab: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(stab.stat, stab["corr"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("corr(season N, season N+1)")
    ax.set_title(f"Year-over-year stability (mp >= {MIN_MP})")
    fig.tight_layout()
    fig.savefig(PLOTS / "stability.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- main
STATS = [
    "per", "ts_percent", "usg_percent", "bpm", "vorp", "ws_48",
    "ast_percent", "trb_percent", "stl_percent", "blk_percent",
    "tov_percent", "x3p_ar", "ft_percent", "x3p_percent", "g", "mp",
]


def main():
    conn = sqlite3.connect(DATA / "nba.db")
    base = pd.read_parquet(DATA / "player_seasons.parquet")

    cross = aging_cross_sectional(conn)
    within = aging_within_player(conn)
    stab = stability(base, STATS)

    plot_aging(cross, within)
    plot_stability(stab)

    cross.to_csv(DATA / "eda_aging_cross.csv", index=False)
    within.to_csv(DATA / "eda_aging_within.csv", index=False)
    stab.to_csv(DATA / "eda_stability.csv", index=False)

    print("=== CROSS-SECTIONAL ===")
    print(cross.round(3).to_string(index=False))
    print("\npeak age:", int(cross.loc[cross.avg_per.idxmax(), "age"]))

    print("\n=== WITHIN-PLAYER ===")
    print(within.round(3).to_string(index=False))
    print("\npeak age:", int(within.loc[within.cum.idxmax(), "age"]))
    print("last age with positive delta:",
          int(within.loc[within.avg_delta > 0, "age"].max()))

    print("\n=== STABILITY ===")
    print(stab.round(3).to_string(index=False))
    print(f"\nplots -> {PLOTS}")


if __name__ == "__main__":
    main()
