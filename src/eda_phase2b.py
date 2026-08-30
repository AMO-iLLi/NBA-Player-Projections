"""
Phase 2.2 + 2.4 + 2.5 — usage vs efficiency, team change impact, and
breakout / decline case review.

Run:  python src/eda_phase2b.py

2.2  Is there really a usage-efficiency tradeoff?
2.4  Do players who change teams swing more than those who stay?
2.5  Do the biggest year-over-year moves pass the eye test?
"""

import sqlite3
from pathlib import Path

from paths import PLOTS, PROCESSED as DATA, RAW, TABLEAU, DB

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MIN_MP = 1000
POSITIONS = ["PG", "SG", "SF", "PF", "C"]


# ------------------------------------------------------------------ setup
def load_pairs(base: pd.DataFrame) -> pd.DataFrame:
    """Season N joined to season N+1, both seasons >= MIN_MP.

    Descriptive only, so conditioning on both seasons is fine here. The
    model table deliberately does not do this.
    """
    cols = ["season", "player_id", "per", "usg_percent", "ts_percent",
            "mp", "team", "age"]
    nxt = base[cols].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={c: f"{c}_next" for c in cols
                              if c not in ("season", "player_id")})

    p = base[base.mp >= MIN_MP].merge(nxt, on=["season", "player_id"])
    p = p[p.mp_next >= MIN_MP].copy()

    p["d_per"] = p.per_next - p.per
    p["d_usg"] = p.usg_percent_next - p.usg_percent
    p["d_ts"] = p.ts_percent_next - p.ts_percent
    p["abs_d_per"] = p.d_per.abs()
    # team code changed between seasons; 2TM/3TM counts as a change
    p["changed_team"] = p.team != p.team_next
    return p


# -------------------------------------------------------------------- 2.2
def usage_efficiency(base: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Cross-sectional and within-player usage vs efficiency.

    Cross-sectional asks: do high-usage players shoot worse than low-usage
    ones? Within-player asks the better question: when the SAME player's
    usage rises, does his efficiency fall? The second controls for talent,
    because stars have both high usage and high efficiency, which masks any
    tradeoff in the cross-section.
    """
    q = base[base.mp >= MIN_MP]
    overall = q.usg_percent.corr(q.ts_percent)
    by_pos = {
        pos: q[q.pos == pos].usg_percent.corr(q[q.pos == pos].ts_percent)
        for pos in POSITIONS if (q.pos == pos).any()
    }
    within = pairs.d_usg.corr(pairs.d_ts)

    # slope of TS% on usage, in TS points per 1pt of usage
    fit = np.polyfit(q.usg_percent.dropna(),
                     q.loc[q.usg_percent.notna(), "ts_percent"].fillna(
                         q.ts_percent.mean()), 1)
    return {"overall": overall, "by_pos": by_pos,
            "within": within, "slope": fit[0], "n_cross": len(q),
            "n_within": len(pairs)}


# -------------------------------------------------------------------- 2.4
def team_change(pairs: pd.DataFrame) -> pd.DataFrame:
    """Do movers swing more than stayers?

    CONFOUND, and it runs both ways: players get traded BECAUSE something
    changed. A decline can cause the move as easily as the move causes the
    decline. This is an association, not a causal estimate, and it should
    be written up that way.
    """
    g = pairs.groupby("changed_team")
    out = g.agg(
        n=("d_per", "size"),
        mean_d_per=("d_per", "mean"),
        mean_abs_d_per=("abs_d_per", "mean"),
        std_d_per=("d_per", "std"),
        mean_age=("age", "mean"),
        mean_per=("per", "mean"),
    ).round(3)
    out.index = ["stayed", "changed"]
    return out


def team_change_matched(pairs: pd.DataFrame) -> pd.DataFrame:
    """Same comparison inside age and performance buckets.

    If movers still swing more after matching on who they are, the effect
    is less likely to be pure selection.
    """
    p = pairs.copy()
    p["age_bin"] = pd.cut(p.age, [17, 24, 27, 30, 50],
                          labels=["<=24", "25-27", "28-30", "31+"])
    p["per_bin"] = pd.qcut(p.per, 3, labels=["low", "mid", "high"])
    out = (p.groupby(["per_bin", "changed_team"], observed=True)
             .agg(n=("d_per", "size"), mean_abs_d_per=("abs_d_per", "mean"))
             .round(3).reset_index())
    out["changed_team"] = out.changed_team.map({False: "stayed", True: "changed"})
    return out


# -------------------------------------------------------------------- 2.5
def extremes(pairs: pd.DataFrame, k: int = 10) -> tuple:
    cols = ["player", "season", "age", "per", "per_next", "d_per",
            "mp", "mp_next", "changed_team"]
    up = pairs.nlargest(k, "d_per")[cols].round(2)
    down = pairs.nsmallest(k, "d_per")[cols].round(2)
    return up, down


# ------------------------------------------------------------------ plots
def plot_usage_efficiency(base, pairs, res):
    q = base[base.mp >= MIN_MP]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].scatter(q.usg_percent, q.ts_percent, s=5, alpha=0.15,
                  color="steelblue", edgecolors="none")
    xs = np.linspace(q.usg_percent.min(), q.usg_percent.max(), 50)
    m, c = np.polyfit(q.dropna(subset=["usg_percent", "ts_percent"]).usg_percent,
                      q.dropna(subset=["usg_percent", "ts_percent"]).ts_percent, 1)
    ax[0].plot(xs, m * xs + c, color="darkred", lw=2)
    ax[0].set_xlabel("Usage rate (%)")
    ax[0].set_ylabel("True shooting %")
    ax[0].set_title(f"Cross-sectional   r = {res['overall']:.3f}")

    ax[1].scatter(pairs.d_usg, pairs.d_ts, s=5, alpha=0.15,
                  color="darkorange", edgecolors="none")
    ax[1].axhline(0, ls="--", lw=1, color="grey")
    ax[1].axvline(0, ls="--", lw=1, color="grey")
    ax[1].set_xlabel("Change in usage rate")
    ax[1].set_ylabel("Change in TS%")
    ax[1].set_title(f"Within-player   r = {res['within']:.3f}")

    fig.suptitle("Usage vs efficiency: the tradeoff that is not there",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "usage_efficiency.png", dpi=140)
    plt.close(fig)


def plot_team_change(pairs, tc):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    stayed = pairs[~pairs.changed_team].d_per
    changed = pairs[pairs.changed_team].d_per
    bins = np.linspace(-8, 8, 45)
    ax[0].hist(stayed, bins=bins, alpha=0.55, density=True, label="stayed")
    ax[0].hist(changed, bins=bins, alpha=0.55, density=True, label="changed")
    ax[0].axvline(0, ls="--", lw=1, color="grey")
    ax[0].set_xlabel("PER change, season N to N+1")
    ax[0].set_ylabel("density")
    ax[0].set_title("Distribution of PER change")
    ax[0].legend()

    ax[1].bar(tc.index, tc.mean_abs_d_per, color=["steelblue", "indianred"])
    for i, v in enumerate(tc.mean_abs_d_per):
        ax[1].text(i, v + 0.03, f"{v:.2f}", ha="center")
    ax[1].set_ylabel("mean |PER change|")
    ax[1].set_title("Volatility: movers vs stayers")

    fig.suptitle("Team change and performance swing (association, not cause)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "team_change.png", dpi=140)
    plt.close(fig)


def plot_extremes(up, down):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    for a, df, color, title in [
        (ax[0], up.iloc[::-1], "seagreen", "Biggest PER jumps"),
        (ax[1], down.iloc[::-1], "indianred", "Biggest PER drops"),
    ]:
        labels = [f"{r.player} {int(r.season)}" for r in df.itertuples()]
        a.barh(labels, df.d_per, color=color)
        a.axvline(0, lw=1, color="grey")
        a.set_xlabel("PER change to next season")
        a.set_title(title)

    fig.suptitle(f"Year-over-year extremes (both seasons >= {MIN_MP} min)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "extremes.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    base = pd.read_parquet(DATA / "player_seasons.parquet")
    pairs = load_pairs(base)

    res = usage_efficiency(base, pairs)
    tc = team_change(pairs)
    tcm = team_change_matched(pairs)
    up, down = extremes(pairs)

    plot_usage_efficiency(base, pairs, res)
    plot_team_change(pairs, tc)
    plot_extremes(up, down)

    tc.to_csv(DATA / "eda_team_change.csv")
    tcm.to_csv(DATA / "eda_team_change_matched.csv", index=False)
    up.to_csv(DATA / "eda_breakouts.csv", index=False)
    down.to_csv(DATA / "eda_declines.csv", index=False)

    with sqlite3.connect(DATA / "nba.db") as conn:
        pairs[["season", "player_id", "player", "age", "per", "per_next",
               "d_per", "d_usg", "d_ts", "changed_team"]].to_sql(
            "season_pairs", conn, if_exists="replace", index=False)

    print(f"pairs (both seasons >= {MIN_MP} min): {len(pairs):,}\n")

    print("=== 2.2  USAGE vs EFFICIENCY ===")
    print(f"cross-sectional r  {res['overall']:+.3f}  (n={res['n_cross']:,})")
    print(f"within-player   r  {res['within']:+.3f}  (n={res['n_within']:,})")
    print("by position:")
    for k, v in res["by_pos"].items():
        print(f"  {k}  {v:+.3f}")

    print("\n=== 2.4  TEAM CHANGE ===")
    print(tc.to_string())
    print("\nby prior-performance tercile:")
    print(tcm.to_string(index=False))

    print("\n=== 2.5  BIGGEST JUMPS ===")
    print(up.to_string(index=False))
    print("\n=== 2.5  BIGGEST DROPS ===")
    print(down.to_string(index=False))

    print(f"\nplots -> {PLOTS}")


if __name__ == "__main__":
    main()
