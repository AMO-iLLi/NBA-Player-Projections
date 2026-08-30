"""
Build a clean player-season base table from the Basketball-Reference CSVs.

Handles the two things that silently corrupt this dataset:
  1. Traded players appear as a 2TM/3TM aggregate row PLUS one row per team.
     We keep the aggregate and drop the splits.
  2. lg contains ABA and BAA alongside NBA.

Output: one row per (season, player_id), written to SQLite and Parquet.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from paths import RAW, PROCESSED as OUT, DB

KEYS = ["season", "player_id"]
# columns repeated across files; keep them once from the anchor table
SHARED = ["lg", "player", "age", "team", "pos", "g", "gs", "mp"]

MIN_SEASON = 1997  # Shooting and Play-By-Play do not exist before this


def load(name):
    return pd.read_csv(RAW / name)


def drop_team_splits(df):
    """Keep the 2TM/3TM aggregate row for traded players, drop per-team rows."""
    is_agg = df["team"].astype(str).str.contains(r"^\dTM$", regex=True, na=False)
    traded = set(map(tuple, df.loc[is_agg, KEYS].values))
    keep = [
        not (tuple(k) in traded and not agg)
        for k, agg in zip(df[KEYS].values, is_agg)
    ]
    return df.loc[keep].copy()


def clean(df, min_season=MIN_SEASON):
    df = df[df["lg"] == "NBA"].copy()
    df = df[df["season"] >= min_season]
    df = drop_team_splits(df)
    return df


def main():
    # ---- anchor table: Advanced carries the target metrics -------------------
    adv = clean(load("Advanced.csv"))

    totals = clean(load("Player_Totals.csv")).drop(columns=SHARED)
    # Season_Info is narrower than the rest; only experience is new
    info = clean(load("Player_Season_Info.csv"))[KEYS + ["experience"]]
    shooting = clean(load("Player_Shooting.csv")).drop(columns=SHARED)
    pbp = clean(load("Player_Play_By_Play.csv")).drop(columns=SHARED)

    # o_rtg / d_rtg are the only Per_100 columns not derivable from Totals
    per100 = clean(load("Per_100_Poss.csv"))[KEYS + ["o_rtg", "d_rtg"]]

    base = adv
    for name, df in [
        ("totals", totals),
        ("info", info),
        ("shooting", shooting),
        ("pbp", pbp),
        ("per100", per100),
    ]:
        before = len(base)
        base = base.merge(df, on=KEYS, how="left", suffixes=("", f"_{name}"))
        assert len(base) == before, f"{name} merge changed row count"

    # ---- team context (season N only; using N+1 would leak the future) ------
    team = load("Team_Summaries.csv")
    team = team[(team["lg"] == "NBA") & team["abbreviation"].notna()]
    team = team[
        [
            "season", "abbreviation", "pace", "o_rtg", "d_rtg",
            "w", "l", "age", "x3p_ar", "ts_percent",
        ]
    ].rename(
        columns={
            "abbreviation": "team",
            "pace": "team_pace",
            "o_rtg": "team_o_rtg",
            "d_rtg": "team_d_rtg",
            "w": "team_w",
            "l": "team_l",
            "age": "team_age",
            "x3p_ar": "team_x3p_ar",
            "ts_percent": "team_ts_percent",
        }
    )
    base = base.merge(team, on=["season", "team"], how="left")

    # ---- static player attributes ------------------------------------------
    # hof is deliberately excluded: it depends on the player's entire future
    career = load("Player_Career_Info.csv")[
        ["player_id", "ht_in_in", "wt", "birth_date", "from", "to"]
    ].rename(columns={"from": "first_season", "to": "last_season"})
    base = base.merge(career, on="player_id", how="left")

    base = base.sort_values(KEYS).reset_index(drop=True)

    # ---- validate -----------------------------------------------------------
    dupes = base.duplicated(subset=KEYS).sum()
    assert dupes == 0, f"{dupes} duplicate player-seasons survived"
    assert base["season"].min() >= MIN_SEASON

    # Traded players keep the aggregate row, which carries a 2TM/3TM code by
    # design. Team_Summaries has no such team, so their team_* fields are null.
    base["was_traded"] = (
        base["team"].astype(str).str.contains(r"^\dTM$", regex=True, na=False)
    )
    traded = int(base["was_traded"].sum())
    orphan_null = int(
        base.loc[~base["was_traded"], "team_pace"].isna().sum()
    )
    assert orphan_null == 0, f"{orphan_null} single-team rows failed team join"

    # ---- write --------------------------------------------------------------
    base.to_parquet(OUT / "player_seasons.parquet", index=False)
    with sqlite3.connect(OUT / "nba.db") as conn:
        base.to_sql("player_seasons", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ps ON player_seasons(season, player_id)"
        )

    print(f"rows           {len(base):,}")
    print(f"players        {base.player_id.nunique():,}")
    print(f"seasons        {base.season.min()} - {base.season.max()}")
    print(f"columns        {base.shape[1]}")
    print(f"traded rows    {traded:,} (team_* null by design)")
    print(f"written        {OUT}/player_seasons.parquet, {OUT}/nba.db")


if __name__ == "__main__":
    main()
