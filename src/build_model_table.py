"""
Build the modeling table: season N features -> season N+1 target.

Inclusion rules (decided in Phase 1):
  - >= MIN_MP minutes in season N only. Filtering on N+1 too would condition
    on the outcome, which is a worse bug than the survivorship it fixes.
  - players absent in N+1 are dropped. Residual attrition is reported, not
    hidden, because it is concentrated in age 33+ and bottom-quartile PER.

Everything here is derived from seasons <= N. Nothing from N+1 enters the
feature set; only the target comes from N+1.
"""

import sqlite3
from pathlib import Path

from paths import PLOTS, PROCESSED as DATA, RAW, TABLEAU, DB

import numpy as np
import pandas as pd

KEYS = ["season", "player_id"]
MIN_MP = 1000
TARGET = "per"

# season-N columns carried forward as features
FEATURES = [
    "age", "experience", "g", "gs", "mp",
    "per", "ts_percent", "usg_percent", "x3p_ar", "f_tr",
    "orb_percent", "drb_percent", "trb_percent", "ast_percent",
    "stl_percent", "blk_percent", "tov_percent",
    "ows", "dws", "ws", "ws_48", "obpm", "dbpm", "bpm", "vorp",
    "o_rtg", "d_rtg",
    "ht_in_in", "wt",
    "team_pace", "team_o_rtg", "team_d_rtg", "team_w", "team_age",
    "avg_dist_fga", "percent_fga_from_x3p_range", "percent_assisted_x2p_fg",
    "pg_percent", "sg_percent", "sf_percent", "pf_percent", "c_percent",
    "was_traded",
]

# columns we look back on for trend/history features
HISTORY = ["per", "mp", "g", "usg_percent", "ts_percent", "bpm", "vorp", "ws_48"]


def load_base():
    df = pd.read_parquet(DATA / "player_seasons.parquet")
    return df.sort_values(["player_id", "season"]).reset_index(drop=True)


def add_history(df):
    """Lags, deltas and rolling means over seasons <= N. Shift-based, so no
    future information can leak in."""
    g = df.groupby("player_id", sort=False)
    out = {}

    for col in HISTORY:
        s = df[col]
        lag1 = g[col].shift(1)
        lag2 = g[col].shift(2)
        out[f"{col}_lag1"] = lag1
        out[f"{col}_lag2"] = lag2
        out[f"{col}_delta1"] = s - lag1
        out[f"{col}_delta2"] = lag1 - lag2
        # rolling mean of the *previous* 3 seasons, current one excluded
        out[f"{col}_roll3"] = (
            g[col].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    # seasons of history actually available for this player at season N
    out["seasons_observed"] = g.cumcount() + 1

    # durability: share of an 82-game season played, this year and last
    out["g_share"] = df["g"] / 82.0
    out["g_share_lag1"] = g["g"].shift(1) / 82.0

    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def add_curves(df):
    """Nonlinear age/experience terms so a linear model can express a peak."""
    df = df.copy()
    df["age_sq"] = df["age"] ** 2
    df["exp_sq"] = df["experience"] ** 2
    df["age_x_mp"] = df["age"] * df["mp"] / 1000.0
    return df


def main():
    base = load_base()
    base = add_history(base)
    base = add_curves(base)

    # ---- attach season N+1 target ------------------------------------------
    nxt = base[KEYS + [TARGET, "mp", "g"]].copy()
    nxt["season"] = nxt["season"] - 1  # align N+1 row onto its season N row
    nxt = nxt.rename(
        columns={TARGET: "target", "mp": "next_mp", "g": "next_g"}
    )

    df = base.merge(nxt, on=KEYS, how="left")
    assert len(df) == len(base), "target merge changed row count"

    # ---- apply inclusion rules ---------------------------------------------
    n_start = len(df)
    df = df[df["season"] < df["season"].max()]  # last season has no N+1
    n_eligible = len(df)

    df = df[df["mp"] >= MIN_MP]  # season N only, never N+1
    n_qualified = len(df)

    attrition = df["target"].isna().mean()
    df = df[df["target"].notna()].copy()
    n_final = len(df)

    # ---- leakage guard ------------------------------------------------------
    leaked = [c for c in df.columns if c.startswith("next_") and c != "target"]
    feature_cols = [c for c in FEATURES if c in df.columns]
    feature_cols += [
        c for c in df.columns
        if any(c.startswith(f"{h}_") for h in HISTORY)
        or c in {"seasons_observed", "g_share", "g_share_lag1",
                 "age_sq", "exp_sq", "age_x_mp"}
    ]
    feature_cols = sorted(set(feature_cols))
    assert not set(feature_cols) & set(leaked + ["target"]), "future column in features"

    keep = KEYS + ["player", "team", "pos", "target"] + feature_cols
    model_df = df[keep].copy()

    # ---- persistence baseline: predict target = this season's PER ----------
    resid = model_df["target"] - model_df[TARGET]
    mae = resid.abs().mean()
    rmse = np.sqrt((resid ** 2).mean())

    # ---- write --------------------------------------------------------------
    model_df.to_parquet(DATA / "model_table.parquet", index=False)
    with sqlite3.connect(DATA / "nba.db") as conn:
        model_df.to_sql("model_table", conn, if_exists="replace", index=False)

    print(f"eligible player-seasons (N with an N+1)   {n_eligible:,}")
    print(f"after mp >= {MIN_MP} in season N            {n_qualified:,}")
    print(f"dropped: no season N+1                     {n_qualified - n_final:,}"
          f"  ({attrition*100:.1f}% attrition)")
    print(f"final modeling rows                        {n_final:,}")
    print(f"players                                    {model_df.player_id.nunique():,}")
    print(f"seasons                                    {model_df.season.min()} - {model_df.season.max()}")
    print(f"features                                   {len(feature_cols)}")
    print()
    print(f"PERSISTENCE BASELINE  MAE {mae:.3f}   RMSE {rmse:.3f}")
    print(f"target std                {model_df.target.std():.3f}")
    print(f"corr(PER_N, PER_N+1)      {model_df[TARGET].corr(model_df.target):.3f}")


if __name__ == "__main__":
    main()
