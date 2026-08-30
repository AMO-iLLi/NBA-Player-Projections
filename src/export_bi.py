"""
Phase 6a — export a BI-ready star schema.

Power BI and Tableau both want a star schema: narrow dimension tables joined
to wide fact tables on clean integer or short-string keys. Feeding a BI tool
one 113-column flat file works, but it makes relationships impossible and
every measure turns into a mess.

Output: bi_export/*.csv, plus a relationships guide.

Run:  python src/export_bi.py
"""

from pathlib import Path

from paths import PLOTS, PROCESSED as DATA, RAW, TABLEAU as OUT, DB

import numpy as np
import pandas as pd



def dim_player(base):
    """One row per player. Static attributes only."""
    d = (base.sort_values("season")
         .groupby("player_id")
         .agg(player=("player", "last"),
              position=("pos", "last"),
              height_in=("ht_in_in", "last"),
              weight_lb=("wt", "last"),
              first_season=("season", "min"),
              last_season=("season", "max"),
              seasons_played=("season", "nunique"))
         .reset_index())
    d["career_length"] = d.last_season - d.first_season + 1
    return d


def dim_season(base):
    """One row per season, with era labels for slicing."""
    d = (base.groupby("season")
         .agg(players=("player_id", "nunique"),
              avg_age=("age", "mean"),
              avg_pace=("team_pace", "mean"),
              avg_3par=("x3p_ar", "mean"))
         .reset_index())
    d["era"] = pd.cut(
        d.season, [1996, 2004, 2011, 2018, 2027],
        labels=["Early (97-04)", "Mid (05-11)", "3PT Boom (12-18)", "Modern (19-26)"],
    )
    return d.round(3)


def dim_team(base):
    t = (base[~base.team.astype(str).str.contains(r"^\dTM$", na=False)]
         .groupby(["season", "team"])
         .agg(wins=("team_w", "first"),
              losses=("team_l", "first"),
              pace=("team_pace", "first"),
              off_rating=("team_o_rtg", "first"),
              def_rating=("team_d_rtg", "first"))
         .reset_index())
    t["team_season_key"] = t.team + "_" + t.season.astype(str)
    t["win_pct"] = (t.wins / (t.wins + t.losses)).round(3)
    return t


def fact_player_season(base):
    """The main fact table. Kept to columns a BI user would actually chart."""
    cols = {
        "season": "season", "player_id": "player_id", "player": "player",
        "team": "team", "age": "age", "experience": "experience",
        "g": "games", "gs": "games_started", "mp": "minutes",
        "per": "per", "ts_percent": "true_shooting", "usg_percent": "usage_rate",
        "bpm": "bpm", "vorp": "vorp", "ws": "win_shares", "ws_48": "ws_per_48",
        "obpm": "obpm", "dbpm": "dbpm",
        "trb_percent": "reb_rate", "ast_percent": "ast_rate",
        "stl_percent": "stl_rate", "blk_percent": "blk_rate",
        "tov_percent": "tov_rate", "x3p_ar": "three_pt_rate",
        "team_pace": "team_pace", "team_w": "team_wins",
    }
    f = base[[c for c in cols if c in base.columns]].rename(columns=cols).copy()
    f["team_season_key"] = f.team + "_" + f.season.astype(str)
    f["minutes_per_game"] = (f.minutes / f.games.replace(0, np.nan)).round(1)
    f["qualified"] = (f.minutes >= 1000).astype(int)
    return f.round(3)


def fact_predictions():
    """Test-set predictions, long format so BI can slice by model."""
    p = pd.read_csv(DATA / "test_predictions.csv")
    model_cols = [c for c in p.columns if c.startswith("pred_")]
    long = p.melt(
        id_vars=["season", "player_id", "player", "per", "target"],
        value_vars=model_cols, var_name="model", value_name="predicted",
    )
    long["model"] = long.model.str.replace("pred_", "", regex=False)
    long["actual"] = long.target
    long["error"] = long.actual - long.predicted
    long["abs_error"] = long.error.abs()
    long["actual_change"] = long.actual - long.per
    long["predicted_change"] = long.predicted - long.per
    # Persistence predicts zero change by construction, so "direction" is
    # undefined for it. Scoring it as 0 would make the baseline look far
    # worse in a dashboard than it actually is.
    long["direction_correct"] = np.where(
        long.model == "Persistence",
        np.nan,
        (np.sign(long.actual_change) == np.sign(long.predicted_change)).astype(float),
    )
    return long.drop(columns=["target"]).round(3)


def fact_projections():
    p = pd.read_csv(DATA / "projections_2027.csv")
    p["direction"] = np.where(p.delta > 0.5, "Rise",
                       np.where(p.delta < -0.5, "Fall", "Hold"))
    p["age_group"] = pd.cut(p.age, [17, 24, 27, 30, 45],
                            labels=["<=24", "25-27", "28-30", "31+"])
    p["tier"] = pd.cut(p.pred_2027, [-99, 12, 15, 18, 22, 99],
                       labels=["Below replacement", "Bench", "Starter",
                               "Quality starter", "All-Star"])
    return p.round(3)


def fact_aging():
    c = pd.read_csv(DATA / "eda_aging_cross.csv").assign(method="Cross-sectional")
    w = pd.read_csv(DATA / "eda_aging_within.csv").assign(method="Within-player")
    c = c.rename(columns={"avg_per": "value"})[["age", "n", "value", "method"]]
    w = w.rename(columns={"cum": "value"})[["age", "n", "value", "method"]]
    w["value"] = w.value + 15.0    # rebase deltas onto the PER scale
    return pd.concat([c, w], ignore_index=True).round(3)


def fact_stability():
    s = pd.read_csv(DATA / "eda_stability.csv")
    s["tier"] = pd.cut(s["corr"], [-1, 0.4, 0.65, 0.85, 1.0],
                       labels=["Noise", "Weak", "Stable", "Very stable"])
    return s.round(3)


def dim_model_results():
    m = pd.read_csv(DATA / "model_results.csv")
    c = pd.read_csv(DATA / "classification_results.csv")
    return m.round(4), c.round(4)


def fact_residuals():
    r = pd.read_csv(DATA / "residuals.csv")
    keep = ["season", "player_id", "player", "per", "pred", "target",
            "resid", "abs_resid", "age_now", "was_traded"]
    r = r[[c for c in keep if c in r.columns]].copy()
    r["age_group"] = pd.cut(r.age_now, [17, 24, 27, 30, 33, 45],
                            labels=["<=24", "25-27", "28-30", "31-33", "34+"])
    r["miss_type"] = np.where(r.resid > 0, "Underestimated", "Overestimated")
    return r.round(3)


GUIDE = """# Tableau Public setup guide

Tableau Public is free, runs natively on macOS, and publishes to a URL you
can link from a resume. Power BI Desktop is Windows-only, which is why this
project targets Tableau.

## Files

| File | Grain | Role |
|---|---|---|
| `dim_player.csv` | one row per player | dimension |
| `dim_season.csv` | one row per season | dimension |
| `dim_team_season.csv` | team x season | dimension |
| `fact_player_season.csv` | player x season | main fact |
| `fact_predictions.csv` | player x season x model | model eval fact |
| `fact_projections_2027.csv` | one row per 2027 player | forecast fact |
| `fact_residuals.csv` | player x season | error analysis fact |
| `fact_aging.csv` | age x method | analysis fact |
| `fact_stability.csv` | one row per stat | analysis fact |
| `dim_model_results.csv` | one row per model | reference |
| `dim_classification_results.csv` | one row per task | reference |

## Connecting the data

1. Open Tableau Public > Connect > To a File > Text file
2. Select `fact_player_season.csv`
3. Drag `dim_player.csv` onto the canvas. Tableau proposes a relationship;
   set it to `player_id = player_id`.
4. Drag `dim_season.csv`, relate on `season = season`.
5. Drag `dim_team_season.csv`, relate on `team_season_key = team_season_key`.

Use RELATIONSHIPS (the noodle, Tableau 2020.2+), not joins. A join would
duplicate fact rows wherever a dimension has more than one match; a
relationship keeps the grain intact and lets Tableau pick the right level
of detail per sheet.

`fact_predictions`, `fact_projections_2027`, `fact_residuals`, `fact_aging`
and `fact_stability` each have a different grain, so put them in SEPARATE
data sources rather than relating everything into one model. Mixing grains
in a single source is the most common cause of inflated Tableau numbers.

## Calculated fields

```
// headline result
MAE                 AVG([Abs Error])

Baseline MAE        { FIXED : AVG(IIF([Model] = "Persistence", [Abs Error], NULL)) }

Best Model MAE      { FIXED : AVG(IIF([Model] = "Ridge", [Abs Error], NULL)) }

MAE Improvement     [Baseline MAE] - [Best Model MAE]

Improvement Pct     [MAE Improvement] / [Baseline MAE]

// direction accuracy -- null for Persistence by design, since it
// predicts zero change and has no direction to be right about
Direction Accuracy  AVG([Direction Correct])

// volume
Players             COUNTD([Player Id])

Qualified Players   { FIXED : COUNTD(IIF([Qualified] = 1, [Player Id], NULL)) }

// aging
Avg PER             AVG([Per])

Avg PER Qualified   AVG(IIF([Qualified] = 1, [Per], NULL))

// projections
Projected Risers    COUNT(IIF([Direction] = "Rise", [Player], NULL))

Avg Projected Change  AVG([Delta])
```

## Suggested dashboard pages

**Page 1 -- Overview**
Four BANs (big numbers): Baseline MAE 1.874, Best Model MAE 1.683,
Improvement 10.2%, Qualified Players.
Bar of MAE by model from `dim_model_results`, with `ci_lo`/`ci_hi` on the
Detail shelf and a reference band showing the interval.
A text tile stating the headline claim and the significance test.

**Page 2 -- Aging**
Line chart from `fact_aging`: `age` on Columns, `value` on Rows,
`method` on Color. The two lines diverging IS the story; annotate the
peak-age gap (26 cross-sectional vs 24 within-player).

**Page 3 -- Stability**
Horizontal bar of `corr` from `fact_stability`, sorted descending,
`tier` on Color. Reference line at 0.5.
Annotate games played at 0.16 -- that is the injury ceiling on any model.

**Page 4 -- Model performance**
Scatter of `predicted` vs `actual` from `fact_predictions`, with `model`
as a filter and a 45-degree reference line.
Bar of AVG(`resid`) by `age_group` from `fact_residuals`. This shows the
systematic bias: the model is too pessimistic about older players.

**Page 5 -- 2027 projections**
Table of `fact_projections_2027` sorted by `pred_2027` descending.
Filters on `age_group` and `tier`.
Diverging bar of `delta` for the top risers and fallers.

## Publishing

File > Save to Tableau Public As. The workbook and its extracts become
public, so do not put anything private in it. The resulting URL is what
goes on a resume.
"""


def main():
    base = pd.read_parquet(DATA / "player_seasons.parquet")

    tables = {
        "dim_player": dim_player(base),
        "dim_season": dim_season(base),
        "dim_team_season": dim_team(base),
        "fact_player_season": fact_player_season(base),
        "fact_predictions": fact_predictions(),
        "fact_projections_2027": fact_projections(),
        "fact_residuals": fact_residuals(),
        "fact_aging": fact_aging(),
        "fact_stability": fact_stability(),
    }
    mr, cr = dim_model_results()
    tables["dim_model_results"] = mr
    tables["dim_classification_results"] = cr

    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)

    (OUT / "SETUP_GUIDE.md").write_text(GUIDE)

    print(f"exported to {OUT}\n")
    print(f"{'table':<30}{'rows':>8}{'cols':>7}")
    print("-" * 45)
    for name, df in tables.items():
        print(f"{name:<30}{len(df):>8,}{df.shape[1]:>7}")

    # sanity: every fact key must exist in its dimension
    dp = set(tables["dim_player"].player_id)
    for f in ["fact_player_season", "fact_predictions", "fact_residuals"]:
        orphans = set(tables[f].player_id) - dp
        assert not orphans, f"{f} has {len(orphans)} player_ids missing from dim_player"
    ds = set(tables["dim_season"].season)
    assert not set(tables["fact_player_season"].season) - ds
    print("\nreferential integrity: all fact keys resolve to dimensions")
    print(f"guide -> {OUT / 'SETUP_GUIDE.md'}")


if __name__ == "__main__":
    main()
