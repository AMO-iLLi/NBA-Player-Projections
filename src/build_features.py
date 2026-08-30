"""
Phase 3 — feature engineering and audit.

Not "add more columns". Three jobs:

  3a  Encode what Phase 2 actually found. The aging curve peaks at 24, not
      26, so the age feature should measure distance past 24. Deviation
      from a player's own career average should predict a bounce back.
  3b  Remove redundancy. Phase 2 left exact duplicate columns behind.
  3c  Audit for leakage, because a feature that quietly contains the future
      is the one bug that makes every downstream number meaningless.

Run:  python src/build_features.py
"""

from pathlib import Path

from paths import PLOTS, PROCESSED as DATA, RAW, TABLEAU, DB

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PEAK_AGE = 24          # from Phase 2.1, within-player curve
REDUNDANT_CUTOFF = 0.98
ID_COLS = ["season", "player_id", "player", "team", "pos", "target"]

# stats to build career-context features on
CAREER_STATS = ["per", "usg_percent", "ts_percent", "bpm", "mp"]


# ------------------------------------------------------------------- 3a
def add_career_context(model: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Career averages and deviation-from-career, as of season N.

    The expanding mean is shifted by one season FIRST, so a player's career
    average never includes the season it is attached to. Without the shift,
    `per` would be baked into `career_per`, and `dev_from_career` would be
    a rescaled copy of the current season rather than a deviation from it.
    """
    b = base.sort_values(["player_id", "season"]).copy()
    g = b.groupby("player_id", sort=False)

    made = ["player_id", "season"]
    for stat in CAREER_STATS:
        prior = g[stat].shift(1)                       # exclude current season
        b[f"career_{stat}"] = (
            prior.groupby(b.player_id).expanding().mean()
            .reset_index(level=0, drop=True)
        )
        made.append(f"career_{stat}")

    out = model.merge(b[made], on=["player_id", "season"], how="left")

    # regression to the mean: how far is this season from the player's norm?
    for stat in CAREER_STATS:
        out[f"dev_{stat}"] = out[stat] - out[f"career_{stat}"]

    # career best and how far below it the player currently sits
    b["career_best_per"] = (
        g["per"].shift(1).groupby(b.player_id).expanding().max()
        .reset_index(level=0, drop=True)
    )
    out = out.merge(b[["player_id", "season", "career_best_per"]],
                    on=["player_id", "season"], how="left")
    out["below_career_best"] = out["career_best_per"] - out["per"]

    return out


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Age encoded as distance from the empirical peak, not raw age.

    Phase 2.1 found improvement stops at 24 and every year after is
    negative on average. A raw `age` column asks a linear model to discover
    that; these columns hand it over directly.
    """
    df = df.copy()
    df["years_past_peak"] = (df["age"] - PEAK_AGE).clip(lower=0)
    df["years_to_peak"] = (PEAK_AGE - df["age"]).clip(lower=0)
    df["is_past_peak"] = (df["age"] > PEAK_AGE).astype(int)
    # decline accelerates rather than running flat
    df["decline_pressure"] = df["years_past_peak"] ** 1.5
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Direction of travel, not just level.

    Two players at PER 18 are different bets if one arrived from 14 and the
    other from 22. per_delta1 already captures one step; this adds the
    two-step slope and a consistency flag.
    """
    df = df.copy()
    df["per_slope2"] = (df["per"] - df["per_lag2"]) / 2.0
    df["improving_2yr"] = (
        (df["per_delta1"] > 0) & (df["per_delta2"] > 0)
    ).astype(int)
    df["declining_2yr"] = (
        (df["per_delta1"] < 0) & (df["per_delta2"] < 0)
    ).astype(int)
    # minutes trend: a role shrinking is a leading indicator
    df["mp_trend"] = df["mp"] - df["mp_roll3"]
    return df


def add_role_features(df: pd.DataFrame) -> pd.DataFrame:
    """Role volatility and workload.

    Phase 2.3 showed role stats (trb%, ast%, usage) are the most stable
    things in the dataset, so a player whose role IS shifting is unusual
    and worth flagging.
    """
    df = df.copy()
    df["usg_shift"] = (df["usg_percent"] - df["usg_percent_lag1"]).abs()
    df["starter_share"] = df["gs"] / df["g"].replace(0, np.nan)
    df["mp_per_g"] = df["mp"] / df["g"].replace(0, np.nan)
    return df


# ------------------------------------------------------------------- 3b
def drop_redundant(df: pd.DataFrame, cutoff=REDUNDANT_CUTOFF):
    """Drop one of each near-perfectly-correlated pair.

    g and g_share are the same column with a different scale. Trees do not
    care, but a linear model does, and duplicated columns split feature
    importance between twins and make the ranking unreadable.
    """
    num = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ID_COLS if c in df.columns], errors="ignore")
    corr = num.corr().abs().to_numpy(copy=True)
    cols = list(num.columns)
    np.fill_diagonal(corr, 0.0)

    dropped = {}
    keep = set(cols)
    iu = np.triu_indices_from(corr, k=1)
    order = np.argsort(-corr[iu])
    for idx in order:
        i, j = iu[0][idx], iu[1][idx]
        r = corr[i, j]
        if r < cutoff:
            break
        a, bcol = cols[i], cols[j]
        if a in keep and bcol in keep:
            # keep whichever correlates more with the target
            drop = bcol if abs(df[a].corr(df.target)) >= abs(df[bcol].corr(df.target)) else a
            keep.discard(drop)
            dropped[drop] = f"r={r:.3f} with {a if drop == bcol else bcol}"
    return df.drop(columns=list(dropped)), dropped


# ------------------------------------------------------------------- 3c
def audit_leakage(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Rank features by |correlation with target|.

    This is a leakage smoke test, not a feature ranking. Persistence gives
    corr(per, target) = 0.777, so anything materially above that is
    suspicious: no legitimate season-N feature should predict season N+1
    better than season N's own PER does.
    """
    num = df[feature_cols].select_dtypes(include=[np.number])
    cor = num.corrwith(df.target)
    out = (pd.DataFrame({"feature": cor.index, "corr": cor.values})
           .assign(abs_corr=lambda d: d["corr"].abs())
           .sort_values("abs_corr", ascending=False)
           .reset_index(drop=True))
    return out


# ---------------------------------------------------------------- plots
def plot_regression_to_mean(df):
    d = df.dropna(subset=["dev_per"]).copy()
    d["bounce"] = d.target - d.per
    d["q"] = pd.qcut(d.dev_per, 10, labels=False)
    g = d.groupby("q").agg(dev=("dev_per", "mean"), bounce=("bounce", "mean"),
                           n=("bounce", "size"))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(d.dev_per, d.bounce, s=4, alpha=0.10,
                  color="steelblue", edgecolors="none")
    m, c = np.polyfit(d.dev_per, d.bounce, 1)
    xs = np.linspace(d.dev_per.min(), d.dev_per.max(), 50)
    ax[0].plot(xs, m * xs + c, color="darkred", lw=2)
    ax[0].axhline(0, ls="--", lw=1, color="grey")
    ax[0].axvline(0, ls="--", lw=1, color="grey")
    ax[0].set_xlabel("PER this season minus career average")
    ax[0].set_ylabel("PER change next season")
    ax[0].set_title(f"Regression to the mean   r = {d.dev_per.corr(d.bounce):.3f}")

    ax[1].bar(range(len(g)), g.bounce,
              color=["seagreen" if v > 0 else "indianred" for v in g.bounce])
    ax[1].axhline(0, lw=1, color="grey")
    ax[1].set_xticks(range(len(g)))
    ax[1].set_xticklabels([f"{v:+.1f}" for v in g.dev], rotation=45, fontsize=8)
    ax[1].set_xlabel("Deviation from career average (decile mean)")
    ax[1].set_ylabel("Mean PER change next season")
    ax[1].set_title("Overperformers fall back, underperformers rebound")

    fig.suptitle("Career deviation predicts next-season direction", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "regression_to_mean.png", dpi=140)
    plt.close(fig)


def plot_age_encoding(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ages = np.arange(19, 40)
    ax.plot(ages, (ages - PEAK_AGE).clip(min=0), marker="o",
            label="years_past_peak")
    ax.plot(ages, ((ages - PEAK_AGE).clip(min=0)) ** 1.5, marker="s",
            label="decline_pressure")
    ax.axvline(PEAK_AGE, ls="--", color="grey", lw=1)
    ax.text(PEAK_AGE + 0.2, 40, f"empirical peak = {PEAK_AGE}", fontsize=9)
    ax.set_xlabel("Age")
    ax.set_ylabel("Feature value")
    ax.set_title("Age encoded as distance past the empirical peak")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "age_encoding.png", dpi=140)
    plt.close(fig)


def plot_target_corr(audit, n=25):
    top = audit.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top.feature, top["corr"],
            color=["steelblue" if v > 0 else "indianred" for v in top["corr"]])
    ax.axvline(0, lw=1, color="grey")
    ax.axvline(0.777, ls="--", lw=1.2, color="darkred")
    ax.text(0.78, 0.4, "persistence (0.777)", rotation=90,
            fontsize=8, color="darkred", va="bottom")
    ax.set_xlabel("corr with next-season PER")
    ax.set_title(f"Top {n} features by correlation with target")
    fig.tight_layout()
    fig.savefig(PLOTS / "feature_target_corr.png", dpi=140)
    plt.close(fig)


def plot_redundancy(df, feature_cols):
    core = [c for c in [
        "per", "bpm", "vorp", "ws", "ws_48", "obpm", "dbpm", "ows", "dws",
        "usg_percent", "ts_percent", "mp", "g", "age", "experience",
        "career_per", "dev_per", "below_career_best", "per_slope2",
    ] if c in feature_cols]
    c = df[core].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(core)))
    ax.set_xticklabels(core, rotation=90, fontsize=8)
    ax.set_yticks(range(len(core)))
    ax.set_yticklabels(core, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    ax.set_title("Feature redundancy among core metrics")
    fig.tight_layout()
    fig.savefig(PLOTS / "feature_redundancy.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    model = pd.read_parquet(DATA / "model_table.parquet")
    base = pd.read_parquet(DATA / "player_seasons.parquet")
    n_before = model.shape[1]

    df = add_career_context(model, base)
    df = add_age_features(df)
    df = add_trend_features(df)
    df = add_role_features(df)
    n_added = df.shape[1] - n_before

    df, dropped = drop_redundant(df)

    feature_cols = [c for c in df.columns if c not in ID_COLS]
    audit = audit_leakage(df, feature_cols)

    # hard leakage gate: nothing may beat persistence
    persistence = abs(df["per"].corr(df.target))
    suspect = audit[audit.abs_corr > persistence + 0.02]
    suspect = suspect[suspect.feature != "per"]

    plot_regression_to_mean(df)
    plot_age_encoding(df)
    plot_target_corr(audit)
    plot_redundancy(df, feature_cols)

    df.to_parquet(DATA / "features.parquet", index=False)
    audit.to_csv(DATA / "feature_audit.csv", index=False)

    print(f"features in       {n_before - len(ID_COLS)}")
    print(f"engineered        +{n_added}")
    print(f"dropped redundant -{len(dropped)}")
    print(f"features out      {len(feature_cols)}")
    print(f"rows              {len(df):,}\n")

    print("=== DROPPED AS REDUNDANT ===")
    for k, v in dropped.items():
        print(f"  {k:<28} {v}")

    print("\n=== LEAKAGE GATE ===")
    print(f"persistence corr(per, target) = {persistence:.3f}")
    if len(suspect):
        print("SUSPECT features beating persistence:")
        print(suspect.to_string(index=False))
    else:
        print("clear: no feature beats persistence")

    print("\n=== TOP 15 BY |CORR| WITH TARGET ===")
    print(audit.head(15).round(3).to_string(index=False))

    print("\n=== NEW FEATURES, RANKED ===")
    new = ["dev_per", "career_per", "below_career_best", "per_slope2",
           "years_past_peak", "decline_pressure", "mp_trend", "usg_shift",
           "improving_2yr", "declining_2yr", "starter_share", "mp_per_g"]
    print(audit[audit.feature.isin(new)].round(3).to_string(index=False))

    print(f"\nplots -> {PLOTS}")


if __name__ == "__main__":
    main()
