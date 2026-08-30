"""
Phase 5 — breakout classification, residual analysis, and 2027 projections.

Three pieces:

5a  BREAKOUT / DECLINE CLASSIFICATION
    Regression buries the interesting cases: a model that shaves 10% off
    MAE does it by being slightly better on everyone, not by calling the
    guys who jump. Framing it as classification asks the question people
    actually care about, and it sidesteps the persistence problem entirely
    (persistence predicts zero change, so it can never flag a breakout).

5b  RESIDUAL ANALYSIS
    Where does the model systematically fail? Phase 1 predicted it would be
    optimistic about declining veterans, because they are the ones dropped
    by the attrition filter. This checks that.

5c  2027 PROJECTIONS
    Refit on everything through 2025, apply to 2026, produce real forecasts.

Run:  python src/phase5_analysis.py
"""

from pathlib import Path

from paths import PLOTS, PROCESSED as DATA, RAW, TABLEAU, DB

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


TRAIN_END, VAL_END = 2018, 2021
TARGET_MIN_MP = 500
THRESHOLD = 2.0          # PER points that count as a real move
SEED = 42
N_BOOT = 2000
ID_COLS = ["season", "player_id", "player", "team", "pos", "target", "next_mp",
           "d_per_actual", "is_breakout", "is_decline"]

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------ data
def load():
    f = pd.read_parquet(DATA / "features.parquet")
    b = pd.read_parquet(DATA / "player_seasons.parquet")
    nxt = b[["season", "player_id", "mp"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"mp": "next_mp"})
    f = f.merge(nxt, on=["season", "player_id"], how="left")
    f = f[f.next_mp >= TARGET_MIN_MP].copy()
    f["was_traded"] = f["was_traded"].astype(int)

    f["d_per_actual"] = f["target"] - f["per"]
    f["is_breakout"] = (f.d_per_actual >= THRESHOLD).astype(int)
    f["is_decline"] = (f.d_per_actual <= -THRESHOLD).astype(int)
    return f, b


def split(df):
    return (df[df.season <= TRAIN_END],
            df[(df.season > TRAIN_END) & (df.season <= VAL_END)],
            df[df.season > VAL_END])


# -------------------------------------------------------------------- 5a
def classify(df, feats, label):
    """Binary classifier for one direction of move.

    Baseline is the training base rate: guessing the class prior for every
    player. AUC 0.5 means the model has learned nothing beyond that.
    """
    tr, va, te = split(df)
    Xtr, ytr = tr[feats].to_numpy(float), tr[label].to_numpy()
    Xva, yva = va[feats].to_numpy(float), va[label].to_numpy()
    Xte, yte = te[feats].to_numpy(float), te[label].to_numpy()

    clf = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=15,
        min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=2.0,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    p = clf.predict_proba(Xte)[:, 1]

    base_rate = ytr.mean()
    auc = roc_auc_score(yte, p)
    ap = average_precision_score(yte, p)

    # bootstrap CI on AUC
    idx = rng.integers(0, len(yte), size=(N_BOOT, len(yte)))
    draws = []
    for i in idx[:500]:                       # 500 is plenty for a CI here
        if yte[i].sum() in (0, len(i)):
            continue
        draws.append(roc_auc_score(yte[i], p[i]))
    lo, hi = np.percentile(draws, [2.5, 97.5])

    return {
        "model": clf, "proba": p, "y": yte, "test": te,
        "auc": auc, "auc_ci": (lo, hi), "ap": ap,
        "base_rate": base_rate, "test_rate": yte.mean(),
        "lift": ap / yte.mean(),
        "brier": brier_score_loss(yte, p),
    }


# -------------------------------------------------------------------- 5b
def residual_analysis(te, pred):
    d = te.copy()
    d["pred"] = pred
    d["resid"] = d.target - d.pred          # positive = model UNDER-predicted
    d["abs_resid"] = d.resid.abs()
    d["age_now"] = d.years_past_peak + 24
    return d


def bias_table(d, by, bins=None, labels=None):
    x = d.copy()
    if bins is not None:
        x["_bin"] = pd.cut(x[by], bins, labels=labels)
    else:
        x["_bin"] = x[by]
    g = x.groupby("_bin", observed=True).agg(
        n=("resid", "size"),
        mean_resid=("resid", "mean"),
        mean_abs=("abs_resid", "mean"),
    ).round(3)
    return g


# -------------------------------------------------------------------- 5c
def project_2027(df, base, feats):
    """Refit on everything through 2025, then predict 2026 -> 2027.

    Uses Ridge because Phase 4 found it tied with the tree models while
    being simpler and more stable to extrapolate with.
    """
    # rebuild season-2026 feature rows the same way Phase 1 and 3 did
    from build_model_table import add_history, add_curves
    from build_features import (add_career_context, add_age_features,
                                add_trend_features, add_role_features)

    b = base.sort_values(["player_id", "season"]).reset_index(drop=True)
    b = add_history(b)
    b = add_curves(b)
    cur = b[(b.season == 2026) & (b.mp >= 1000)].copy()
    cur["target"] = np.nan

    cur = add_career_context(cur, base)
    cur = add_age_features(cur)
    cur = add_trend_features(cur)
    cur = add_role_features(cur)
    cur["was_traded"] = cur["was_traded"].astype(int)

    missing = [c for c in feats if c not in cur.columns]
    for c in missing:
        cur[c] = np.nan

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0, random_state=SEED)),
    ])
    model.fit(df[feats].to_numpy(float), df["target"].to_numpy(float))

    cur["pred_2027"] = model.predict(cur[feats].to_numpy(float))
    cur["delta"] = cur.pred_2027 - cur.per
    out = cur[["player", "team", "age", "mp", "per", "pred_2027", "delta"]]
    return out.sort_values("pred_2027", ascending=False).reset_index(drop=True), missing


# ----------------------------------------------------------------- plots
def plot_classification(res_b, res_d):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    for a, r, name, color in [
        (ax[0], res_b, "Breakout", "seagreen"),
        (ax[1], res_d, "Decline", "indianred"),
    ]:
        prec, rec, _ = precision_recall_curve(r["y"], r["proba"])
        a.plot(rec, prec, color=color, lw=2,
               label=f"model  AP={r['ap']:.3f}")
        a.axhline(r["test_rate"], ls="--", color="grey", lw=1.2,
                  label=f"base rate={r['test_rate']:.3f}")
        a.set_xlabel("Recall")
        a.set_ylabel("Precision")
        a.set_title(f"{name}: AUC {r['auc']:.3f} "
                    f"[{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}]")
        a.legend()
        a.set_ylim(0, 1)

    fig.suptitle(f"Identifying moves of >= {THRESHOLD} PER points", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "breakout_classification.png", dpi=140)
    plt.close(fig)


def plot_calibration(res_b, res_d):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r, name, color in [(res_b, "Breakout", "seagreen"),
                           (res_d, "Decline", "indianred")]:
        pt, pp = calibration_curve(r["y"], r["proba"], n_bins=8, strategy="quantile")
        ax.plot(pp, pt, marker="o", color=color,
                label=f"{name}  Brier={r['brier']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration: are the probabilities honest?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "calibration.png", dpi=140)
    plt.close(fig)


def plot_residual_bias(d):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    g = bias_table(d, "age_now", [18, 24, 27, 30, 33, 45],
                   ["<=24", "25-27", "28-30", "31-33", "34+"])
    ax[0].bar(g.index.astype(str), g.mean_resid,
              color=["seagreen" if v > 0 else "indianred" for v in g.mean_resid])
    ax[0].axhline(0, lw=1, color="grey")
    ax[0].set_ylabel("mean residual (actual - predicted)")
    ax[0].set_title("Bias by age\npositive = model too pessimistic")
    for i, (v, n) in enumerate(zip(g.mean_resid, g.n)):
        ax[0].text(i, v, f"n={n}", ha="center",
                   va="bottom" if v > 0 else "top", fontsize=8)

    d2 = d.copy()
    d2["per_q"] = pd.qcut(d2.per, 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    g2 = bias_table(d2, "per_q")
    ax[1].bar(g2.index.astype(str), g2.mean_resid,
              color=["seagreen" if v > 0 else "indianred" for v in g2.mean_resid])
    ax[1].axhline(0, lw=1, color="grey")
    ax[1].set_title("Bias by current PER\nregression to the mean, overdone")

    g3 = bias_table(d, "was_traded")
    ax[2].bar(["stayed", "traded"], g3.mean_resid,
              color=["steelblue", "darkorange"])
    ax[2].axhline(0, lw=1, color="grey")
    ax[2].set_title("Bias by traded status")

    fig.suptitle("Systematic error: where the model is wrong on purpose",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "residual_bias.png", dpi=140)
    plt.close(fig)


def plot_projections(proj, k=15):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    top = proj.head(k).iloc[::-1]
    ax[0].barh(top.player, top.pred_2027, color="steelblue")
    ax[0].set_xlabel("Projected 2027 PER")
    ax[0].set_title(f"Top {k} projected players, 2027")
    ax[0].set_xlim(15, top.pred_2027.max() + 2)

    movers = pd.concat([proj.nlargest(8, "delta"), proj.nsmallest(8, "delta")])
    movers = movers.sort_values("delta")
    ax[1].barh(movers.player, movers.delta,
               color=["indianred" if v < 0 else "seagreen" for v in movers.delta])
    ax[1].axvline(0, lw=1, color="grey")
    ax[1].set_xlabel("Projected change in PER, 2026 to 2027")
    ax[1].set_title("Biggest projected risers and fallers")

    fig.tight_layout()
    fig.savefig(PLOTS / "projections_2027.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    df, base = load()
    feats = [c for c in df.columns if c not in ID_COLS]

    print("=" * 66)
    print("PHASE 5 — CLASSIFICATION, RESIDUALS, PROJECTIONS")
    print("=" * 66)
    print(f"rows {len(df):,}   features {len(feats)}   threshold ±{THRESHOLD} PER")

    # ---- 5a -----------------------------------------------------------
    print("\n--- 5a  BREAKOUT / DECLINE CLASSIFICATION ---")
    res_b = classify(df, feats, "is_breakout")
    res_d = classify(df, feats, "is_decline")

    for r, name in [(res_b, "BREAKOUT"), (res_d, "DECLINE")]:
        print(f"\n{name}  (>= {THRESHOLD} PER move)")
        print(f"  base rate in test      {r['test_rate']:.3f}")
        print(f"  ROC AUC                {r['auc']:.3f}  "
              f"95% CI [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]")
        print(f"  avg precision          {r['ap']:.3f}  "
              f"({r['lift']:.2f}x the base rate)")
        print(f"  Brier score            {r['brier']:.3f}")

    plot_classification(res_b, res_d)
    plot_calibration(res_b, res_d)

    # top flagged players
    tb = res_b["test"].copy()
    tb["p_breakout"] = res_b["proba"]
    print("\n  highest breakout probabilities in test set:")
    print(tb.nlargest(10, "p_breakout")[
        ["player", "season", "per", "target", "p_breakout", "is_breakout"]
    ].round(3).to_string(index=False))

    # ---- 5b -----------------------------------------------------------
    print("\n--- 5b  RESIDUAL ANALYSIS ---")
    tr, va, te = split(df)
    model = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler()),
                      ("model", Ridge(alpha=10.0, random_state=SEED))])
    model.fit(pd.concat([tr, va])[feats].to_numpy(float),
              pd.concat([tr, va])["target"].to_numpy(float))
    pred = model.predict(te[feats].to_numpy(float))
    d = residual_analysis(te, pred)

    print("\nbias by age (positive = model too pessimistic):")
    print(bias_table(d, "age_now", [18, 24, 27, 30, 33, 45],
                     ["<=24", "25-27", "28-30", "31-33", "34+"]).to_string())
    d2 = d.copy()
    d2["per_q"] = pd.qcut(d2.per, 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    print("\nbias by current PER quintile:")
    print(bias_table(d2, "per_q").to_string())

    print("\nworst misses:")
    print(d.nlargest(8, "abs_resid")[
        ["player", "season", "per", "pred", "target", "resid"]
    ].round(2).to_string(index=False))

    plot_residual_bias(d)
    d.to_csv(DATA / "residuals.csv", index=False)

    # ---- 5c -----------------------------------------------------------
    print("\n--- 5c  2027 PROJECTIONS ---")
    proj, missing = project_2027(df, base, feats)
    if missing:
        print(f"  note: {len(missing)} features unavailable for 2026, imputed")
    print(f"  players projected: {len(proj)}")
    print("\ntop 15 projected 2027 PER:")
    print(proj.head(15).round(2).to_string(index=False))
    print("\nbiggest projected risers:")
    print(proj.nlargest(8, "delta").round(2).to_string(index=False))
    print("\nbiggest projected fallers:")
    print(proj.nsmallest(8, "delta").round(2).to_string(index=False))

    plot_projections(proj)
    proj.to_csv(DATA / "projections_2027.csv", index=False)

    pd.DataFrame([
        {"task": "breakout", "auc": res_b["auc"], "ap": res_b["ap"],
         "base_rate": res_b["test_rate"], "brier": res_b["brier"]},
        {"task": "decline", "auc": res_d["auc"], "ap": res_d["ap"],
         "base_rate": res_d["test_rate"], "brier": res_d["brier"]},
    ]).to_csv(DATA / "classification_results.csv", index=False)

    print(f"\nplots -> {PLOTS}")


if __name__ == "__main__":
    main()
