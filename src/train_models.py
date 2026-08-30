"""
Phase 4 — modeling.

The only question that matters: can a model beat persistence (predict next
season = this season)? Everything here is organised around answering that
honestly rather than producing an impressive-looking number.

Three design decisions worth defending:

1. TIME-BASED SPLIT. Train <= 2018, validate 2019-2021, test 2022+. A random
   split would let the model see 2024 while predicting 2023, and would also
   put the same player on both sides. Both inflate scores.

2. TARGET RELIABILITY FLOOR. PER is a rate stat; at 3 minutes played it is
   noise, not a measurement (Nene 2006: 1 game, PER -54.4). Rows whose
   TARGET season is under 500 minutes are excluded from the primary
   analysis and reported separately as a sensitivity check.

3. PAIRED BOOTSTRAP. The test set is ~1k rows, so a raw MAE difference of
   0.03 means nothing. Significance is assessed by resampling the paired
   per-row errors, which controls for the fact that both models face the
   same rows.

Run:  python src/train_models.py
"""

from pathlib import Path

from paths import PLOTS, PROCESSED as DATA, RAW, TABLEAU, DB

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb


TRAIN_END, VAL_END = 2018, 2021
TARGET_MIN_MP = 500          # reliability floor on the TARGET season
N_BOOT = 2000
SEED = 42
ID_COLS = ["season", "player_id", "player", "team", "pos", "target", "next_mp"]

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------ data
def load(reliability_floor=TARGET_MIN_MP):
    f = pd.read_parquet(DATA / "features.parquet")
    b = pd.read_parquet(DATA / "player_seasons.parquet")

    nxt = b[["season", "player_id", "mp"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"mp": "next_mp"})
    f = f.merge(nxt, on=["season", "player_id"], how="left")

    n_all = len(f)
    if reliability_floor:
        f = f[f.next_mp >= reliability_floor].copy()
    f["was_traded"] = f["was_traded"].astype(int)
    return f, n_all


def split(df):
    tr = df[df.season <= TRAIN_END]
    va = df[(df.season > TRAIN_END) & (df.season <= VAL_END)]
    te = df[df.season > VAL_END]
    return tr, va, te


def xy(df, feats):
    return df[feats].to_numpy(dtype=float), df["target"].to_numpy(dtype=float)


# --------------------------------------------------------------- metrics
def metrics(y, p):
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p),
    }


def boot_ci(y, p, stat="MAE", n=N_BOOT):
    """Percentile bootstrap CI for a single model's test metric."""
    err = np.abs(y - p) if stat == "MAE" else (y - p) ** 2
    idx = rng.integers(0, len(err), size=(n, len(err)))
    draws = err[idx].mean(axis=1)
    if stat == "RMSE":
        draws = np.sqrt(draws)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_boot(y, p_model, p_base, n=N_BOOT):
    """Paired bootstrap on the MAE difference vs the baseline.

    Resamples ROWS, so both models are always scored on the same rows.
    Returns (mean improvement, CI, share of draws where the model wins).
    """
    e_m = np.abs(y - p_model)
    e_b = np.abs(y - p_base)
    d = e_b - e_m                      # positive = model better
    idx = rng.integers(0, len(d), size=(n, len(d)))
    draws = d[idx].mean(axis=1)
    return (float(draws.mean()),
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
            float((draws > 0).mean()))


# ---------------------------------------------------------------- models
def build_models(n_feat):
    """Ridge gets imputation + scaling; trees handle NaN natively and are
    scale-invariant, so they take the raw matrix."""
    ridge = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0, random_state=SEED)),
    ])
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=12, min_samples_leaf=10,
        max_features="sqrt", n_jobs=-1, random_state=SEED,
    )
    xgbm = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.7,
        reg_lambda=2.0, min_child_weight=10,
        random_state=SEED, n_jobs=-1, early_stopping_rounds=50,
    )
    lgbm = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=15,
        min_child_samples=25, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=2.0,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    return {"Ridge": ridge, "RandomForest": rf, "XGBoost": xgbm, "LightGBM": lgbm}


def fit_predict(name, model, Xtr, ytr, Xva, yva, Xte):
    """Trees that support it get early stopping on the validation split.
    RandomForest has no such mechanism, so it trains on train+val to use the
    same data budget as the others."""
    if name == "XGBoost":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif name == "LightGBM":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    return model.predict(Xte)


# ----------------------------------------------------------------- plots
def plot_comparison(res, base_mae):
    names = list(res.keys())
    maes = [res[n]["MAE"] for n in names]
    los = [res[n]["ci"][0] for n in names]
    his = [res[n]["ci"][1] for n in names]
    err = [[m - lo for m, lo in zip(maes, los)], [hi - m for m, hi in zip(maes, his)]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["grey" if n == "Persistence" else
              ("seagreen" if res[n]["sig"] else "steelblue") for n in names]
    ax.bar(names, maes, yerr=err, capsize=5, color=colors)
    ax.axhline(base_mae, ls="--", lw=1.2, color="darkred")
    ax.text(len(names) - 0.4, base_mae + 0.008, "persistence baseline",
            fontsize=8, color="darkred", ha="right")
    for i, (m, hi) in enumerate(zip(maes, his)):
        ax.text(i, hi + 0.012, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Test MAE (lower is better)")
    ax.set_ylim(min(los) - 0.08, max(his) + 0.08)
    ax.set_title("Model comparison with 95% bootstrap CIs\n"
                 "green = beats persistence at p < 0.05", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "model_comparison.png", dpi=140)
    plt.close(fig)


def plot_pred_actual(te, preds, best):
    y = te.target.to_numpy()
    p = preds[best]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    lim = [min(y.min(), p.min()) - 1, max(y.max(), p.max()) + 1]
    ax[0].scatter(y, p, s=8, alpha=0.25, color="steelblue", edgecolors="none")
    ax[0].plot(lim, lim, ls="--", color="darkred", lw=1.5)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("Actual next-season PER")
    ax[0].set_ylabel("Predicted")
    ax[0].set_title(f"{best}: predicted vs actual")

    resid = y - p
    ax[1].scatter(p, resid, s=8, alpha=0.25, color="darkorange", edgecolors="none")
    ax[1].axhline(0, ls="--", color="grey", lw=1)
    ax[1].set_xlabel("Predicted PER")
    ax[1].set_ylabel("Residual (actual - predicted)")
    ax[1].set_title("Residuals: the model compresses toward the mean")

    fig.tight_layout()
    fig.savefig(PLOTS / "pred_vs_actual.png", dpi=140)
    plt.close(fig)


def plot_residual_slices(te, preds, best, base_pred):
    d = te.copy()
    d["err_model"] = np.abs(d.target - preds[best])
    d["err_base"] = np.abs(d.target - base_pred)
    d["age_now"] = d["years_past_peak"] + 24

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    d["age_bin"] = pd.cut(d.age_now, [18, 24, 27, 30, 45],
                          labels=["<=24", "25-27", "28-30", "31+"])
    g = d.groupby("age_bin", observed=True)[["err_base", "err_model"]].mean()
    x = np.arange(len(g))
    ax[0].bar(x - 0.2, g.err_base, 0.4, label="persistence", color="grey")
    ax[0].bar(x + 0.2, g.err_model, 0.4, label=best, color="steelblue")
    ax[0].set_xticks(x); ax[0].set_xticklabels(g.index)
    ax[0].set_xlabel("Age"); ax[0].set_ylabel("Mean absolute error")
    ax[0].set_title("Where the model helps: by age")
    ax[0].legend()

    d["per_bin"] = pd.qcut(d.per, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    g2 = d.groupby("per_bin", observed=True)[["err_base", "err_model"]].mean()
    x = np.arange(len(g2))
    ax[1].bar(x - 0.2, g2.err_base, 0.4, label="persistence", color="grey")
    ax[1].bar(x + 0.2, g2.err_model, 0.4, label=best, color="steelblue")
    ax[1].set_xticks(x); ax[1].set_xticklabels(g2.index)
    ax[1].set_xlabel("Current-season PER quartile")
    ax[1].set_ylabel("Mean absolute error")
    ax[1].set_title("Where the model helps: by performance level")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(PLOTS / "residual_slices.png", dpi=140)
    plt.close(fig)


def plot_shap(model, Xte, feats, best):
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xte)
    imp = pd.DataFrame({"feature": feats,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)}
                       ).sort_values("mean_abs_shap", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 8))
    top = imp.head(20).iloc[::-1]
    ax.barh(top.feature, top.mean_abs_shap, color="steelblue")
    ax.set_xlabel("mean |SHAP value|  (PER points)")
    ax.set_title(f"{best}: what the model actually uses")
    fig.tight_layout()
    fig.savefig(PLOTS / "shap_importance.png", dpi=140)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, Xte, feature_names=feats, max_display=18, show=False)
    plt.title(f"{best}: SHAP summary", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return imp


def plot_ridge_coefs(pipe, feats, n=20):
    coef = pipe.named_steps["model"].coef_
    c = (pd.DataFrame({"feature": feats, "coef": coef})
         .assign(abs_coef=lambda d: d.coef.abs())
         .sort_values("abs_coef", ascending=False)
         .reset_index(drop=True))
    top = c.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top.feature, top.coef,
            color=["steelblue" if v > 0 else "indianred" for v in top.coef])
    ax.axvline(0, lw=1, color="grey")
    ax.set_xlabel("standardized coefficient (PER points per 1 SD)")
    ax.set_title("Ridge: direction and size of each effect")
    fig.tight_layout()
    fig.savefig(PLOTS / "ridge_coefficients.png", dpi=140)
    plt.close(fig)
    return c


# ------------------------------------------------------------------ main
def run(reliability_floor, verbose=True):
    df, n_all = load(reliability_floor)
    feats = [c for c in df.columns if c not in ID_COLS]
    tr, va, te = split(df)
    Xtr, ytr = xy(tr, feats)
    Xva, yva = xy(va, feats)
    Xte, yte = xy(te, feats)

    base_pred = te["per"].to_numpy(dtype=float)
    res = {"Persistence": {**metrics(yte, base_pred),
                           "ci": boot_ci(yte, base_pred),
                           "improve": 0.0, "sig": False, "win": 0.5}}
    preds = {"Persistence": base_pred}
    fitted = {}

    for name, model in build_models(len(feats)).items():
        p = fit_predict(name, model, Xtr, ytr, Xva, yva, Xte)
        m, lo, hi, win = paired_boot(yte, p, base_pred)
        res[name] = {**metrics(yte, p), "ci": boot_ci(yte, p),
                     "improve": m, "imp_ci": (lo, hi),
                     "sig": lo > 0, "win": win}
        preds[name] = p
        fitted[name] = model

    return df, tr, va, te, feats, res, preds, fitted, base_pred, n_all


def main():
    df, tr, va, te, feats, res, preds, fitted, base_pred, n_all = run(TARGET_MIN_MP)

    print("=" * 66)
    print("PHASE 4 — MODELING")
    print("=" * 66)
    print(f"rows after reliability floor (target season >= {TARGET_MIN_MP} mp): "
          f"{len(df):,} of {n_all:,}  ({(1-len(df)/n_all)*100:.1f}% removed)")
    print(f"train <= {TRAIN_END}      {len(tr):,}")
    print(f"val   {TRAIN_END+1}-{VAL_END}     {len(va):,}")
    print(f"test  {VAL_END+1}+        {len(te):,}")
    print(f"features           {len(feats)}")

    print("\n--- TEST RESULTS ---")
    hdr = f"{'model':<14}{'MAE':>7}{'95% CI':>16}{'RMSE':>7}{'R2':>7}{'vs base':>10}{'p<.05':>7}"
    print(hdr)
    print("-" * len(hdr))
    for n, r in res.items():
        ci = f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]"
        imp = f"{r['improve']:+.3f}" if n != "Persistence" else "     -"
        sig = "yes" if r["sig"] else ("no" if n != "Persistence" else "-")
        print(f"{n:<14}{r['MAE']:>7.3f}{ci:>16}{r['RMSE']:>7.3f}"
              f"{r['R2']:>7.3f}{imp:>10}{sig:>7}")

    best = min((n for n in res if n != "Persistence"), key=lambda n: res[n]["MAE"])
    r = res[best]
    print(f"\nbest model: {best}")
    print(f"  MAE improvement vs persistence: {r['improve']:+.3f} "
          f"95% CI [{r['imp_ci'][0]:+.3f}, {r['imp_ci'][1]:+.3f}]")
    print(f"  relative reduction: {r['improve']/res['Persistence']['MAE']*100:.1f}%")
    print(f"  bootstrap draws where it wins: {r['win']*100:.1f}%")
    print(f"  significant at 5%: {'YES' if r['sig'] else 'NO'}")

    # ---- are the models actually distinguishable from each other? ----------
    print("\n--- HEAD TO HEAD vs best (paired bootstrap) ---")
    yte = te["target"].to_numpy(dtype=float)
    for n in res:
        if n in ("Persistence", best):
            continue
        m, lo, hi, win = paired_boot(yte, preds[best], preds[n])
        verdict = "distinguishable" if lo > 0 else "TIED"
        print(f"  {best} vs {n:<13} {m:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  {verdict}")

    plot_comparison(res, res["Persistence"]["MAE"])
    plot_pred_actual(te, preds, best)
    plot_residual_slices(te, preds, best, base_pred)

    # ---- interpretability ---------------------------------------------------
    Xte = te[feats].to_numpy(dtype=float)

    # SHAP needs a tree model, so use XGBoost even when Ridge nominally wins
    imp = plot_shap(fitted["XGBoost"], Xte, feats, "XGBoost")
    imp.to_csv(DATA / "shap_importance.csv", index=False)
    print("\n--- TOP 15 FEATURES, XGBoost (mean |SHAP|) ---")
    print(imp.head(15).round(4).to_string(index=False))

    coefs = plot_ridge_coefs(fitted["Ridge"], feats)
    coefs.to_csv(DATA / "ridge_coefficients.csv", index=False)
    print("\n--- TOP 15 RIDGE COEFFICIENTS (standardized) ---")
    print(coefs.head(15).round(3).to_string(index=False))

    # ---- sensitivity: no reliability floor ---------------------------------
    print("\n--- SENSITIVITY: no reliability floor ---")
    _, _, _, te2, _, res2, _, _, _, _ = run(None)
    print(f"test rows {len(te2):,}")
    for n in ["Persistence", best]:
        rr = res2[n]
        print(f"  {n:<14} MAE {rr['MAE']:.3f}  "
              f"vs base {rr['improve']:+.3f}" if n != "Persistence"
              else f"  {n:<14} MAE {rr['MAE']:.3f}")

    pd.DataFrame([
        {"model": n, "MAE": r["MAE"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1],
         "RMSE": r["RMSE"], "R2": r["R2"], "improve_vs_base": r["improve"],
         "significant": r["sig"]}
        for n, r in res.items()
    ]).to_csv(DATA / "model_results.csv", index=False)

    out = te[["season", "player_id", "player", "per", "target"]].copy()
    for n, p in preds.items():
        out[f"pred_{n}"] = p
    out.to_csv(DATA / "test_predictions.csv", index=False)

    print(f"\nplots -> {PLOTS}")


if __name__ == "__main__":
    main()
