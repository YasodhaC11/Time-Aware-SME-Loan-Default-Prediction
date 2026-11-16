"""feature_selection_study.py - compare none / SelectKBest / RFE and SAVE the results.

Run:  python feature_selection_study.py
Saves: outputs/feature_selection_results.csv  and  outputs/feature_selection_results.png

For each model (Logistic, LightGBM) and each setting, it reports the
time-based cross-validation ROC-AUC (train 2019-20 -> validate 2021,
train 2019-21 -> validate 2022). The 2023 test year is NOT used.
Note: results depend on which columns are active in config.py (DROP_COLS).
"""
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save to file; no window pops up
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import config as cfg
from data_prep import load_data, split_train_test, time_folds
from preprocessing import make_pipe, make_preprocessor

K_VALUES = [20, 40, 80]                    # how many columns to keep when selecting


def make_model(name):
    if name == "Logistic":
        return LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=7, learning_rate=0.03,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=5, random_state=cfg.SEED, verbose=-1)


if __name__ == "__main__":
    d = load_data()
    train, _ = split_train_test(d)
    folds = time_folds(train.approval_year)

    n_all = make_preprocessor().fit_transform(train[cfg.FEATURES]).shape[1]
    print(f"\nColumns after encoding (before any selection): {n_all}\n")

    settings = [("none", None)]
    settings += [("kbest", k) for k in K_VALUES]
    settings += [("rfe", k) for k in K_VALUES]
    original_k = cfg.K_FEATURES

    rows = []
    for model_name in ["Logistic", "LightGBM"]:
        for option, k in settings:
            label = "none" if option == "none" else f"{option} K={k}"
            if k is not None:
                cfg.K_FEATURES = k           # make_selector reads this
            t0 = time.time()
            try:
                pipe = make_pipe(make_model(model_name), option)
                s = cross_val_score(pipe, train[cfg.FEATURES], train[cfg.TARGET],
                                    cv=folds, scoring="roc_auc")
                rows.append({"Model": model_name, "Setting": label,
                             "Columns used": n_all if option == "none" else k,
                             "CV ROC-AUC": round(s.mean(), 4), "CV std": round(s.std(), 4),
                             "Fold 2021": round(s[0], 4), "Fold 2022": round(s[1], 4),
                             "Seconds": round(time.time() - t0), "Note": ""})
                print(f"{model_name:9s} {label:12s} CV ROC-AUC {s.mean():.4f} "
                      f"(std {s.std():.4f})  {time.time() - t0:.0f}s")
            except Exception as e:           # e.g. K larger than the number of columns
                rows.append({"Model": model_name, "Setting": label, "Columns used": k,
                             "CV ROC-AUC": None, "CV std": None, "Fold 2021": None,
                             "Fold 2022": None, "Seconds": round(time.time() - t0),
                             "Note": str(e)[:80]})
                print(f"{model_name:9s} {label:12s} FAILED: {str(e)[:80]}")
    cfg.K_FEATURES = original_k

    res = pd.DataFrame(rows)
    res.to_csv(f"{cfg.OUT}/feature_selection_results.csv", index=False)
    print("\n=== FEATURE SELECTION RESULTS ===")
    print(res.drop(columns="Note").to_string(index=False))

    # grouped bar chart: one group per setting, one bar per model
    order = ["none"] + [f"{o} K={k}" for o in ("kbest", "rfe") for k in K_VALUES]
    pivot = res.pivot(index="Setting", columns="Model", values="CV ROC-AUC").reindex(order)
    ax = pivot.plot.bar(figsize=(10, 5), rot=30)
    lo, hi = pivot.min().min(), pivot.max().max()
    ax.set_ylim(lo - 0.01, hi + 0.01)
    ax.set_ylabel("CV ROC-AUC (time-based)")
    ax.set_title("Feature selection comparison (higher is better)")
    plt.tight_layout()
    plt.savefig(f"{cfg.OUT}/feature_selection_results.png", dpi=150)
    plt.close()
    print(f"\nSaved: {cfg.OUT}/feature_selection_results.csv and .png")
