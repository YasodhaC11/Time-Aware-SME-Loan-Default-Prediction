"""preprocessing.py - scaling, encoding and feature selection (all inside a Pipeline).

Because everything is in a Pipeline, it is learned from TRAIN data only (no leakage)
and re-learned inside every cross-validation fold.
"""
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import config as cfg
from data_prep import time_folds


def make_preprocessor():
    return ColumnTransformer([
        # numbers: fill missing with the median, then scale (mean 0, std 1)
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), cfg.NUM_COLS),
        # text categories -> 0/1 columns. Levels with < 200 train rows are merged
        # into one "infrequent" group (unseen levels join it too).
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=200, sparse_output=False,
                              dtype=np.float32), cfg.CAT_COLS),
    ])


def make_selector(option):
    if option == "kbest":     # keep the K columns most related to default (ANOVA F-test)
        return SelectKBest(f_classif, k=cfg.K_FEATURES)
    if option == "rfe":       # repeatedly drop the weakest columns using a logistic model
        return RFE(LogisticRegression(C=0.1, max_iter=500),
                   n_features_to_select=cfg.K_FEATURES, step=0.2)
    return "passthrough"      # keep everything


def make_pipe(model, selection=None):
    """preprocessing -> feature selection -> model"""
    selection = cfg.FEATURE_SELECTION if selection is None else selection
    return Pipeline([("prep", make_preprocessor()),
                     ("select", make_selector(selection)),
                     ("model", model)])


def compare_selection(train):
    """Optional: compare none / kbest / rfe using Logistic and time-based CV."""
    rows = []
    for opt in ["none", "kbest", "rfe"]:
        t0 = time.time()
        pipe = make_pipe(LogisticRegression(C=0.1, class_weight="balanced",
                                            max_iter=2000), opt)
        s = cross_val_score(pipe, train[cfg.FEATURES], train[cfg.TARGET],
                            cv=time_folds(train.approval_year), scoring="roc_auc")
        rows.append({"selection": opt, "CV ROC-AUC": s.mean().round(4),
                     "std": s.std().round(4), "seconds": round(time.time() - t0)})
    print("\nFeature selection comparison (Logistic, time-based CV):")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"-> currently using FEATURE_SELECTION='{cfg.FEATURE_SELECTION}'. "
          "Change it in config.py only if another option is clearly better "
          "(differences under 0.005 are noise).\n")
