"""evaluate.py - score the models on the test year and build the comparison table.

All metrics measure RANKING quality: with ~2% defaults, accuracy is meaningless,
and several models give scores that are not true probabilities.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import config as cfg


def get_score(model, X):
    """Risk score where higher = riskier (SVMs give a distance, not a probability)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def recall_at(y, score, frac):
    """Share of ALL real defaults found in the riskiest `frac` of loans.
    Random scoring would give `frac` (e.g. 10%)."""
    k = int(len(y) * frac)
    top = np.argsort(score)[::-1][:k]
    return y[top].sum() / y.sum()


def build_table(fitted, cv_info, best_params, seconds, test):
    """Returns (comparison table, test scores per model, y_test)."""
    y_test = test[cfg.TARGET].to_numpy()
    rows, test_scores = [], {}
    for name, model in fitted.items():
        s = get_score(model, test[cfg.FEATURES])
        test_scores[name] = s
        ap = average_precision_score(y_test, s)
        rows.append({
            "Model": name,
            "CV ROC-AUC": cv_info[name][0],          # used to CHOOSE the model
            "CV std": cv_info[name][1],
            "Test ROC-AUC": roc_auc_score(y_test, s),  # final report on 2023
            "Test PR-AUC": ap,
            "Lift": ap / y_test.mean(),               # PR-AUC relative to the base rate
            "Recall@10%": recall_at(y_test, s, 0.10),
            "Recall@20%": recall_at(y_test, s, 0.20),
            "Seconds": seconds[name],
            "Best settings": str(best_params[name]),
        })
    table = pd.DataFrame(rows).set_index("Model")
    table = table.sort_values("CV ROC-AUC", ascending=False)   # choose by CV, not by test
    print(f"\n=== COMPARISON (test = 2023, default rate {y_test.mean():.4f}; "
          "random Recall@10% = 0.10) ===")
    print(table.drop(columns="Best settings").round(3))
    print("\nChoose the model by the CV columns; the Test columns are the final report.")
    return table, test_scores, y_test
