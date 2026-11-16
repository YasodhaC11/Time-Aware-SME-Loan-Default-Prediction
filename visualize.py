"""visualize.py - comparison graphs and SHAP plots (saved as PNG files in outputs/)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save to files; no window pops up
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

import config as cfg


def plot_comparison(table):
    """Bar charts of the main test metrics for every model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Test ROC-AUC", "Test PR-AUC", "Recall@10%"]):
        table[col].sort_values().plot.barh(ax=ax, color="steelblue")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(f"{cfg.OUT}/model_comparison.png", dpi=150)
    plt.close()


def plot_curves(test_scores, y_test):
    """ROC curve, precision-recall curve, and defaults-caught curve for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for name, s in test_scores.items():
        fpr, tpr, _ = roc_curve(y_test, s)
        axes[0].plot(fpr, tpr, label=name)
        prec, rec, _ = precision_recall_curve(y_test, s)
        axes[1].plot(rec, prec, label=name)
        order = np.argsort(-s)
        axes[2].plot(np.arange(1, len(s) + 1) / len(s),
                     np.cumsum(y_test[order]) / y_test.sum(), label=name)

    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(title="ROC curve", xlabel="False positive rate",
                ylabel="True positive rate")
    axes[1].axhline(y_test.mean(), color="k", ls="--")
    axes[1].set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    axes[2].plot([0, 1], [0, 1], "k--")
    axes[2].set(title="Defaults caught vs loans reviewed",
                xlabel="Share of loans reviewed (riskiest first)",
                ylabel="Share of defaults caught")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{cfg.OUT}/curves.png", dpi=150)
    plt.close()


def plot_shap(fitted, test, model_name=cfg.SHAP_MODEL):
    """SHAP shows how much each column pushes a loan's risk up or down.
    Works for tree models (LightGBM, RandomForest, DecisionTree)."""
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return

    pipe = fitted[model_name]
    Xt = pipe[:-1].transform(test[cfg.FEATURES])               # data as the model sees it
    try:
        names = [n.split("__", 1)[-1] for n in pipe[:-1].get_feature_names_out()]
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]

    idx = np.random.RandomState(cfg.SEED).choice(len(Xt), min(3000, len(Xt)),
                                                 replace=False)
    Xs = pd.DataFrame(Xt[idx], columns=names)

    sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xs)
    if isinstance(sv, list):                 # older shap: one array per class
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]

    shap.summary_plot(sv, Xs, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{cfg.OUT}/shap_summary.png", dpi=150)
    plt.close()
    shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{cfg.OUT}/shap_importance.png", dpi=150)
    plt.close()
