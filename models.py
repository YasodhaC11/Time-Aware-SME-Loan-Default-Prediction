"""models.py - the 8 models, their small tuning grids, and the tune+train loop.

Each model: try a small grid of settings, score each by time-based cross-validation
(ROC-AUC), then refit the best setting on all training data.
"""
import time
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC

import config as cfg
from data_prep import time_folds
from preprocessing import make_pipe


def get_models():
    """{name: (model, settings to try, is_slow)}. is_slow=True -> uses the smaller sample."""
    return {
        "Logistic": (LogisticRegression(class_weight="balanced", max_iter=2000),
                     {"model__C": [0.01, 0.1, 1]}, False),
        "KNN": (KNeighborsClassifier(weights="distance", n_jobs=-1),
                {"model__n_neighbors": [50, 100, 200]}, True),
        "NaiveBayes": (GaussianNB(),
                       {"model__var_smoothing": [1e-9, 1e-6, 1e-3]}, False),
        "DecisionTree": (DecisionTreeClassifier(class_weight="balanced",
                                                random_state=cfg.SEED),
                         {"model__max_depth": [4, 6, 8],
                          "model__min_samples_leaf": [100, 200]}, False),
        "RandomForest": (RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                                class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=cfg.SEED),
                         {"model__min_samples_leaf": [20, 50, 100]}, False),
        "LinearSVM": (LinearSVC(class_weight="balanced", dual=False, max_iter=5000),
                      {"model__C": [0.001, 0.01, 0.1]}, False),
        "RBF_SVM": (SVC(kernel="rbf", class_weight="balanced"),
                    {"model__C": [0.5, 1, 2]}, True),
        "XGBoost": (XGBClassifier(learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                             reg_lambda=5, eval_metric="logloss", n_jobs=-1,
                             random_state=cfg.SEED, verbosity=0),
               {"model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 4, 6]}, False),
        "LightGBM": (lgb.LGBMClassifier(learning_rate=0.03, subsample=0.8,
                                        subsample_freq=1, colsample_bytree=0.7,
                                        reg_lambda=5, random_state=cfg.SEED, verbose=-1),
                     {"model__n_estimators": [100, 200, 300],
                      "model__num_leaves": [7, 15]}, False),
    }


def tune_and_train(train, train_small):
    """Returns (fitted pipelines, CV scores, best settings, seconds) per model name."""
    fitted, cv_info, best_params, seconds = {}, {}, {}, {}
    for name, (estimator, grid, is_slow) in get_models().items():
        t0 = time.time()
        df_fit = train_small if is_slow else train
        search = GridSearchCV(make_pipe(estimator), grid, scoring="roc_auc",
                              cv=time_folds(df_fit.approval_year),
                              refit=True, n_jobs=1)
        search.fit(df_fit[cfg.FEATURES], df_fit[cfg.TARGET])

        i = search.best_index_
        fitted[name] = search.best_estimator_
        cv_info[name] = (search.cv_results_["mean_test_score"][i],
                         search.cv_results_["std_test_score"][i])
        best_params[name] = {k.replace("model__", ""): v
                             for k, v in search.best_params_.items()}
        seconds[name] = round(time.time() - t0)
        print(f"done {name:13s} CV ROC-AUC {cv_info[name][0]:.3f}  "
              f"best={best_params[name]}  ({seconds[name]}s)")
    return fitted, cv_info, best_params, seconds
