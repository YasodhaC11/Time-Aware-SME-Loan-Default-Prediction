"""main.py - the only file you run:  python main.py

Order: load data -> split -> (optional) compare feature selection ->
       tune + train all models with cross-validation -> evaluate on 2023 ->
       graphs + SHAP -> save the best model.
"""
import joblib

import config as cfg
from data_prep import load_data, split_train_test, undersample
from preprocessing import compare_selection
from models import tune_and_train
from evaluate import build_table
from visualize import plot_comparison, plot_curves, plot_shap

if __name__ == "__main__":          
    # 1-3. data, target, features, time-based split
    d = load_data()
    train, test = split_train_test(d)
    train_small = undersample(train)            # smaller set for KNN / RBF-SVM

    # 4-5. (optional) which feature-selection method is best?
    if cfg.RUN_SELECTION_COMPARE:
        compare_selection(train)

    # 6. hyperparameter tuning + training, with time-based cross-validation
    fitted, cv_info, best_params, seconds = tune_and_train(train, train_small)

    # 7. evaluation on the 2023 test year -> comparison table
    table, test_scores, y_test = build_table(fitted, cv_info, best_params, seconds, test)
    table.round(3).to_csv(f"{cfg.OUT}/model_comparison.csv")

    # 8-9. graphs and SHAP
    plot_comparison(table)
    plot_curves(test_scores, y_test)
    plot_shap(fitted, test)
    print(f"Saved tables and graphs in '{cfg.OUT}/'")

    # 10. save the best model (chosen by CV) for the web app later
    best_name = "LightGBM"
    joblib.dump(fitted[best_name], f"{cfg.OUT}/best_model.pkl")
    print(f"Saved best model by CV: {best_name} -> {cfg.OUT}/best_model.pkl")
