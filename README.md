# SME Loan Default Risk – Flask Web App

This Flask interface uses the **complete fitted scikit-learn pipeline** saved by your ML project at `outputs/best_model.pkl`.

## Keep your ML project files alongside `app.py`

- `config.py`
- `data_prep.py`
- `preprocessing.py`
- `main.py`
- Your CSV dataset configured in `config.py`

The Flask folder also contains `app.py`, `templates/`, `static/`, and `requirements.txt`.

## Run locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train models and save the best pipeline (skip if you already have a compatible `outputs/best_model.pkl`):

   ```bash
   python main.py
   ```

3. Start Flask:

   ```bash
   python app.py
   ```

4. Open `http://127.0.0.1:5000`.

## Included result-page improvements

- Model score gauge for models that provide `predict_proba`.
- Illustrative Low / Medium / High categories (20% and 50% cut-offs).
- Submitted loan details summary.
- Optional top-five SHAP feature contributions for supported tree estimators (LightGBM, Random Forest, Decision Tree, Extra Trees, Gradient Boosting, XGBoost).
- Safe fallback message when SHAP is unavailable or the selected estimator is not supported.

## Important modelling cautions

- The `predict_proba` value is **not automatically a calibrated probability**. The page labels it as a model-estimated default score. Calibrate and validate it before interpreting it as a true probability of default.
- Risk thresholds are illustrative only. Select operational thresholds using out-of-time validation, business costs, lending policy, and fairness review. Do not use these demonstration categories to approve or reject real loans.
- SHAP values describe how model features contribute to the model output; they do not establish causation.
- Categorical form values should match the conventions in the training data. Review the options and add dropdowns based on actual unique values from your dataset before production use.
- Disable Flask debug mode and use a production WSGI server when deploying.
