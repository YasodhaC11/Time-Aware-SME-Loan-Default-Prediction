"""Flask web app for the SME Loan Default Risk model.

Order:   python main.py   ->   python make_app_meta.py   ->   python app.py
Run from the project folder (the one containing config.py).
"""
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

import config as cfg
from data_prep import add_features
from evaluate import get_score

warnings.filterwarnings("ignore")

app = Flask(__name__)
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / cfg.OUT / "best_model.pkl"
META_PATH = BASE / cfg.OUT / "app_meta.json"

# ---------------------------------------------------------------------------
# Load the model and metadata once, when the app starts
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"{MODEL_PATH} not found. Run `python main.py` first.")
if not META_PATH.exists():
    raise FileNotFoundError(f"{META_PATH} not found. Run `python make_app_meta.py` first.")

model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text())
ACTIVE = list(model.feature_names_in_)            # columns this model was trained on
if ACTIVE != meta["active_features"]:
    raise RuntimeError("app_meta.json was built for a different model. "
                       "Re-run `python make_app_meta.py`.")
QUANTILES = np.array(meta["quantiles"])

# ---------------------------------------------------------------------------
# Form definition. Only the inputs the saved model needs are shown.
# ---------------------------------------------------------------------------
SELECT_FIELDS = [
    ("BusinessType", "Business type"),
    ("BusinessAge", "Business age"),
    ("ProcessingMethod", "Processing method"),
    ("FixedorVariableInterestInd", "Interest rate type"),
    ("RevolverStatus", "Revolver status"),
    ("CollateralInd", "Collateral available"),
    ("ProjectState", "Project state"),
    ("BankName", "Bank / lender"),
]

NUMBER_FIELDS = {   # label, lowest / highest allowed, strict lower bound, range key
    "GrossApproval": dict(label="Gross loan approval", lo=0, hi=1e9, strict=True, rk="amount"),
    "SBAGuaranteedApproval": dict(label="SBA guaranteed amount", lo=0, hi=1e9, strict=False, rk=None),
    "InitialInterestRate": dict(label="Initial interest rate", lo=0, hi=100, strict=False, rk="rate"),
    "JobsSupported": dict(label="Jobs supported", lo=0, hi=100000, strict=False, rk="jobs"),
}

INPUTS_FOR = {      # which form inputs each model column needs
    "log_amount": ["GrossApproval"],
    "guarantee_pct": ["GrossApproval", "SBAGuaranteedApproval"],
    "amount_per_job": ["GrossApproval", "JobsSupported"],
    "rate": ["InitialInterestRate"],
    "jobs": ["JobsSupported"],
    "zero_jobs": ["JobsSupported"],
    "is_franchise": ["Franchise"],
    "naics3": ["NaicsCode"],
}
NEEDED = set()
for _f in ACTIVE:
    NEEDED.update(INPUTS_FOR.get(_f, [_f]))       # dropdown columns need themselves

SHOW = {          # which of the three form sections are needed
    "loan": bool(NEEDED & {"GrossApproval", "SBAGuaranteedApproval",
                           "InitialInterestRate", "JobsSupported"}),
    "business": bool(NEEDED & {"BusinessType", "BusinessAge", "NaicsCode", "Franchise"}),
    "terms": bool(NEEDED & {"FixedorVariableInterestInd", "RevolverStatus",
                            "CollateralInd", "ProcessingMethod", "ProjectState", "BankName"}),
}
ALLOWED = {c: {o["value"] for o in opts} for c, opts in meta["options"].items()}
OPTION_LABEL = {c: {o["value"]: o["label"] for o in opts} for c, opts in meta["options"].items()}


# ---------------------------------------------------------------------------
# Reading and checking the form
# ---------------------------------------------------------------------------
def read_form(form):
    """Returns (numbers, naics digits, franchise answer, errors, notices)."""
    errors, notices, values = [], [], {}

    for key, spec in NUMBER_FIELDS.items():
        if key not in NEEDED:
            values[key] = np.nan
            continue
        try:
            v = float((form.get(key) or "").strip())
        except ValueError:
            v = np.nan
        values[key] = v
        if not np.isfinite(v):
            errors.append(f"{spec['label']}: enter a valid number.")
        elif (v <= spec["lo"]) if spec["strict"] else (v < spec["lo"]):
            errors.append(f"{spec['label']} is too low.")
        elif v > spec["hi"]:
            errors.append(f"{spec['label']} is too high.")
        elif spec["rk"]:
            lo, _, hi = meta["ranges"][spec["rk"]]
            if not (lo <= v <= hi):
                notices.append(f"{spec['label']} is outside the range seen in most training "
                               f"loans ({lo:,.4g} to {hi:,.4g}); the result may be less reliable.")

    g, s = values["GrossApproval"], values["SBAGuaranteedApproval"]
    if np.isfinite(g) and np.isfinite(s) and s > g:
        errors.append("SBA guaranteed amount cannot be more than the gross loan approval.")

    digits = ""
    if "NaicsCode" in NEEDED:
        digits = re.sub(r"\D", "", form.get("NaicsCode", ""))
        if len(digits) < 3:
            errors.append("NAICS code: enter at least the first 3 digits.")
        elif meta["naics3_levels"] and digits[:3] not in meta["naics3_levels"]:
            notices.append(f"Industry code {digits[:3]} is rare in the training data and "
                           "is treated as 'other'.")

    franchise = form.get("Franchise")
    if "Franchise" in NEEDED and franchise not in ("yes", "no"):
        errors.append("Franchise: choose yes or no.")

    for key, label in SELECT_FIELDS:
        if key in NEEDED and form.get(key) not in ALLOWED[key]:
            errors.append(f"{label}: choose a value from the list.")
    return values, digits, franchise, errors, notices


def make_model_input(form, values, digits, franchise):
    """Build raw columns, then reuse the SAME feature engineering as training."""
    raw = {
        "GrossApproval": values["GrossApproval"],
        "SBAGuaranteedApproval": values["SBAGuaranteedApproval"],
        "InitialInterestRate": values["InitialInterestRate"],
        "JobsSupported": values["JobsSupported"],
        "NaicsCode": digits or None,
        "FranchiseCode": "1" if franchise == "yes" else None,
    }
    for key, _ in SELECT_FIELDS:
        raw[key] = form.get(key) if key in NEEDED else None
    return add_features(pd.DataFrame([raw]))[ACTIVE]


# ---------------------------------------------------------------------------
# Explanations (SHAP), grouped by original input so they are easy to read
# ---------------------------------------------------------------------------
TREE_MODELS = {"LGBMClassifier", "RandomForestClassifier", "DecisionTreeClassifier",
               "ExtraTreesClassifier", "GradientBoostingClassifier", "XGBClassifier"}
GROUP_LABELS = {
    "log_amount": "Loan amount", "guarantee_pct": "SBA guarantee share",
    "amount_per_job": "Loan amount per job", "rate": "Interest rate",
    "jobs": "Jobs supported", "zero_jobs": "No jobs reported", "is_franchise": "Franchise",
    "naics3": "Industry (NAICS)", "BusinessType": "Business type",
    "BusinessAge": "Business age", "ProjectState": "Project state",
    "ProcessingMethod": "Processing method", "BankName": "Bank / lender",
    "FixedorVariableInterestInd": "Interest rate type",
    "RevolverStatus": "Revolver status", "CollateralInd": "Collateral",
}
CAT_GROUPS = sorted([c for c, _ in SELECT_FIELDS] + ["naics3"], key=len, reverse=True)
EXPLAINER, GROUPS, SHAP_NOTE = None, None, ""


def group_of(encoded_name):
    """'cat__BankName_WELLS FARGO' -> 'BankName';  'num__rate' -> 'rate'."""
    name = encoded_name.split("__", 1)[-1]
    for col in CAT_GROUPS:
        if name.startswith(col + "_"):
            return col
    return name


def init_shap():
    global EXPLAINER, GROUPS, SHAP_NOTE
    estimator = model.named_steps["model"]
    if type(estimator).__name__ not in TREE_MODELS:
        SHAP_NOTE = "Feature explanations are available for tree-based models only."
        return
    try:
        import shap
        names = list(model[:-1].get_feature_names_out())
        EXPLAINER = shap.TreeExplainer(estimator)      # built once, not per request
        GROUPS = [group_of(n) for n in names]
    except ImportError:
        SHAP_NOTE = "Install SHAP (`pip install shap`) to enable feature explanations."
    except Exception as exc:
        SHAP_NOTE = f"Feature explanations are unavailable ({type(exc).__name__})."


init_shap()


def display_value(group, row, p):
    if group == "log_amount":
        return f"${p['gross']:,.0f}"
    if group == "guarantee_pct":
        return f"{p['guaranteed'] / p['gross']:.0%}"
    if group == "amount_per_job":
        return f"${p['gross'] / p['jobs']:,.0f} per job" if p["jobs"] else "no jobs"
    if group == "rate":
        return f"{p['rate']:g}%"
    if group == "jobs":
        return f"{p['jobs']:g}"
    if group == "zero_jobs":
        return "Yes" if p["jobs"] == 0 else "No"
    if group == "is_franchise":
        return "Yes" if p["franchise"] else "No"
    value = str(row[group])
    return OPTION_LABEL.get(group, {}).get(value, value)


def explain(X, p):
    """Top 5 inputs pushing this loan's score up or down."""
    if EXPLAINER is None:
        return [], SHAP_NOTE
    try:
        vals = EXPLAINER.shap_values(model[:-1].transform(X))
        if isinstance(vals, list):                     # older shap: one array per class
            vals = vals[1] if len(vals) > 1 else vals[0]
        vals = np.asarray(vals)
        vals = (vals[0, :, 1] if vals.shape[-1] > 1 else vals[0, :, 0]) if vals.ndim == 3 \
            else vals.reshape(-1)
        by_input = pd.Series(vals, index=GROUPS).groupby(level=0).sum()
        top = by_input.reindex(by_input.abs().sort_values(ascending=False).index).head(5)
        biggest = float(top.abs().max()) or 1.0
        row = X.iloc[0]
        return [{"name": GROUP_LABELS.get(g, g), "value": display_value(g, row, p),
                 "up": bool(v > 0), "width": max(4, round(abs(v) / biggest * 100))}
                for g, v in top.items()], ""
    except Exception as exc:
        app.logger.exception("SHAP failed")
        return [], f"Feature explanation could not be generated ({type(exc).__name__})."


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_context(**extra):
    ctx = dict(needed=NEEDED, show=SHOW, options=meta["options"], ranges=meta["ranges"],
               select_fields=[(k, lbl) for k, lbl in SELECT_FIELDS if k in NEEDED],
               form={}, errors=[])
    ctx.update(extra)
    return ctx


def loan_summary(form, v, digits, franchise):
    rows = []
    if "GrossApproval" in NEEDED:
        rows.append(("Gross loan approval", f"${v['GrossApproval']:,.2f}"))
    if "SBAGuaranteedApproval" in NEEDED:
        rows.append(("SBA guaranteed amount", f"${v['SBAGuaranteedApproval']:,.2f}"))
    if "InitialInterestRate" in NEEDED:
        rows.append(("Initial interest rate", f"{v['InitialInterestRate']:g}%"))
    if "JobsSupported" in NEEDED:
        rows.append(("Jobs supported", f"{v['JobsSupported']:g}"))
    if "NaicsCode" in NEEDED:
        rows.append(("NAICS industry code", digits))
    if "Franchise" in NEEDED:
        rows.append(("Franchise", "Yes" if franchise == "yes" else "No"))
    for key, label in SELECT_FIELDS:
        if key in NEEDED:
            rows.append((label, OPTION_LABEL[key].get(form[key], form[key])))
    return rows


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", **page_context())


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form.to_dict()
    values, digits, franchise, errors, notices = read_form(form)
    if errors:
        return render_template("index.html", **page_context(form=form, errors=errors)), 400

    try:
        X = make_model_input(form, values, digits, franchise)
        score = float(get_score(model, X)[0])
        # Where does this loan sit among the 2023 test loans? (0-100)
        percentile = float(np.interp(score, QUANTILES, np.arange(101)))
        tier = "High" if percentile >= 90 else "Medium" if percentile >= 70 else "Low"

        parsed = dict(gross=values["GrossApproval"], guaranteed=values["SBAGuaranteedApproval"],
                      jobs=values["JobsSupported"], rate=values["InitialInterestRate"],
                      franchise=(franchise == "yes"))
        drivers, shap_note = explain(X, parsed)

        return render_template(
            "result.html", meta=meta, score=score, percentile=percentile, tier=tier,
            tier_class=tier.lower(), tier_stats=meta["tiers"][tier],
            base_rate=meta["base_rate_test"], notices=notices,
            loan_details=loan_summary(form, values, digits, franchise),
            drivers=drivers, shap_note=shap_note)
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template("index.html", **page_context(
            form=form, errors=["Prediction failed. Check the server log for details."])), 500


if __name__ == "__main__":
    # debug=True is for local development only. Turn it off for any real deployment.
    app.run(debug=True)