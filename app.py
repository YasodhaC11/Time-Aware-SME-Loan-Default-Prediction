"""Flask web app for the SME Loan Default Risk model.

Run from the project folder (the one containing config.py).
The model and app_meta.json must already exist in outputs/.
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
    raise FileNotFoundError(
        f"{MODEL_PATH} not found. Make sure best_model.pkl is deployed."
    )
if not META_PATH.exists():
    raise FileNotFoundError(
        f"{META_PATH} not found. Run make_app_meta.py locally and deploy it."
    )

model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8"))

ACTIVE = list(model.feature_names_in_)
if ACTIVE != meta.get("active_features"):
    raise RuntimeError(
        "app_meta.json was built for a different model. "
        "Run python make_app_meta.py again using this model."
    )

QUANTILES = np.asarray(meta["quantiles"], dtype=float)
if len(QUANTILES) != 101:
    raise RuntimeError("app_meta.json must contain 101 score quantiles.")

# ---------------------------------------------------------------------------
# Form definition. Only inputs required by the saved model are shown.
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

NUMBER_FIELDS = {
    "GrossApproval": {
        "label": "Gross loan approval", "lo": 0, "hi": 1e9,
        "strict": True, "rk": "amount"
    },
    "SBAGuaranteedApproval": {
        "label": "SBA guaranteed amount", "lo": 0, "hi": 1e9,
        "strict": False, "rk": None
    },
    "InitialInterestRate": {
        "label": "Initial interest rate", "lo": 0, "hi": 100,
        "strict": False, "rk": "rate"
    },
    "JobsSupported": {
        "label": "Jobs supported", "lo": 0, "hi": 100000,
        "strict": False, "rk": "jobs"
    },
}

INPUTS_FOR = {
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
for feature in ACTIVE:
    NEEDED.update(INPUTS_FOR.get(feature, [feature]))

SHOW = {
    "loan": bool(NEEDED & {
        "GrossApproval", "SBAGuaranteedApproval",
        "InitialInterestRate", "JobsSupported"
    }),
    "business": bool(NEEDED & {
        "BusinessType", "BusinessAge", "NaicsCode", "Franchise"
    }),
    "terms": bool(NEEDED & {
        "FixedorVariableInterestInd", "RevolverStatus", "CollateralInd",
        "ProcessingMethod", "ProjectState", "BankName"
    }),
}

# Normalize metadata options once. Training data is normalized to uppercase
# in data_prep.add_features(), so submitted categorical values must match it.
ALLOWED = {
    column: {
        str(option["value"]).strip().upper()
        for option in options
    }
    for column, options in meta.get("options", {}).items()
}
OPTION_LABEL = {
    column: {
        str(option["value"]).strip().upper(): option["label"]
        for option in options
    }
    for column, options in meta.get("options", {}).items()
}

# ---------------------------------------------------------------------------
# Read and validate submitted form values
# ---------------------------------------------------------------------------
def read_form(form):
    """Return (numeric values, NAICS digits, franchise, errors, notices).

    The form dictionary is normalized in place for categorical values so that
    validation, display, and feature engineering all use the same values.
    """
    errors = []
    notices = []
    values = {}

    # Match data_prep.add_features(): uppercase and strip categorical values.
    for key, _label in SELECT_FIELDS:
        raw_value = form.get(key)
        if raw_value is not None:
            form[key] = raw_value.strip().upper()

    franchise = (form.get("Franchise") or "").strip().lower()
    if "Franchise" in NEEDED and franchise not in ("yes", "no"):
        errors.append("Franchise: choose yes or no.")

    for key, spec in NUMBER_FIELDS.items():
        if key not in NEEDED:
            values[key] = np.nan
            continue

        raw_value = (form.get(key) or "").strip()
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = np.nan

        values[key] = value

        if not np.isfinite(value):
            errors.append(f"{spec['label']}: enter a valid number.")
            continue

        if value <= spec["lo"] if spec["strict"] else value < spec["lo"]:
            errors.append(f"{spec['label']} is too low.")
            continue

        if value > spec["hi"]:
            errors.append(f"{spec['label']} is too high.")
            continue

        if spec["rk"]:
            range_values = meta.get("ranges", {}).get(spec["rk"])
            if range_values and len(range_values) == 3:
                low, _median, high = range_values
                if not (low <= value <= high):
                    notices.append(
                        f"{spec['label']} is outside the range seen in most "
                        f"training loans ({low:,.4g} to {high:,.4g}); "
                        "the result may be less reliable."
                    )

    gross = values["GrossApproval"]
    guaranteed = values["SBAGuaranteedApproval"]
    if (
        np.isfinite(gross)
        and np.isfinite(guaranteed)
        and guaranteed > gross
    ):
        errors.append(
            "SBA guaranteed amount cannot be more than the gross loan approval."
        )

    digits = ""
    if "NaicsCode" in NEEDED:
        digits = re.sub(r"\D", "", form.get("NaicsCode", ""))
        if len(digits) < 3:
            errors.append("NAICS code: enter at least the first 3 digits.")
        else:
            known_levels = set(meta.get("naics3_levels", []))
            if known_levels and digits[:3] not in known_levels:
                notices.append(
                    f"Industry code {digits[:3]} is rare in the training data "
                    "and is treated as an unseen category by the model."
                )

    for key, label in SELECT_FIELDS:
        if key not in NEEDED:
            continue
        allowed_values = ALLOWED.get(key, set())
        submitted = form.get(key, "")
        if not submitted:
            errors.append(f"{label}: choose a value from the list.")
        elif submitted not in allowed_values:
            errors.append(
                f"{label}: the selected value is not in the current model options. "
                "Refresh the page and choose an available value."
            )

    return values, digits, franchise, errors, notices


def make_model_input(form, values, digits, franchise):
    """Build raw columns and reuse the exact training feature engineering."""
    raw = {
        "GrossApproval": values["GrossApproval"],
        "SBAGuaranteedApproval": values["SBAGuaranteedApproval"],
        "InitialInterestRate": values["InitialInterestRate"],
        "JobsSupported": values["JobsSupported"],
        "NaicsCode": digits or None,
        "FranchiseCode": "1" if franchise == "yes" else None,
    }

    for key, _label in SELECT_FIELDS:
        raw[key] = form.get(key) if key in NEEDED else None

    # add_features() applies the same engineered features and categorical
    # normalization used during training.
    model_input = add_features(pd.DataFrame([raw]))

    missing_columns = [column for column in ACTIVE if column not in model_input.columns]
    if missing_columns:
        raise ValueError(
            "Prediction input is missing model features: "
            + ", ".join(missing_columns)
        )

    return model_input[ACTIVE]


# ---------------------------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------------------------
TREE_MODELS = {
    "LGBMClassifier", "RandomForestClassifier", "DecisionTreeClassifier",
    "ExtraTreesClassifier", "GradientBoostingClassifier", "XGBClassifier"
}
GROUP_LABELS = {
    "log_amount": "Loan amount",
    "guarantee_pct": "SBA guarantee share",
    "amount_per_job": "Loan amount per job",
    "rate": "Interest rate",
    "jobs": "Jobs supported",
    "zero_jobs": "No jobs reported",
    "is_franchise": "Franchise",
    "naics3": "Industry (NAICS)",
    "BusinessType": "Business type",
    "BusinessAge": "Business age",
    "ProjectState": "Project state",
    "ProcessingMethod": "Processing method",
    "BankName": "Bank / lender",
    "FixedorVariableInterestInd": "Interest rate type",
    "RevolverStatus": "Revolver status",
    "CollateralInd": "Collateral",
}
CAT_GROUPS = sorted(
    [column for column, _label in SELECT_FIELDS] + ["naics3"],
    key=len,
    reverse=True
)
EXPLAINER, GROUPS, SHAP_NOTE = None, None, ""


def group_of(encoded_name):
    """Map encoded feature names back to their original input column."""
    name = encoded_name.split("__", 1)[-1]
    for column in CAT_GROUPS:
        if name.startswith(column + "_"):
            return column
    return name


def init_shap():
    global EXPLAINER, GROUPS, SHAP_NOTE

    try:
        estimator = model.named_steps["model"]
    except (AttributeError, KeyError):
        SHAP_NOTE = "Feature explanations are unavailable for this model pipeline."
        return

    if type(estimator).__name__ not in TREE_MODELS:
        SHAP_NOTE = "Feature explanations are available for tree-based models only."
        return

    try:
        import shap
        feature_names = list(model[:-1].get_feature_names_out())
        EXPLAINER = shap.TreeExplainer(estimator)
        GROUPS = [group_of(name) for name in feature_names]
    except ImportError:
        SHAP_NOTE = "Install SHAP to enable feature explanations."
    except Exception as exc:
        app.logger.exception("SHAP initialization failed")
        SHAP_NOTE = (
            f"Feature explanations are unavailable ({type(exc).__name__})."
        )


init_shap()


def display_value(group, row, parsed):
    if group == "log_amount":
        return f"${parsed['gross']:,.0f}"
    if group == "guarantee_pct":
        return (
            f"{parsed['guaranteed'] / parsed['gross']:.0%}"
            if parsed["gross"] else "Not available"
        )
    if group == "amount_per_job":
        return (
            f"${parsed['gross'] / parsed['jobs']:,.0f} per job"
            if parsed["jobs"] else "no jobs"
        )
    if group == "rate":
        return f"{parsed['rate']:g}%"
    if group == "jobs":
        return f"{parsed['jobs']:g}"
    if group == "zero_jobs":
        return "Yes" if parsed["jobs"] == 0 else "No"
    if group == "is_franchise":
        return "Yes" if parsed["franchise"] else "No"

    if group not in row.index:
        return "Not available"

    value = str(row[group])
    return OPTION_LABEL.get(group, {}).get(value, value)


def explain(X, parsed):
    """Return the five strongest grouped SHAP drivers for this prediction."""
    if EXPLAINER is None or GROUPS is None:
        return [], SHAP_NOTE

    try:
        transformed = model[:-1].transform(X)
        shap_values = EXPLAINER.shap_values(transformed)

        if isinstance(shap_values, list):
            shap_values = (
                shap_values[1] if len(shap_values) > 1 else shap_values[0]
            )

        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = (
                shap_values[0, :, 1]
                if shap_values.shape[-1] > 1
                else shap_values[0, :, 0]
            )
        else:
            shap_values = shap_values.reshape(-1)

        if len(shap_values) != len(GROUPS):
            raise ValueError("SHAP output does not match transformed feature names.")

        grouped = pd.Series(shap_values, index=GROUPS).groupby(level=0).sum()
        top = grouped.reindex(
            grouped.abs().sort_values(ascending=False).index
        ).head(5)

        biggest = float(top.abs().max()) if len(top) else 1.0
        biggest = biggest or 1.0
        row = X.iloc[0]

        drivers = [
            {
                "name": GROUP_LABELS.get(group, group),
                "value": display_value(group, row, parsed),
                "up": bool(value > 0),
                "width": max(4, round(abs(value) / biggest * 100)),
            }
            for group, value in top.items()
        ]
        return drivers, ""
    except Exception as exc:
        app.logger.exception("SHAP explanation failed")
        return [], (
            f"Feature explanation could not be generated ({type(exc).__name__})."
        )


# ---------------------------------------------------------------------------
# Pages and presentation helpers
# ---------------------------------------------------------------------------
def page_context(**extra):
    context = {
        "needed": NEEDED,
        "show": SHOW,
        "options": meta.get("options", {}),
        "ranges": meta.get("ranges", {}),
        "select_fields": [
            (key, label) for key, label in SELECT_FIELDS if key in NEEDED
        ],
        "form": {},
        "errors": [],
        "notices": [],
    }
    context.update(extra)
    return context


def loan_summary(form, values, digits, franchise):
    rows = []

    if "GrossApproval" in NEEDED:
        rows.append(("Gross loan approval", f"${values['GrossApproval']:,.2f}"))
    if "SBAGuaranteedApproval" in NEEDED:
        rows.append((
            "SBA guaranteed amount",
            f"${values['SBAGuaranteedApproval']:,.2f}"
        ))
    if "InitialInterestRate" in NEEDED:
        rows.append((
            "Initial interest rate",
            f"{values['InitialInterestRate']:g}%"
        ))
    if "JobsSupported" in NEEDED:
        rows.append(("Jobs supported", f"{values['JobsSupported']:g}"))
    if "NaicsCode" in NEEDED:
        rows.append(("NAICS industry code", digits))
    if "Franchise" in NEEDED:
        rows.append(("Franchise", "Yes" if franchise == "yes" else "No"))

    for key, label in SELECT_FIELDS:
        if key in NEEDED:
            selected = form.get(key, "")
            display = OPTION_LABEL.get(key, {}).get(selected, selected)
            rows.append((label, display))

    return rows


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", **page_context())


@app.route("/health", methods=["GET"])
def health():
    """Simple health endpoint for deployment checks."""
    return {"status": "ok", "model_loaded": True}, 200


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form.to_dict()
    values, digits, franchise, errors, notices = read_form(form)

    if errors:
        # Do not log submitted loan values; log only validation messages.
        app.logger.warning("FORM VALIDATION ERRORS: %s", errors)
        return render_template(
            "index.html",
            **page_context(
                form=form,
                errors=errors,
                notices=notices,
            )
        ), 400

    try:
        X = make_model_input(form, values, digits, franchise)
        score = float(get_score(model, X)[0])

        if not np.isfinite(score):
            raise ValueError("The model returned a non-finite risk score.")

        # Percentile position among the 2023 test-loan scores.
        percentile = float(np.interp(score, QUANTILES, np.arange(101)))
        tier = (
            "High" if percentile >= 90
            else "Medium" if percentile >= 70
            else "Low"
        )

        parsed = {
            "gross": values["GrossApproval"],
            "guaranteed": values["SBAGuaranteedApproval"],
            "jobs": values["JobsSupported"],
            "rate": values["InitialInterestRate"],
            "franchise": franchise == "yes",
        }
        drivers, shap_note = explain(X, parsed)
        #drivers = []
        #shap_note = "Feature explanations temporarily disabled."

        return render_template(
            "result.html",
            meta=meta,
            score=score,
            percentile=percentile,
            tier=tier,
            tier_class=tier.lower(),
            tier_stats=meta["tiers"][tier],
            base_rate=meta["base_rate_test"],
            notices=notices,
            loan_details=loan_summary(form, values, digits, franchise),
            drivers=drivers,
            shap_note=shap_note,
        )

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            **page_context(
                form=form,
                errors=[
                    "Prediction could not be completed. "
                    f"Please contact the project maintainer with this error type: "
                    f"{type(exc).__name__}."
                ],
                notices=notices,
            )
        ), 500


if __name__ == "__main__":
    app.run(debug=False)
