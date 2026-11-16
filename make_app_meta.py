"""Build outputs/app_meta.json for the Flask application.

Run locally after main.py:
    python make_app_meta.py

Run again whenever you retrain or replace outputs/best_model.pkl.
The CSV is needed locally to build metadata; it is not read by the deployed
Flask app when users submit predictions.
"""

import json
from pathlib import Path

import joblib
import numpy as np

import config as cfg
from data_prep import load_data, split_train_test
from evaluate import get_score

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / cfg.OUT / "best_model.pkl"
META_PATH = BASE / cfg.OUT / "app_meta.json"

# Keep this consistent with the min_frequency / rare-category policy used by
# the OneHotEncoder in the training pipeline.
MIN_LEVEL = 200

SELECT_COLS = [
    "BusinessType",
    "BusinessAge",
    "ProjectState",
    "ProcessingMethod",
    "FixedorVariableInterestInd",
    "RevolverStatus",
    "CollateralInd",
    "BankName",
]

LABELS = {
    "FixedorVariableInterestInd": {"F": "Fixed", "V": "Variable"},
    "CollateralInd": {"Y": "Yes", "N": "No"},
    "RevolverStatus": {
        "Y": "Yes (revolving)",
        "N": "No (term loan)",
        "1": "Yes (revolving)",
        "0": "No (term loan)",
    },
}


def numeric_range(series):
    """Return the 1st, 50th and 99th percentiles as JSON-safe floats."""
    values = series.dropna()
    if values.empty:
        return [0.0, 0.0, 0.0]
    return [float(value) for value in values.quantile([0.01, 0.50, 0.99])]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run python main.py first."
        )

    data = load_data()
    train, test = split_train_test(data)

    if train.empty:
        raise ValueError("Training split is empty; cannot create app metadata.")
    if test.empty:
        raise ValueError("2023 test split is empty; cannot create risk tiers.")

    model = joblib.load(MODEL_PATH)
    active = list(model.feature_names_in_)
    model_name = type(model[-1]).__name__

    print("Model:", model_name, "| columns used:", active)

    # Dropdown options come from normalized training columns. add_features()
    # uppercases and strips categorical values in the same way at prediction.
    options = {}
    for column in SELECT_COLS:
        if column not in active:
            continue

        normalized = (
            train[column]
            .fillna("MISSING")
            .astype(str)
            .str.upper()
            .str.strip()
        )
        counts = normalized.value_counts()
        levels = sorted(counts[counts >= MIN_LEVEL].index.tolist())

        # The encoder's configured rare-category handling must support this
        # catch-all option. Verify that configuration in the training pipeline.
        if (counts.lt(MIN_LEVEL).any() or not levels) and "OTHER" not in levels:
            levels.append("OTHER")

        options[column] = [
            {
                "value": value,
                "label": (
                    LABELS.get(column, {}).get(value, value)
                    if value != "OTHER"
                    else "OTHER (rare in training data)"
                ),
            }
            for value in levels
        ]

    naics3_levels = []
    if "naics3" in active:
        naics_counts = train["naics3"].dropna().astype(str).value_counts()
        naics3_levels = sorted(
            naics_counts[naics_counts >= MIN_LEVEL].index.tolist()
        )

    ranges = {
        "amount": numeric_range(train["GrossApproval"]),
        "rate": numeric_range(train["InitialInterestRate"]),
        "jobs": numeric_range(train["jobs"]),
    }

    # Risk tiers are based on out-of-sample scores from the 2023 test year.
    scores = np.asarray(get_score(model, test[active]), dtype=float).reshape(-1)
    targets = test[cfg.TARGET].to_numpy()

    if len(scores) != len(targets):
        raise ValueError("Model scores and test targets have different lengths.")
    if not np.isfinite(scores).all():
        raise ValueError("Test predictions contain non-finite values.")
    if targets.sum() == 0:
        raise ValueError("The 2023 test split contains no positive defaults.")

    quantiles = np.percentile(scores, np.arange(101))
    tier_labels = np.where(
        scores >= quantiles[90],
        "High",
        np.where(scores >= quantiles[70], "Medium", "Low"),
    )

    tiers = {}
    for tier_name in ["Low", "Medium", "High"]:
        mask = tier_labels == tier_name
        count = int(mask.sum())
        defaults = int(targets[mask].sum())
        tiers[tier_name] = {
            "n": count,
            "default_rate": float(targets[mask].mean()) if count else 0.0,
            "share_of_defaults": float(defaults / targets.sum()),
        }

    print("\nDropdown options:")
    for column, column_options in options.items():
        shown = ", ".join(option["value"] for option in column_options[:6])
        suffix = " ..." if len(column_options) > 6 else ""
        print(f"  {column:28s} {len(column_options):3d} options: {shown}{suffix}")

    print("\nTypical ranges (1st / 50th / 99th percentile):")
    for key, values in ranges.items():
        print(
            f"  {key:7s} {values[0]:,.2f}  "
            f"{values[1]:,.2f}  {values[2]:,.2f}"
        )

    print("\nRisk tiers on 2023 test year:")
    for tier_name, stats in tiers.items():
        print(
            f"  {tier_name:6s} n={stats['n']:6d} "
            f"default rate {stats['default_rate']:.2%} "
            f"share of all defaults {stats['share_of_defaults']:.1%}"
        )

    metadata = {
        "active_features": active,
        "model_name": model_name,
        "options": options,
        "naics3_levels": naics3_levels,
        "ranges": ranges,
        "quantiles": [float(value) for value in quantiles],
        "tiers": tiers,
        "base_rate_test": float(targets.mean()),
        "test_year": 2023,
        "train_years": "2019-2022",
        "horizon_months": cfg.HORIZON,
    }

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {META_PATH}. Next: deploy app.py and app_meta.json.")


if __name__ == "__main__":
    main()
