"""make_app_meta.py - builds outputs/app_meta.json for the web app.

Run after main.py (it needs outputs/best_model.pkl):   python make_app_meta.py
Re-run it whenever you retrain or save a different model.

It creates:
  - dropdown options taken from the TRAINING data (so users cannot mistype a lender)
  - typical value ranges (to warn about unusual inputs)
  - risk-tier cut-offs from the 2023 test year (out-of-sample scores) and the
    default rate actually observed in each tier
"""
import json

import joblib
import numpy as np

import config as cfg
from data_prep import load_data, split_train_test
from evaluate import get_score

MODEL_PATH = f"{cfg.OUT}/best_model.pkl"
META_PATH = f"{cfg.OUT}/app_meta.json"
MIN_LEVEL = 200            # same rare-level rule as the pipeline's OneHotEncoder

SELECT_COLS = ["BusinessType", "BusinessAge", "ProjectState", "ProcessingMethod",
               "FixedorVariableInterestInd", "RevolverStatus", "CollateralInd",
               "BankName"]

# Friendlier labels for coded values. Unknown values simply show as they are.
LABELS = {
    "FixedorVariableInterestInd": {"F": "Fixed", "V": "Variable"},
    "CollateralInd": {"Y": "Yes", "N": "No"},
    "RevolverStatus": {"Y": "Yes (revolving)", "N": "No (term loan)",
                       "1": "Yes (revolving)", "0": "No (term loan)"},
}

if __name__ == "__main__":
    d = load_data()
    train, test = split_train_test(d)
    model = joblib.load(MODEL_PATH)
    active = list(model.feature_names_in_)          # columns this model really uses
    model_name = type(model[-1]).__name__
    print("Model:", model_name, "| columns used:", active)

    # ---- dropdown options from training data --------------------------------
    options = {}
    for c in SELECT_COLS:
        if c not in active:
            continue
        col = train[c].fillna("MISSING").astype(str).str.upper().str.strip()
        counts = col.value_counts()
        levels = sorted(counts[counts >= MIN_LEVEL].index.tolist())
        if ((counts < MIN_LEVEL).any() or not levels) and "OTHER" not in levels:
            levels.append("OTHER")                  # rare levels are handled as 'other'
        options[c] = [{"value": v,
                       "label": LABELS.get(c, {}).get(v, v) if v != "OTHER"
                       else "OTHER (rare in training data)"} for v in levels]

    naics3_levels = []
    if "naics3" in active:
        c = train["naics3"].value_counts()
        naics3_levels = sorted(c[c >= MIN_LEVEL].index.tolist())

    # ---- typical ranges (1st, 50th, 99th percentile) -------------------------
    def rng(col):
        return [float(v) for v in train[col].quantile([0.01, 0.5, 0.99])]
    ranges = {"amount": rng("GrossApproval"), "rate": rng("InitialInterestRate"),
              "jobs": rng("jobs")}

    # ---- risk tiers from the 2023 test year (out-of-sample) ------------------
    s = get_score(model, test[active])
    y = test[cfg.TARGET].to_numpy()
    q = np.percentile(s, np.arange(101))
    tier = np.where(s >= q[90], "High", np.where(s >= q[70], "Medium", "Low"))
    tiers = {}
    for t in ["Low", "Medium", "High"]:
        m = tier == t
        tiers[t] = {"n": int(m.sum()), "default_rate": float(y[m].mean()),
                    "share_of_defaults": float(y[m].sum() / y.sum())}

    # ---- print what the app will use, so you can sanity-check it -------------
    print("\nDropdown options built from your data (check these look right):")
    for c, opts in options.items():
        shown = ", ".join(o["value"] for o in opts[:6])
        print(f"  {c:28s} {len(opts):3d} options: {shown}{' ...' if len(opts) > 6 else ''}")
    print("\nTypical ranges (1st / 50th / 99th percentile):")
    for k, v in ranges.items():
        print(f"  {k:7s} {v[0]:,.2f}  {v[1]:,.2f}  {v[2]:,.2f}")
    print("\nRisk tiers on the 2023 test year (default rate should rise Low -> High):")
    for t, v in tiers.items():
        print(f"  {t:6s} n={v['n']:6d}  default rate {v['default_rate']:.2%}  "
              f"share of all defaults {v['share_of_defaults']:.1%}")

    meta = {"active_features": active, "model_name": model_name,
            "options": options, "naics3_levels": naics3_levels, "ranges": ranges,
            "quantiles": [float(v) for v in q], "tiers": tiers,
            "base_rate_test": float(y.mean()), "test_year": 2023,
            "train_years": "2019-2022", "horizon_months": cfg.HORIZON}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {META_PATH}.  Next:  python app.py")
