import joblib
import config as cfg
from pathlib import Path
from data_prep import load_data, split_train_test

model = joblib.load(Path(cfg.OUT) / "best_model.pkl")

data = load_data()
train, test = split_train_test(data)

X_test = test[cfg.FEATURES]

test["predicted_score"] = model.predict_proba(X_test)[:, 1]

high_risk = test[test["predicted_score"] >= 0.50]

high_risk = high_risk.sort_values(
    "predicted_score", ascending=False
)

print("High-risk records found:", len(high_risk))

if not high_risk.empty:
    print(high_risk[
        [
            "GrossApproval",
            "SBAGuaranteedApproval",
            "InitialInterestRate",
            "JobsSupported",
            "BusinessType",
            "BusinessAge",
            "NaicsCode",
            "ProjectState",
            "BankName",
            "predicted_score"
        ]
    ].head(5).to_string(index=False))
else:
    print("No test records crossed the 50% threshold.")