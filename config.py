"""config.py - all settings in ONE place. Change things here."""
import os

CSV_PATH = "FOIA_7a_FY2020_Present_asof_260930.csv"
DATA_DATE = "2026-09-30"      # "as of" date of the file
HORIZON = 36                  # default = charged off within 36 months of disbursement
TARGET = "default_36m"
SEED = 42
OUT = "outputs"               # graphs, tables and the saved model go here
os.makedirs(OUT, exist_ok=True)

FEATURE_SELECTION = "none"   # "none", "kbest" (SelectKBest) or "rfe" (RFE)
K_FEATURES = 40               # columns to keep when selecting
RUN_SELECTION_COMPARE = False  # compare none / kbest / rfe with Logistic (set False to skip)
N_NEG_SLOW = 15000            # good loans used for the slow models (KNN, RBF-SVM)
SHAP_MODEL = "LightGBM"       # which model to explain with SHAP

# Columns read from the CSV (outcome/date columns are used only for the target and split)
KEEP_COLS = ["GrossApproval", "SBAGuaranteedApproval", "InitialInterestRate",
             "FixedorVariableInterestInd", "RevolverStatus", "CollateralInd",
             "ProcessingMethod", "NaicsCode", "BusinessType", "BusinessAge",
             "FranchiseCode", "JobsSupported", "ProjectState", "BankName",
             "ApprovalDate", "FirstDisbursementDate", "LoanStatus", "ChargeOffDate"]

# Model inputs (created in data_prep.py)
NUM_COLS = ["log_amount", "guarantee_pct", "amount_per_job", "rate",
            "jobs", "zero_jobs", "is_franchise"]
CAT_COLS = ["naics3", "BusinessAge", "ProjectState", "ProcessingMethod",
            "FixedorVariableInterestInd", "RevolverStatus", "CollateralInd",
            "BusinessType", "BankName"]
# Columns you decided are unnecessary: put their names here and re-run.
# Find candidates with:  python column_check.py
# Example: DROP_COLS = ["is_franchise", "RevolverStatus"]
DROP_COLS = []
NUM_COLS = [c for c in NUM_COLS if c not in DROP_COLS]
CAT_COLS = [c for c in CAT_COLS if c not in DROP_COLS]
FEATURES = NUM_COLS + CAT_COLS