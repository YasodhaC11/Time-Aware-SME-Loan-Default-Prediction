"""data_prep.py - load the CSV, build the target, create features, split by time."""
import numpy as np
import pandas as pd

import config as cfg


def load_data():
    """CSV -> one clean DataFrame with features, target and approval_year."""
    data = pd.read_csv(cfg.CSV_PATH, usecols=cfg.KEEP_COLS, low_memory=False)
    for c in ["ApprovalDate", "FirstDisbursementDate", "ChargeOffDate"]:
        data[c] = pd.to_datetime(data[c], errors="coerce")

    # Keep real, disbursed loans. Paid-in-full and exempt stay (label 0 = good loan);
    # cancelled / never-funded loans are dropped.
    data = data[data["LoanStatus"].isin(["CHGOFF", "P I F", "EXEMPT"])]
    data = data[data["FirstDisbursementDate"].notna()]

    # Keep only loans watched for the FULL 36 months. Newer loans haven't had time
    # to default, so labelling them 0 would teach the model wrong answers.
    cutoff = pd.Timestamp(cfg.DATA_DATE) - pd.DateOffset(months=cfg.HORIZON)
    data = data[data["FirstDisbursementDate"] <= cutoff].copy()

    window_end = data["FirstDisbursementDate"] + pd.DateOffset(months=cfg.HORIZON)
    data[cfg.TARGET] = (data["ChargeOffDate"].notna() &
                        (data["ChargeOffDate"] <= window_end)).astype(int)
    data["approval_year"] = data["ApprovalDate"].dt.year      # used ONLY for splitting
    data = data[data["approval_year"] <= 2023]

    d = add_features(data)
    print("Rows:", len(d), "| default rate:", round(d[cfg.TARGET].mean(), 4))
    return d


def add_features(data):
    """Create model inputs from raw columns and tidy the text columns."""
    d = data.copy()
    d["log_amount"] = np.log1p(d["GrossApproval"])             # loan sizes are skewed. Log compresses them so the model isn’t dominated by outliers.
    d["guarantee_pct"] = d["SBAGuaranteedApproval"] / d["GrossApproval"] # SBAGuaranteedApproval = The SBA promises to cover part of the bank’s loss if the borrower defaults
    d["jobs"] = pd.to_numeric(d["JobsSupported"], errors="coerce") # Example: a restaurant says the loan supports 8 jobs.
    d["zero_jobs"] = (d["jobs"] == 0).astype(int)              # "no jobs" is a signal - one-person operation
    d["amount_per_job"] = np.log1p(d["GrossApproval"] / d["jobs"].replace(0, np.nan)) # Amount per job — also useful to stabilize very high ratios.
    d["rate"] = pd.to_numeric(d["InitialInterestRate"], errors="coerce") # Example: 7.0% vs 11.5%. Higher rate usually means riskier loan.
    d["naics3"] = d["NaicsCode"].astype("string").str[:3]           # industry group. With .astype("string"), the blank stays a real missing value, so fillna("MISSING") catches it and labels it "MISSING"
    d["is_franchise"] = d["FranchiseCode"].notna().astype(int)  # FranchiseCode != empty → franchise

    for c in cfg.CAT_COLS:                                      # explicit "MISSING" level
        d[c] = d[c].fillna("MISSING").astype(str).str.upper().str.strip()
    return d


def split_train_test(d):
    """Time-based split. Train on the PAST (<=2022), test on a LATER year (2023).
    Never split randomly: default rates change a lot by year."""
    train = d[d.approval_year <= 2022].reset_index(drop=True)
    test = d[d.approval_year == 2023].reset_index(drop=True)
    print(f"Train: {len(train)} rows, {train[cfg.TARGET].sum()} defaults | "
          f"Test (2023): {len(test)} rows, {test[cfg.TARGET].sum()} defaults")
    return train, test


def time_folds(years):
    """Cross-validation by year (always train on the past, validate on the next year):
       train 2019-2020 -> validate 2021
       train 2019-2021 -> validate 2022"""
    years = np.asarray(years)
    return [(np.where(years <= last)[0], 
            np.where(years == val)[0])
            for last, val in [(2020, 2021), (2021, 2022)]]


def undersample(df, n_neg=cfg.N_NEG_SLOW):
    """All defaults + a random sample of good loans. Used only for slow models
    (KNN, RBF-SVM) so they finish in minutes instead of hours."""
    pos = df[df[cfg.TARGET] == 1]
    neg_all = df[df[cfg.TARGET] == 0]
    neg = neg_all.sample(min(n_neg, len(neg_all)), random_state=cfg.SEED)
    return pd.concat([pos, neg]).reset_index(drop=True)
