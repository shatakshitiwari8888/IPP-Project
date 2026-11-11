"""
=============================================================================
CREDIT RISK AND LOAN DEFAULT PREDICTION
Course: Introduction to Python Programming (College Project)
Model: Multiple Logistic Regression (Econometric / FinTech Approach)
Dataset: LendingClub Accepted Loans (2007-2018Q4)
Target: Probability of Default (PD) for Retail Loan Applicants
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# CONFIGURATION
DATA_FILE_PATH = "data/accepted_2007_2018Q4.csv"
OUTPUT_DIR = "outputs"
FINAL_FEATURES = ["annual_inc", "dti", "fico_range_low", "loan_amnt", "LTI"]

# Calibrated cutoffs: class_weight='balanced' inflates predicted PD,
# so cutoffs are tuned to ensure monotonic and realistic grade spreads.
GRADE_CUTOFFS = {
    "A": 0.32,
    "B": 0.44,
    "C": 0.54,
    "D": 0.64,
    "E": 0.74,
    "F": 1.00
}

INTEREST_RATES = {
    "A": 0.0699,  #  6.99% APR
    "B": 0.0999,  #  9.99% APR
    "C": 0.1399,  # 13.99% APR
    "D": 0.1799,  # 17.99% APR
    "E": 0.2299,  # 22.99% APR
    "F": 0.2899   # 28.99% APR
}

def load_data(file_path: str = DATA_FILE_PATH) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 1: RAW DATA INGESTION")
    print("=" * 70)
    if not os.path.exists(file_path):
        import generate_sample_data
        generate_sample_data.create_sample_csv(file_path, n_rows=5000)
    try:
        header_sample = pd.read_csv(file_path, nrows=2)
        candidate_cols = ["id", "member_id", "loan_status", "annual_inc", "dti", "fico_range_low", "loan_amnt"]
        cols_to_load = [c for c in candidate_cols if c in header_sample.columns]
        df = pd.read_csv(file_path, usecols=cols_to_load, low_memory=False)
        print(f"[✓] Successfully loaded {len(df):,} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 70)
    df = raw_df.copy()
    # String handling
    df["loan_status"] = df["loan_status"].astype(str).str.strip()
    
    # Filter resolved loans only (avoids Look-Ahead Bias)
    resolved_mask = df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])
    df = df[resolved_mask].copy()
    df["target"] = df["loan_status"].apply(lambda s: 0 if s == "Fully Paid" else 1)
    
    # Drop identifiers
    drop_cols = [c for c in ["id", "member_id", "loan_status"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    
    # Complete-case analysis: drop rows with missing values
    numeric_cols = ["annual_inc", "dti", "fico_range_low", "loan_amnt"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=numeric_cols, inplace=True)
            
    # Edge cases & clipping
    df["annual_inc"] = df["annual_inc"].apply(lambda inc: max(1000.0, float(inc)))
    inc_upper = float(df["annual_inc"].quantile(0.995))
    df["annual_inc"] = np.clip(df["annual_inc"], a_min=1000.0, a_max=inc_upper)
    dti_upper = min(45.0, float(df["dti"].quantile(0.99)))
    df["dti"] = np.clip(df["dti"], a_min=0.0, a_max=dti_upper)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)
    df["LTI"] = df["loan_amnt"] / df["annual_inc"]
    df["LTI"] = np.clip(df["LTI"], a_min=0.0, a_max=1.5)
    return df

def split_and_scale(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 4 & 5: PARTITIONING (80/20 STRATIFIED) & NORMALIZATION")
    print("=" * 70)
    X = df[FINAL_FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Fit scaler strictly on train to prevent Data Leakage
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=FINAL_FEATURES, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FINAL_FEATURES, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test

def train_model(X_train_scaled: pd.DataFrame, y_train: pd.Series):
    print("\n" + "=" * 70)
    print("STEP 6: LOGISTIC REGRESSION TRAINING")
    print("=" * 70)
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    sm_model = None
    if HAS_STATSMODELS:
        try:
            X_train_sm = sm.add_constant(X_train_scaled)
            sm_model = sm.Logit(y_train, X_train_sm).fit(disp=False)
        except Exception:
            pass
    return clf, sm_model

def evaluate_model(clf, sm_model, X_test_scaled, y_test):
    print("\n" + "=" * 70)
    print("STEP 7: PROBABILITY GENERATION & EVALUATION")
    print("=" * 70)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = clf.predict(X_test_scaled)
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Safe (0)", "Default (1)"]))
    return y_prob, auc_score

def assign_grade(pd_val: float) -> str:
    pd_val = max(0.0, min(1.0, pd_val))
    if pd_val < GRADE_CUTOFFS["A"]: return "A"
    elif pd_val < GRADE_CUTOFFS["B"]: return "B"
    elif pd_val < GRADE_CUTOFFS["C"]: return "C"
    elif pd_val < GRADE_CUTOFFS["D"]: return "D"
    elif pd_val < GRADE_CUTOFFS["E"]: return "E"
    else: return "F"

def score_applicant(profile_dict: dict, clf, scaler) -> dict:
    annual_inc = float(max(1000.0, profile_dict["annual_inc"]))
    loan_amnt = float(profile_dict["loan_amnt"])
    dti = float(profile_dict["dti"])
    fico = float(profile_dict["fico_range_low"])
    lti = loan_amnt / annual_inc
    applicant_df = pd.DataFrame([{
        "annual_inc": annual_inc,
        "dti": dti,
        "fico_range_low": fico,
        "loan_amnt": loan_amnt,
        "LTI": lti
    }])[FINAL_FEATURES]
    scaled = scaler.transform(applicant_df)
    pd_score = float(clf.predict_proba(scaled)[0, 1])
    grade = assign_grade(pd_score)
    apr = INTEREST_RATES[grade]
    decision = "Approved" if grade in ["A", "B", "C", "D"] else "Manual Review / High Risk"
    return {
        "PD": pd_score,
        "grade": grade,
        "apr": apr,
        "decision": decision
    }
