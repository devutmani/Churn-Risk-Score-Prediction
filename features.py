"""Shared data cleaning + feature engineering (used by train.py and app.py)."""
import numpy as np
import pandas as pd

TARGET = "churn_risk_score"
REF_DATE = pd.Timestamp("2018-01-01")          # dataset joining dates end 2017-12-31
ID_COLS = ["customer_id", "Name", "security_no"]

CAT_COLS = ["gender", "region_category", "membership_category", "joined_through_referral",
            "preferred_offer_types", "medium_of_operation", "internet_option",
            "used_special_discount", "offer_application_preference", "past_complaint",
            "complaint_status", "feedback"]

NUM_COLS = ["age", "days_since_last_login", "avg_time_spent", "avg_transaction_value",
            "avg_frequency_login_days", "points_in_wallet", "tenure_days", "join_year",
            "join_month", "visit_hour", "has_referral_id", "missing_login_gap",
            "missing_time_spent", "missing_freq", "missing_points"]

MODEL_COLS = CAT_COLS + NUM_COLS

RISK_BAND = {1: "Low", 2: "Low", 3: "Medium", 4: "High", 5: "High"}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready frame (raw -> cleaned + engineered features)."""
    df = df.copy()
    df = df.replace("?", np.nan)

    # numeric columns with sentinel / invalid values -> NaN (+ missing flag)
    df["avg_frequency_login_days"] = pd.to_numeric(df["avg_frequency_login_days"], errors="coerce")
    df["missing_login_gap"] = (df["days_since_last_login"] < 0).astype(int)
    df.loc[df["days_since_last_login"] < 0, "days_since_last_login"] = np.nan
    df["missing_time_spent"] = (df["avg_time_spent"] < 0).astype(int)
    df.loc[df["avg_time_spent"] < 0, "avg_time_spent"] = np.nan
    df["missing_freq"] = (df["avg_frequency_login_days"].isna() | (df["avg_frequency_login_days"] < 0)).astype(int)
    df.loc[df["avg_frequency_login_days"] < 0, "avg_frequency_login_days"] = np.nan
    df["missing_points"] = (df["points_in_wallet"].isna() | (df["points_in_wallet"] < 0)).astype(int)
    df.loc[df["points_in_wallet"] < 0, "points_in_wallet"] = np.nan

    # dates / times
    jd = pd.to_datetime(df["joining_date"], errors="coerce")
    df["tenure_days"] = (REF_DATE - jd).dt.days
    df["join_year"] = jd.dt.year
    df["join_month"] = jd.dt.month
    df["visit_hour"] = pd.to_numeric(df["last_visit_time"].astype(str).str[:2], errors="coerce")

    # referral id present? ('xxxxxxxx' means no referral)
    df["has_referral_id"] = (df["referral_id"].notna() & (df["referral_id"] != "xxxxxxxx")).astype(int)

    # categorical missing -> explicit "Unknown"
    for c in CAT_COLS:
        df[c] = df[c].fillna("Unknown").astype(str)
    return df
