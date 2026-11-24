import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import CAT_COLS, ID_COLS, MODEL_COLS, NUM_COLS, RISK_BAND, TARGET, clean

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])


def make_models() -> dict:
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(objective="multi:softprob", n_estimators=300, max_depth=5,
                                 learning_rate=0.08, subsample=0.9, colsample_bytree=0.8,
                                 tree_method="hist", n_jobs=-1, random_state=SEED),
    }


def data_quality_table(raw: pd.DataFrame) -> pd.DataFrame:
    n = len(raw)
    freq = pd.to_numeric(raw["avg_frequency_login_days"], errors="coerce")
    items = [
        ("Target = -1 (invalid label)", (raw[TARGET] == -1).sum(), "Rows dropped from training"),
        ("days_since_last_login = -999", (raw["days_since_last_login"] < 0).sum(), "Set to NaN, median-imputed, flag added"),
        ("avg_time_spent negative", (raw["avg_time_spent"] < 0).sum(), "Set to NaN, median-imputed, flag added"),
        ("avg_frequency_login_days = 'Error'", (raw["avg_frequency_login_days"] == "Error").sum(), "Set to NaN, median-imputed, flag added"),
        ("avg_frequency_login_days negative", (freq < 0).sum(), "Set to NaN, median-imputed, flag added"),
        ("points_in_wallet missing", raw["points_in_wallet"].isna().sum(), "Median-imputed, flag added"),
        ("points_in_wallet negative", (raw["points_in_wallet"] < 0).sum(), "Set to NaN, median-imputed, flag added"),
        ("region_category missing", raw["region_category"].isna().sum(), "Category 'Unknown'"),
        ("joined_through_referral = '?'", (raw["joined_through_referral"] == "?").sum(), "Category 'Unknown'"),
        ("medium_of_operation = '?'", (raw["medium_of_operation"] == "?").sum(), "Category 'Unknown'"),
        ("preferred_offer_types missing", raw["preferred_offer_types"].isna().sum(), "Category 'Unknown'"),
    ]
    return pd.DataFrame([{"Issue": a, "Rows affected": int(b), "Share %": round(100 * b / n, 2),
                          "Treatment": c} for a, b, c in items])


def group_importance(names, values) -> pd.DataFrame:
    """Collapse one-hot columns back to their original feature."""
    groups = {}
    for nm, v in zip(names, values):
        base = nm.split("__", 1)[1]
        if nm.startswith("cat__"):
            base = max((c for c in CAT_COLS if base.startswith(c + "_")), key=len, default=base)
        groups[base] = groups.get(base, 0.0) + float(v)
    s = pd.Series(groups).sort_values(ascending=False)
    return pd.DataFrame({"feature": s.index, "importance": s.values.round(4)})


def main():
    t0 = time.time()
    raw = pd.read_csv(ROOT / "data" / "train.csv")
    test_raw = pd.read_csv(ROOT / "data" / "test.csv")
    data_quality_table(raw).to_csv(ART / "data_quality.csv", index=False)

    df = clean(raw)
    df = df[df[TARGET] != -1].reset_index(drop=True)          # drop invalid labels
    X, y = df[MODEL_COLS], df[TARGET] - 1                      # classes 0..4 = scores 1..5
    print(f"Clean training rows: {len(df)} (dropped {len(raw) - len(df)} invalid labels)")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted, preds = [], {}, {}
    for name, model in make_models().items():
        pipe = Pipeline([("prep", preprocessor()), ("clf", model)])
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1).mean()
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        rows.append({"Model": name, "CV macro-F1": cv_f1,
                     "Test accuracy": accuracy_score(y_te, p),
                     "Test macro-F1": f1_score(y_te, p, average="macro"),
                     "Test weighted-F1": f1_score(y_te, p, average="weighted")})
        fitted[name], preds[name] = pipe, p
        print(f"{name:22s} CV F1={cv_f1:.3f}  test acc={rows[-1]['Test accuracy']:.3f}  "
              f"test F1={rows[-1]['Test macro-F1']:.3f}")
    comp = pd.DataFrame(rows).round(4).sort_values("CV macro-F1", ascending=False)
    comp.to_csv(ART / "model_comparison.csv", index=False)

    best_name = comp.iloc[0]["Model"]
    best, bp = fitted[best_name], preds[best_name]
    print("Best model (by CV macro-F1):", best_name)

    # per-class report + confusion matrix (scores shown as 1..5)
    labels = [1, 2, 3, 4, 5]
    rep = pd.DataFrame(classification_report(y_te + 1, bp + 1, labels=labels, output_dict=True,
                                             zero_division=0)).T.round(3)
    rep.to_csv(ART / "classification_report.csv")
    cm = pd.DataFrame(confusion_matrix(y_te + 1, bp + 1, labels=labels),
                      index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    cm.to_csv(ART / "confusion_matrix.csv")

    # SHAP / importance (tree models) on a sample of the hold-out set
    prep, clf = best.named_steps["prep"], best.named_steps["clf"]
    names = prep.get_feature_names_out()
    sample = X_te.sample(min(1500, len(X_te)), random_state=SEED)
    Xs = prep.transform(sample)
    if hasattr(clf, "feature_importances_"):
        sv = shap.TreeExplainer(clf).shap_values(Xs)
        sv = np.stack(sv, -1) if isinstance(sv, list) else sv
        imp_vals = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(0)
    else:
        imp_vals = np.abs(clf.coef_).mean(0)
    group_importance(names, imp_vals).to_csv(ART / "feature_importance.csv", index=False)

    # EDA tables
    clean_train = raw[raw[TARGET] != -1]
    for col, fname in [("membership_category", "insights_membership.csv"),
                       ("feedback", "insights_feedback.csv"),
                       ("complaint_status", "insights_complaint.csv"),
                       ("medium_of_operation", "insights_medium.csv")]:
        (pd.crosstab(clean_train[col], clean_train[TARGET], normalize="index") * 100).round(1) \
            .rename(columns=lambda c: f"Score {c} %").to_csv(ART / fname)
    pd.DataFrame({"score": labels, "customers": clean_train[TARGET].value_counts().reindex(labels).values,
                  "band": [RISK_BAND[l] for l in labels]}).to_csv(ART / "risk_distribution.csv", index=False)

    # refit best on ALL clean training data, then score test.csv
    final = Pipeline([("prep", preprocessor()), ("clf", make_models()[best_name])]).fit(X, y)
    Xt = clean(test_raw)[MODEL_COLS]
    proba = final.predict_proba(Xt)
    pred = proba.argmax(1) + 1
    out = pd.DataFrame({"customer_id": test_raw["customer_id"], "name": test_raw["Name"],
                        "membership": test_raw["membership_category"], "feedback": test_raw["feedback"],
                        "predicted_score": pred,
                        "risk_band": [RISK_BAND[s] for s in pred],
                        "confidence": proba.max(1).round(3)})
    for i, l in enumerate(labels):
        out[f"p_score_{l}"] = proba[:, i].round(3)
    out["expected_score"] = (proba @ np.array(labels)).round(2)
    out.to_csv(ART / "test_predictions.csv", index=False)

    # schema for the single-customer form
    schema = {"cat": {c: sorted(df[c].unique().tolist()) for c in CAT_COLS},
              "num": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                          "median": float(df[c].median())} for c in
                      ["age", "days_since_last_login", "avg_time_spent", "avg_transaction_value",
                       "avg_frequency_login_days", "points_in_wallet", "tenure_days", "visit_hour"]},
              "mode": {c: df[c].mode()[0] for c in CAT_COLS}}
    (ART / "schema.json").write_text(json.dumps(schema, indent=2))

    best_row = comp.iloc[0]
    meta = {"best_model": best_name, "train_rows_raw": len(raw), "train_rows_clean": len(df),
            "test_rows": len(test_raw), "features_used": len(MODEL_COLS),
            "test_accuracy": float(best_row["Test accuracy"]),
            "test_macro_f1": float(best_row["Test macro-F1"]),
            "cv_macro_f1": float(best_row["CV macro-F1"]),
            "high_risk_share_test_pred": float((out["risk_band"] == "High").mean())}
    (ART / "metrics.json").write_text(json.dumps(meta, indent=2))
    joblib.dump({"pipeline": final, "model_name": best_name}, ART / "model.joblib")
    print("Done in %.0fs" % (time.time() - t0), meta)


if __name__ == "__main__":
    main()