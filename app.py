"""Customer Churn Risk - results dashboard.  Run: streamlit run app.py  ->  http://localhost:8501"""
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import MODEL_COLS, REF_DATE, RISK_BAND

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="Churn Risk Dashboard", page_icon="📉", layout="wide")

if not (ART / "model.joblib").exists():
    st.error("Model nahi mila. Pehle terminal mein `python train.py` chalayein, phir app dobara run karein.")
    st.stop()


@st.cache_data
def csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(ART / name, **kw)


@st.cache_resource
def bundle():
    return joblib.load(ART / "model.joblib")


meta = json.loads((ART / "metrics.json").read_text())
schema = json.loads((ART / "schema.json").read_text())
model = bundle()["pipeline"]

st.title("📉 Customer Churn Risk Prediction System")
st.caption(f"Model: **{meta['best_model']}** | Target: `churn_risk_score` (1 = lowest risk, 5 = highest risk) | "
           f"Bands: 1-2 Low, 3 Medium, 4-5 High")

tabs = st.tabs(["Overview", "Data quality", "Insights", "Models", "Churn drivers",
                "Test predictions", "Predict a customer"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Training rows (clean)", f"{meta['train_rows_clean']:,}",
                f"-{meta['train_rows_raw'] - meta['train_rows_clean']:,} invalid labels", delta_color="off")
    c[1].metric("Test rows scored", f"{meta['test_rows']:,}")
    c[2].metric("Hold-out accuracy", f"{meta['test_accuracy']:.1%}")
    c[3].metric("Hold-out macro-F1", f"{meta['test_macro_f1']:.3f}")
    c[4].metric("High-risk (test pred.)", f"{meta['high_risk_share_test_pred']:.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk score distribution (training)")
        dist = csv("risk_distribution.csv")
        st.bar_chart(dist.set_index("score")["customers"])
        st.dataframe(dist, hide_index=True)
    with right:
        st.subheader("Predicted risk bands on test.csv")
        preds = csv("test_predictions.csv")
        band = preds["risk_band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
        st.bar_chart(band)
        st.dataframe(pd.DataFrame({"Band": band.index, "Customers": band.values,
                                   "Share %": (100 * band.values / band.sum()).round(1)}),
                     hide_index=True)
    st.info("Analogy: yeh system ek bank ke 'credit officer' jaisa hai jo har customer ko 1 se 5 tak risk "
            "number deta hai, taake company sab se risky customers ko pehle retention offer de.")

# ------------------------------------------------------------------ Data quality
with tabs[1]:
    st.subheader("Data problems found in train.csv and how they were handled")
    st.dataframe(csv("data_quality.csv"), hide_index=True)
    st.caption("Invalid (-1) labels were ~3% and spread uniformly over every feature, so they are label noise "
               "and were removed from training. test.csv may contain such customers too; the model predicts 1-5 only.")

# ------------------------------------------------------------------ Insights
with tabs[2]:
    st.subheader("Kaun se customers zyada risky hain? (row-wise %, har row 100%)")
    for title, f in [("Membership category", "insights_membership.csv"),
                     ("Feedback given", "insights_feedback.csv"),
                     ("Complaint status", "insights_complaint.csv"),
                     ("Medium of operation", "insights_medium.csv")]:
        st.markdown(f"**{title}**")
        t = csv(f, index_col=0)
        st.dataframe(t.style.background_gradient(cmap="Reds", axis=None).format("{:.1f}"))
    st.success("Pattern: Basic/No Membership walon ka score aksar 4-5 hota hai, Platinum/Premium ka 1-3. "
               "Negative feedback (poor service/website/product/ads) wale customers hamesha score 3-5 mein hain.")

# ------------------------------------------------------------------ Models
with tabs[3]:
    st.subheader("Model comparison (5-fold CV on train split, then hold-out test)")
    comp = csv("model_comparison.csv")
    st.dataframe(comp.style.highlight_max(subset=comp.columns[1:], color="#c6efce")
                 .format({c: "{:.3f}" for c in comp.columns[1:]}),
                 hide_index=True)
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Per-class report ({meta['best_model']})**")
        st.dataframe(csv("classification_report.csv", index_col=0))
    with b:
        st.markdown("**Confusion matrix (hold-out)**")
        cm = csv("confusion_matrix.csv", index_col=0)
        st.dataframe(cm.style.background_gradient(cmap="Blues", axis=None))
        st.caption("Ghalatiyan zyada tar paas ke scores mein hain (jaise 4 vs 5), door ke scores mein nahi.")

# ------------------------------------------------------------------ Drivers
with tabs[4]:
    st.subheader("Model ke hisaab se sab se important features (SHAP)")
    imp = csv("feature_importance.csv")
    st.bar_chart(imp.head(12).set_index("feature")["importance"])
    st.dataframe(imp, hide_index=True)

# ------------------------------------------------------------------ Test predictions
with tabs[5]:
    st.subheader("Predictions for test.csv")
    preds = csv("test_predictions.csv")
    f1, f2, f3 = st.columns(3)
    bands = f1.multiselect("Risk band", ["Low", "Medium", "High"], default=["High"])
    mems = f2.multiselect("Membership", sorted(preds["membership"].unique()))
    q = f3.text_input("Search name / customer id")
    view = preds[preds["risk_band"].isin(bands)] if bands else preds
    if mems:
        view = view[view["membership"].isin(mems)]
    if q:
        view = view[view["name"].str.contains(q, case=False, na=False)
                    | view["customer_id"].str.contains(q, case=False, na=False)]
    st.write(f"**{len(view):,}** customers match")
    st.dataframe(view.sort_values(["expected_score", "confidence"], ascending=False).head(1000),
                 hide_index=True,
                 column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1)})
    st.download_button("Download all predictions (CSV)", preds.to_csv(index=False),
                       "test_predictions.csv", "text/csv")

# ------------------------------------------------------------------ Predict a customer
with tabs[6]:
    st.subheader("Ek customer ka risk check karein")
    cat, num, mode = schema["cat"], schema["num"], schema["mode"]

    def sel(label, col, container, default=None):
        opts = cat[col]
        d = default or mode[col]
        return container.selectbox(label, opts, index=opts.index(d) if d in opts else 0)

    c1, c2, c3 = st.columns(3)
    membership = sel("Membership", "membership_category", c1)
    feedback = sel("Feedback", "feedback", c1)
    gender = sel("Gender", "gender", c1)
    region = sel("Region", "region_category", c1)
    age = c2.slider("Age", 10, 64, int(num["age"]["median"]))
    points = c2.number_input("Points in wallet", 0.0, 3000.0, float(round(num["points_in_wallet"]["median"], 1)))
    txn = c2.number_input("Avg transaction value", 0.0, 100000.0, float(round(num["avg_transaction_value"]["median"], 0)))
    time_spent = c2.number_input("Avg time spent (min)", 0.0, 4000.0, float(round(num["avg_time_spent"]["median"], 0)))
    days_login = c3.slider("Days since last login", 0, 26, int(num["days_since_last_login"]["median"]))
    freq = c3.slider("Avg login frequency (days)", 0, 75, int(num["avg_frequency_login_days"]["median"]))
    joined = c3.date_input("Joining date", date(2016, 6, 15), min_value=date(2015, 1, 1), max_value=date(2017, 12, 31))
    hour = c3.slider("Last visit hour", 0, 23, 12)

    d1, d2, d3, d4 = st.columns(4)
    referral = sel("Joined through referral", "joined_through_referral", d1)
    offers = sel("Preferred offers", "preferred_offer_types", d1)
    medium = sel("Medium of operation", "medium_of_operation", d2)
    internet = sel("Internet option", "internet_option", d2)
    discount = sel("Used special discount", "used_special_discount", d3)
    offer_pref = sel("Offer application preference", "offer_application_preference", d3)
    complaint = d4.selectbox("Past complaint", ["No", "Yes"])
    if complaint == "No":
        cstatus = "Not Applicable"
        d4.caption("Complaint status: Not Applicable")
    else:
        cstatus = d4.selectbox("Complaint status", [s for s in cat["complaint_status"] if s != "Not Applicable"])

    if st.button("Predict risk", type="primary"):
        jd = pd.Timestamp(joined)
        row = pd.DataFrame([{
            "gender": gender, "region_category": region, "membership_category": membership,
            "joined_through_referral": referral, "preferred_offer_types": offers,
            "medium_of_operation": medium, "internet_option": internet,
            "used_special_discount": discount, "offer_application_preference": offer_pref,
            "past_complaint": complaint, "complaint_status": cstatus, "feedback": feedback,
            "age": age, "days_since_last_login": days_login, "avg_time_spent": time_spent,
            "avg_transaction_value": txn, "avg_frequency_login_days": freq, "points_in_wallet": points,
            "tenure_days": (REF_DATE - jd).days, "join_year": jd.year, "join_month": jd.month,
            "visit_hour": hour, "has_referral_id": int(referral == "Yes"), "missing_login_gap": 0,
            "missing_time_spent": 0, "missing_freq": 0, "missing_points": 0}])[MODEL_COLS]
        p = model.predict_proba(row)[0]
        score = int(p.argmax()) + 1
        band = RISK_BAND[score]
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted risk score", score)
        m2.metric("Risk band", band)
        m3.metric("Confidence", f"{p.max():.0%}")
        st.bar_chart(pd.Series(p, index=[f"Score {i}" for i in range(1, 6)]))

        tips = []
        if feedback in ("Poor Customer Service", "Poor Website", "Poor Product Quality", "Too many ads",
                        "No reason specified"):
            tips.append(f"Feedback '{feedback}': is masle par seedha rabta karke solution dein.")
        if membership in ("No Membership", "Basic Membership"):
            tips.append("Membership upgrade par trial offer ya discount dein.")
        if points < num["points_in_wallet"]["median"] * 0.7:
            tips.append("Wallet points kam hain: bonus points ya reward campaign chalayein.")
        if complaint == "Yes" and cstatus in ("Unsolved", "No Information Available"):
            tips.append("Complaint unsolved hai: priority par resolve karke follow-up karein.")
        if band == "High" and tips:
            st.subheader("Retention suggestions")
            for t in tips:
                st.write("• " + t)
        elif band == "Low":
            st.success("Customer stable hai, special action ki zaroorat nahi.")
