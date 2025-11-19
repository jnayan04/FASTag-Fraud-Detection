import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------
# Load Model + Metadata
# ----------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fastag_fraud_model.pkl"
ALERTS_PATH = ROOT / "alerts.csv"

# Load saved model dictionary: {"model": model, "features": FEATURES}
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
FEATURES = artifact["features"]

# ----------------------------
# UI CONFIG
# ----------------------------

st.set_page_config(page_title="FASTag Fraud Detection", layout="wide")

st.title("FASTag Fraud Detection")

tabs = st.tabs(["Real-time Scoring", "Bulk Upload & Score", "Alerts / Cases"])

# =========================================================
# TAB 1 — REAL-TIME SCORING
# =========================================================
with tabs[0]:

    st.header("Real-time scoring")

    txn_id = st.text_input("Transaction ID")
    tag_id = st.text_input("Tag ID")

    # Feature inputs
    time_since_last_tx = st.number_input("Time since last tx", min_value=0.0)
    tx_count_1h = st.number_input("Tx count 1h", min_value=0)
    amount = st.number_input("Amount", min_value=0.0)
    unique_plazas_7d = st.number_input("Unique plazas (7d)", min_value=0)
    mismatched_ocr = st.number_input("OCR mismatch (0/1)", min_value=0, max_value=1)
    velocity_kmph = st.number_input("Velocity (kmph)", min_value=0.0)

    if st.button("Predict Fraud Score"):
        payload = {
            "time_since_last_tx": time_since_last_tx,
            "tx_count_1h": tx_count_1h,
            "amount": amount,
            "unique_plazas_7d": unique_plazas_7d,
            "mismatched_ocr": mismatched_ocr,
            "velocity_kmph": velocity_kmph,
        }

        df = pd.DataFrame([{k: payload[k] for k in FEATURES}])
        fraud_score = float(model.predict_proba(df)[0][1])

        st.metric("Fraud Probability", f"{fraud_score:.2f}")

        if fraud_score > 0.5:
            st.error("⚠️ HIGH RISK – Possible Fraud Detected!")
        else:
            st.success("✅ Low Fraud Probability")

        # Save alert if needed
        if fraud_score > 0.5:
            alert_row = pd.DataFrame([{
                "transaction_id": txn_id,
                "tag_id": tag_id,
                "fraud_score": fraud_score
            }])
            if ALERTS_PATH.exists():
                old = pd.read_csv(ALERTS_PATH)
                pd.concat([old, alert_row], ignore_index=True).to_csv(ALERTS_PATH, index=False)
            else:
                alert_row.to_csv(ALERTS_PATH, index=False)

# =========================================================
# TAB 2 — BULK CSV UPLOAD & SCORING
# =========================================================
with tabs[1]:

    st.header("Bulk CSV upload & scoring")

    uploaded = st.file_uploader("Upload a CSV of transactions", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # Check if required columns exist
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")

        else:
            # Score all transactions
            df["fraud_score"] = model.predict_proba(df[FEATURES])[:, 1]

            # Filter flagged transactions ONLY
            flagged = df[df["fraud_score"] >= ALERT_THRESHOLD].copy()

            st.success(f"Scoring complete! Flagged {len(flagged)} high-risk transactions.")

            if flagged.empty:
                st.info("No transactions exceeded the fraud threshold.")
            else:
                # Sort by risk descending
                flagged = flagged.sort_values("fraud_score", ascending=False)

                st.subheader("Flagged Fraudulent Transactions")
                st.dataframe(flagged)

                # Download only flagged rows
                csv = flagged.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download flagged CSV",
                    csv,
                    "flagged_transactions.csv"
                )


# =========================================================
# TAB 3 — VIEW ALERTS / CASES
# =========================================================
with tabs[2]:

    st.header("Alerts / Cases")

    if not ALERTS_PATH.exists():
        st.info("No alerts recorded yet.")
    else:
        df_alerts = pd.read_csv(ALERTS_PATH)
        st.dataframe(df_alerts)

        # Allow filtering
        tag = st.text_input("Filter by Tag ID")
        if tag:
            filtered = df_alerts[df_alerts["tag_id"] == tag]
            st.dataframe(filtered)

        st.info("These alerts were triggered when fraud probability exceeded 0.5.")
