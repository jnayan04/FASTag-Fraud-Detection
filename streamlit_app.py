import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sqlite3
import json

# -----------------------------
# PATHS
# -----------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fastag_fraud_model.pkl"
DB_PATH = ROOT / "alerts.db"

# -----------------------------
# LOAD MODEL ONCE
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# FEATURES (fixed!)
# -----------------------------
FEATURES = [
    "amount",
    "time_since_last_tx",
    "tx_count_1h",
    "unique_plazas_7d",
    "mismatched_ocr",
    "velocity_kmph"
]

# -----------------------------
# DATABASE FUNCTIONS
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            tag_id TEXT,
            fraud_score REAL,
            payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_alert(transaction_id, tag_id, fraud_score, payload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO alerts (transaction_id, tag_id, fraud_score, payload)
        VALUES (?, ?, ?, ?)
    """, (transaction_id, tag_id, fraud_score, json.dumps(payload)))
    conn.commit()
    conn.close()

def fetch_alerts():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM alerts ORDER BY created_at DESC", conn)
    conn.close()
    return df

# Initialize DB at startup
init_db()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("FASTag Fraud Detection")

tabs = st.tabs(["Real-time Scoring", "Bulk Upload & Score", "Alerts / Cases"])


# ================================================================
# 1️⃣ REAL-TIME SCORING
# ================================================================
with tabs[0]:
    st.header("Real-time scoring")

    transaction_id = st.text_input("Transaction ID")
    tag_id = st.text_input("Tag ID")
    amount = st.number_input("Amount", min_value=0.0, value=100.0)
    time_since_last_tx = st.number_input("time_since_last_tx (s)", min_value=0, value=1000)
    tx_count_1h = st.number_input("tx_count_1h", min_value=0, value=1)
    unique_plazas_7d = st.number_input("unique_plazas_7d", min_value=0, value=1)
    mismatched_ocr = st.number_input("mismatched_ocr", min_value=0, max_value=1, value=0)
    velocity_kmph = st.number_input("velocity_kmph", min_value=0, value=60)

    if st.button("Score transaction"):
        payload = {
            "amount": amount,
            "time_since_last_tx": time_since_last_tx,
            "tx_count_1h": tx_count_1h,
            "unique_plazas_7d": unique_plazas_7d,
            "mismatched_ocr": mismatched_ocr,
            "velocity_kmph": velocity_kmph
        }

        df = pd.DataFrame([{k: payload[k] for k in FEATURES}])
        fraud_score = float(model.predict_proba(df)[:, 1][0])

        st.write(f"### Fraud Score: **{fraud_score:.4f}**")

        if fraud_score > 0.6:
            save_alert(transaction_id, tag_id, fraud_score, payload)
            st.error("⚠️ Fraud Alert Generated & Saved!")


# ================================================================
# 2️⃣ BULK CSV UPLOAD
# ================================================================
with tabs[1]:
    st.header("Bulk CSV Upload & Scoring")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # check if features exist
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df["fraud_score"] = model.predict_proba(df[FEATURES])[:, 1]

            st.write("### Scored Results")
            st.dataframe(df)

            # save all high-risk alerts
            for idx, row in df.iterrows():
                if row["fraud_score"] > 0.6:
                    save_alert(row.get("transaction_id", "N/A"),
                               row.get("tag_id", "N/A"),
                               float(row["fraud_score"]),
                               row.to_dict())


# ================================================================
# 3️⃣ ALERTS / CASES VIEW
# ================================================================
with tabs[2]:
    st.header("Fraud Alerts / Cases")
    df = fetch_alerts()
    st.dataframe(df)


