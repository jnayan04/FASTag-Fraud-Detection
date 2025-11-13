# src/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import requests
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from pathlib import Path
from src.utils import fetch_alerts, ALERT_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'fastag_fraud_model.pkl'
artifact = joblib.load(MODEL_PATH)
FEATURES = artifact['features']

st.set_page_config(page_title='FASTag Fraud Dashboard', layout='wide')
st.title('FASTag Fraud Detection')

tabs = st.tabs(['Real-time Scoring','Bulk Upload & Score','Alerts / Cases'])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.header('Real-time scoring')
    cols = st.columns(2)
    with cols[0]:
        transaction_id = st.text_input('Transaction ID', 'TX000001')
        tag_id = st.text_input('Tag ID','TAG00001')
        timestamp = st.text_input('Timestamp','2025-01-01T12:00:00')
        amount = st.number_input('Amount', value=100.0)
        time_since_last_tx = st.number_input('time_since_last_tx (s)', value=120)
    with cols[1]:
        tx_count_1h = st.number_input('tx_count_1h', value=1)
        unique_plazas_7d = st.number_input('unique_plazas_7d', value=1)
        mismatched_ocr = st.selectbox('mismatched_ocr', [0,1])
        velocity_kmph = st.number_input('velocity_kmph', value=40)

    if st.button('Score transaction'):
        payload = {
            'transaction_id': transaction_id,
            'tag_id': tag_id,
            'timestamp': timestamp,
            'amount': float(amount),
            'time_since_last_tx': int(time_since_last_tx),
            'tx_count_1h': int(tx_count_1h),
            'unique_plazas_7d': int(unique_plazas_7d),
            'mismatched_ocr': int(mismatched_ocr),
            'velocity_kmph': int(velocity_kmph)
        }
        try:
            res = requests.post('http://localhost:8000/predict', json=payload, timeout=5).json()
            score = res['fraud_score']
            st.metric('Fraud score', f'{score:.3f}')
            if score >= ALERT_THRESHOLD:
                st.warning('High fraud risk — added to alerts')
        except Exception as e:
            st.error('Error calling API: ' + str(e))

# ---------------------------------------------------------------------------
with tabs[1]:
    st.header('Bulk CSV upload & scoring')
    uploaded = st.file_uploader('Upload a CSV of transactions', type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        missing = set(FEATURES) - set(df.columns)
        if missing:
            st.error('CSV missing columns: ' + ', '.join(missing))
        else:
            model = artifact['model']
            df['fraud_score'] = model.predict_proba(df[FEATURES])[:,1]
            st.write('Top flagged transactions')
            flagged = df[df['fraud_score'] >= ALERT_THRESHOLD].sort_values('fraud_score', ascending=False)
            st.dataframe(flagged.head(100))
            st.download_button('Download flagged CSV', flagged.to_csv(index=False), file_name='flagged.csv')

# ---------------------------------------------------------------------------
with tabs[2]:
    st.header('Alerts / Cases')
    st.info('Alerts come from the realtime API when score >= threshold')
    rows = fetch_alerts(200)
    if not rows:
        st.write('No alerts yet')
    else:
        df_rows = []
        for r in rows:
            payload = json.loads(r[3])
            df_rows.append({
                'id': r[0], 'transaction_id': r[1], 'tag_id': r[2], 'fraud_score': r[4], 'payload': payload, 'created_at': r[5]
            })
        adf = pd.DataFrame(df_rows)
        st.dataframe(adf.sort_values('fraud_score', ascending=False))
