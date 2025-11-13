# src/inference_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from .utils import ALERT_THRESHOLD, save_alert

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'fastag_fraud_model.pkl'

app = FastAPI()
artifact = joblib.load(MODEL_PATH)
model = artifact['model']
FEATURES = artifact['features']

class Transaction(BaseModel):
    transaction_id: str
    tag_id: str
    timestamp: str
    amount: float
    time_since_last_tx: int
    tx_count_1h: int
    unique_plazas_7d: int
    mismatched_ocr: int
    velocity_kmph: int

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/predict')
def predict(tx: Transaction):
    txd = tx.dict()
    df = pd.DataFrame([ {k: txd[k] for k in FEATURES} ])
    score = float(model.predict_proba(df)[:,1][0])
    result = {'fraud_score': score, 'threshold': ALERT_THRESHOLD}
    if score >= ALERT_THRESHOLD:
        # persist alert for UI
        save_alert({**txd, 'fraud_score': score})
    return result

# To run:
# uvicorn src.inference_api:app --reload --port 8000
