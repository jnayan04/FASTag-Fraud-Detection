# src/client_send_tx.py
import requests

payload = {
    'transaction_id':'TX999999',
    'tag_id':'TAG00123',
    'timestamp':'2025-01-01T12:00:00',
    'amount':50.0,
    'time_since_last_tx':5,
    'tx_count_1h':8,
    'unique_plazas_7d':4,
    'mismatched_ocr':1,
    'velocity_kmph':120
}

res = requests.post('http://localhost:8000/predict', json=payload)
print(res.json())
