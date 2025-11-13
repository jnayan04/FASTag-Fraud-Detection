# src/utils.py
from pathlib import Path
import sqlite3
import json

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'alerts.db'
ALERT_THRESHOLD = 0.7

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT,
        tag_id TEXT,
        payload TEXT,
        fraud_score REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit(); conn.close()

def save_alert(tx_payload: dict):
    init_db()
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('INSERT INTO alerts(transaction_id, tag_id, payload, fraud_score) VALUES (?,?,?,?)',
              (tx_payload.get('transaction_id'), tx_payload.get('tag_id'), json.dumps(tx_payload), tx_payload.get('fraud_score')))
    conn.commit(); conn.close()

def fetch_alerts(limit=100):
    init_db()
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    rows = c.execute('SELECT id, transaction_id, tag_id, payload, fraud_score, created_at FROM alerts ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return rows
