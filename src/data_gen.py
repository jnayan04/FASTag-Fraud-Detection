# src/data_gen.py
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'data'
OUT.mkdir(parents=True, exist_ok=True)

def make_synthetic(n=20000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        'transaction_id': [f'TX{i:06d}' for i in range(n)],
        'tag_id': [f'TAG{np.random.randint(1,5000):05d}' for _ in range(n)],
        'timestamp': pd.date_range('2025-01-01', periods=n, freq='T').astype(str),
        'amount': np.round(np.abs(np.random.normal(150, 80, n)),2).clip(5,10000),
        'time_since_last_tx': np.abs(np.random.exponential(300, n)).astype(int),
        'tx_count_1h': np.random.poisson(2, n),
        'unique_plazas_7d': np.random.poisson(1, n),
        'mismatched_ocr': np.random.binomial(1, 0.02, n),
        'velocity_kmph': np.abs(np.random.normal(50, 25, n)).astype(int)
    })

    # simple rule-based label for demo
    df['is_fraud'] = (
        (df['mismatched_ocr'] == 1) |
        ((df['time_since_last_tx'] < 10) & (df['tx_count_1h'] > 5)) |
        (df['amount'] > 500) |
        (df['velocity_kmph'] > 300)
    ).astype(int)

    return df

if __name__ == '__main__':
    df = make_synthetic(20000)
    out_file = OUT / 'synthetic_transactions.csv'
    df.to_csv(out_file, index=False)
    print('Wrote', out_file)
