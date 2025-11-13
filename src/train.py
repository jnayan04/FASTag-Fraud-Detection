# src/train.py
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'synthetic_transactions.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)

FEATURES = ['time_since_last_tx','tx_count_1h','amount','unique_plazas_7d','mismatched_ocr','velocity_kmph']

def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:,1]
    y_pred = (proba >= 0.5).astype(int)

    print('ROC AUC:', roc_auc_score(y_test, proba))
    print('Precision/Recall/F1:', precision_recall_fscore_support(y_test, y_pred, average='binary'))

    joblib.dump({'model': model, 'features': FEATURES}, MODELS / 'fastag_fraud_model.pkl')
    print('Saved model to', MODELS / 'fastag_fraud_model.pkl')

if __name__ == '__main__':
    main()
