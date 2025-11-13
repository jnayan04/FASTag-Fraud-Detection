
# FASTag Fraud Detection System 🚗💳

A real-time and batch FASTag transaction fraud detection system built using:
- 🧠 Machine Learning (Random Forest)
- ⚙️ FastAPI (backend inference)
- 💻 Streamlit (frontend dashboard)
- 🗄️ SQLite (alerts storage)

## Features
- Real-time transaction scoring
- Bulk CSV upload & fraud scoring
- Fraud alerts management system

## How to Run Locally
```bash
pip install -r requirements.txt
uvicorn src.inference_api:app --reload --port 8000
streamlit run src/streamlit_app.py
=======

