# VitalPredict (Flask + SQLite + scikit-learn)
    pip install -r requirements.txt
    python app.py          # http://127.0.0.1:5000
Email OTP: set SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, optional SMTP_FROM.
Without SMTP the OTP is printed to the console and shown on the verify screen (dev mode only).
Retrain: delete data/models.joblib and restart, or `python ml.py`. Put a real CSV at data/diabetes_data.csv
(columns: age,bmi,glucose,bp,activity,food_intake,stress,family_history,outcome) to train on real data.
