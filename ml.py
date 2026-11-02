"""VitalPredict ML pipeline.
1 Data collection -> 2 median imputation + scaling -> 3 RFE -> 4 Random Forest (primary) + Logistic Regression (baseline)
"""
import json, os
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE, "data", "diabetes_data.csv")
MODEL_PATH = os.path.join(BASE, "data", "models.joblib")
METRICS_PATH = os.path.join(BASE, "data", "metrics.json")

FEATURES = ["age", "bmi", "glucose", "bp", "activity", "food_intake", "stress", "family_history"]
LABELS = {"age": "Age", "bmi": "BMI", "glucose": "Blood Glucose", "bp": "Systolic BP",
          "activity": "Physical Activity", "food_intake": "Food Intake (kcal/day)",
          "stress": "Stress Level", "family_history": "Family History"}
N_SELECT = 6  # RFE keeps the 6 most relevant features


def generate_dataset(n=3000, seed=42):
    """Synthetic data used when no real CSV exists. Replace data/diabetes_data.csv with a real dataset
    (same columns + 'outcome') to train on real data."""
    r = np.random.default_rng(seed)
    age = r.integers(18, 85, n)
    bmi = np.clip(r.normal(28, 5.5, n) + (age - 45) * 0.03, 15, 55)
    glucose = np.clip(r.normal(105, 25, n) + (bmi - 28) * 1.2 + (age - 45) * 0.3, 55, 330)
    bp = np.clip(r.normal(122, 15, n) + (age - 45) * 0.35 + (bmi - 28) * 0.5, 80, 220)
    activity = r.choice([0, 1, 2], n, p=[.35, .45, .20])          # 0 low, 1 moderate, 2 high
    food = np.clip(r.normal(2300, 450, n), 1200, 4200)
    stress = r.integers(1, 11, n)                                  # 1-10
    fam = r.binomial(1, .3, n)
    z = (-9.2 + 0.045 * (glucose - 100) * 1.0 + 0.11 * (bmi - 25) + 0.03 * (age - 40) + 0.012 * (bp - 120)
         - 0.45 * activity + 0.0004 * (food - 2200) + 0.09 * stress + 0.9 * fam + 6.0)
    z = (z - z.mean()) * 1.0
    y = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame(dict(age=age, bmi=bmi, glucose=glucose, bp=bp, activity=activity,
                           food_intake=food, stress=stress, family_history=fam, outcome=y))
    for c in ["bmi", "glucose", "bp", "food_intake"]:               # simulate missing readings
        df.loc[r.random(n) < 0.06, c] = np.nan
    return df.round(1)


def load_data():
    if not os.path.exists(DATA_CSV):
        generate_dataset().to_csv(DATA_CSV, index=False)
    df = pd.read_csv(DATA_CSV)
    # clinically impossible zeros -> missing (handled by median imputation)
    for c in ["bmi", "glucose", "bp"]:
        df.loc[df[c] <= 0, c] = np.nan
    return df


def build(clf):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("rfe", RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=N_SELECT)),
        ("clf", clf),
    ])


def evaluate(pipe, X, y):
    p = pipe.predict_proba(X)[:, 1]
    yh = (p >= .5).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh), recall=recall_score(y, yh),
                f1=f1_score(y, yh), roc_auc=roc_auc_score(y, p)), p


def train():
    df = load_data()
    X, y = df[FEATURES], df["outcome"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    rf = build(RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)).fit(Xtr, ytr)
    lr = build(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    rf_m, rf_p = evaluate(rf, Xte, yte)
    lr_m, lr_p = evaluate(lr, Xte, yte)
    sel = [f for f, k in zip(FEATURES, rf.named_steps["rfe"].support_) if k]
    imp = dict(zip(sel, rf.named_steps["clf"].feature_importances_.round(4).tolist()))
    meta = dict(
        n_samples=len(df), n_test=len(yte), missing_before={c: int(v) for c, v in df[FEATURES].isna().sum().items()},
        medians={c: float(v) for c, v in zip(FEATURES, rf.named_steps["impute"].statistics_)},
        selected=sel, rfe_ranking=dict(zip(FEATURES, rf.named_steps["rfe"].ranking_.tolist())),
        importance=imp, labels=LABELS, random_forest=rf_m, logistic_regression=lr_m,
        test_labels=yte.tolist(), test_proba_rf=rf_p.round(4).tolist(), test_proba_lr=lr_p.round(4).tolist())
    joblib.dump(dict(rf=rf, lr=lr), MODEL_PATH)
    json.dump(meta, open(METRICS_PATH, "w"))
    return meta


_models = None
def models():
    global _models
    if _models is None:
        if not os.path.exists(MODEL_PATH):
            train()
        _models = joblib.load(MODEL_PATH)
    return _models


def metrics():
    models()
    return json.load(open(METRICS_PATH))


def predict(row):
    """row: dict of FEATURES. Returns probability, category, drivers (from LR contributions on selected features)."""
    m = models()
    X = pd.DataFrame([{f: row.get(f) for f in FEATURES}])
    prob = float(m["rf"].predict_proba(X)[0, 1])
    prob_lr = float(m["lr"].predict_proba(X)[0, 1])
    cat = "High Risk" if prob >= .65 else "Moderate Risk" if prob >= .35 else "Low Risk"
    lr = m["lr"]
    Zs = lr[:-1].transform(X)[0]
    sel = [f for f, k in zip(FEATURES, lr.named_steps["rfe"].support_) if k]
    contrib = Zs * lr.named_steps["clf"].coef_[0]
    drivers = [dict(feature=LABELS[f], value=row.get(f)) for f, c in
               sorted(zip(sel, contrib), key=lambda t: -t[1]) if c > 0.15][:4]
    return dict(prob=round(prob * 100, 1), prob_baseline=round(prob_lr * 100, 1), risk=cat, drivers=drivers)

if __name__ == "__main__":
    m = train()
    print("Selected by RFE:", m["selected"])
    print("RF:", {k: round(v, 3) for k, v in m["random_forest"].items()})
    print("LR:", {k: round(v, 3) for k, v in m["logistic_regression"].items()})
