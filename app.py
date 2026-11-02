import os, re, secrets, smtplib, sqlite3, hashlib, hmac
from datetime import datetime, timedelta
from email.message import EmailMessage
from functools import wraps
try:
    from flask import Flask, g, jsonify, request, session, send_from_directory  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: Flask. Install it with `pip install flask`.") from exc

import ml


def generate_password_hash(password, method="pbkdf2_sha256", salt_length=16):
    if method != "pbkdf2_sha256":
        raise ValueError(f"Unsupported password hashing method: {method}")
    salt = secrets.token_hex(salt_length // 2)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return f"{method}${salt}${digest.hex()}"


def check_password_hash(password_hash, password):
    try:
        method, salt, expected = password_hash.split("$", 2)
    except ValueError:
        return False
    if method != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000).hex()
    return hmac.compare_digest(digest, expected)
import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
from sklearn.metrics import accuracy_score  # type: ignore[import-not-found]

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "data", "vitalpredict.db")
OTP_TTL_MIN, OTP_MAX_TRIES, OTP_RESEND_SEC = 10, 5, 30

app = Flask(__name__, static_folder="static", static_url_path="")
os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
_kf = os.path.join(BASE, "data", ".secret")
if os.environ.get("SECRET_KEY"):
    app.secret_key = os.environ["SECRET_KEY"]
else:  # persist a generated key so sessions survive restarts
    if not os.path.exists(_kf): open(_kf, "w").write(secrets.token_hex(32))
    app.secret_key = open(_kf).read().strip()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

SMTP = dict(host=os.environ.get("SMTP_HOST"), port=int(os.environ.get("SMTP_PORT", 587)),
            user=os.environ.get("SMTP_USER"), password=os.environ.get("SMTP_PASSWORD"),
            sender=os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER"))
SMTP_READY = bool(SMTP["host"] and SMTP["user"] and SMTP["password"])

# ---------------- database ----------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN('doctor','patient')),
  verified INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS otps(
  email TEXT PRIMARY KEY, code_hash TEXT NOT NULL, expires_at TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0, sent_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS records(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  patient_name TEXT NOT NULL, age REAL, bmi REAL, glucose REAL, bp REAL, activity INTEGER,
  food_intake REAL, stress INTEGER, family_history INTEGER, risk TEXT NOT NULL, prob REAL NOT NULL,
  created_at TEXT NOT NULL);
"""

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH); g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db

@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d: d.close()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH); c.executescript(SCHEMA); c.commit(); c.close()

now = lambda: datetime.utcnow().isoformat(timespec="seconds")

# ---------------- OTP + email ----------------
def hash_otp(email, code):
    return hmac.new(app.secret_key.encode(), f"{email}:{code}".encode(), hashlib.sha256).hexdigest()

def send_email(to, subject, body):
    if not SMTP_READY:
        print(f"\n[DEV EMAIL - SMTP not configured] To: {to}\n{subject}\n{body}\n")
        return False
    msg = EmailMessage(); msg["From"], msg["To"], msg["Subject"] = SMTP["sender"], to, subject
    msg.set_content(body)
    with smtplib.SMTP(SMTP["host"], SMTP["port"], timeout=15) as s:
        s.starttls(); s.login(SMTP["user"], SMTP["password"]); s.send_message(msg)
    return True

def issue_otp(email, name):
    code = f"{secrets.randbelow(10**6):06d}"
    db().execute("INSERT OR REPLACE INTO otps(email,code_hash,expires_at,attempts,sent_at) VALUES(?,?,?,0,?)",
                 (email, hash_otp(email, code), (datetime.utcnow() + timedelta(minutes=OTP_TTL_MIN)).isoformat(timespec="seconds"), now()))
    db().commit()
    sent = send_email(email, "Your VitalPredict verification code",
                      f"Hi {name},\n\nYour verification code is {code}.\nIt expires in {OTP_TTL_MIN} minutes.\n\nIf you did not sign up, ignore this email.")
    return code, sent

def otp_response(code, sent, **extra):
    out = dict(ok=True, email_sent=sent, **extra)
    if not sent:  # local dev only: no SMTP configured, so surface the code to the UI
        out["dev_otp"] = code
    return out

# ---------------- helpers ----------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
def err(msg, code=400): return jsonify(ok=False, error=msg), code

def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if "uid" not in session: return err("Please sign in.", 401)
        u = db().execute("SELECT * FROM users WHERE id=? AND verified=1", (session["uid"],)).fetchone()
        if not u: session.clear(); return err("Please sign in.", 401)
        g.user = u
        return f(*a, **k)
    return w

def user_json(u): return dict(id=u["id"], name=u["name"], email=u["email"], role=u["role"])

RANGES = dict(age=(1, 110), bmi=(10, 60), glucose=(40, 400), bp=(60, 240), food_intake=(500, 6000),
              stress=(1, 10), activity=(0, 2), family_history=(0, 1))
def parse_inputs(d):
    row = {}
    for k, (lo, hi) in RANGES.items():
        v = d.get(k)
        if v in (None, ""):
            row[k] = None  # missing -> median imputation in pipeline
            continue
        try: v = float(v)
        except (TypeError, ValueError): raise ValueError(f"{k} must be a number")
        if not lo <= v <= hi: raise ValueError(f"{k} must be between {lo} and {hi}")
        row[k] = v
    if row["age"] is None or row["glucose"] is None: raise ValueError("Age and blood glucose are required")
    return row

# ---------------- auth routes ----------------
@app.post("/api/auth/register")
def register():
    d = request.get_json(force=True, silent=True) or {}
    name, email, pw, role = (d.get("name") or "").strip(), (d.get("email") or "").strip().lower(), d.get("password") or "", d.get("role")
    if len(name) < 2: return err("Please enter your full name.")
    if not EMAIL_RE.match(email): return err("Enter a valid email address.")
    if len(pw) < 8: return err("Password must be at least 8 characters.")
    if role not in ("doctor", "patient"): return err("Choose a role.")
    ex = db().execute("SELECT verified FROM users WHERE email=?", (email,)).fetchone()
    if ex and ex["verified"]: return err("An account with this email already exists.", 409)
    if ex: db().execute("UPDATE users SET name=?,password_hash=?,role=? WHERE email=?", (name, generate_password_hash(pw), role, email))
    else: db().execute("INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)", (name, email, generate_password_hash(pw), role, now()))
    db().commit()
    code, sent = issue_otp(email, name)
    return jsonify(otp_response(code, sent, email=email))

@app.post("/api/auth/resend-otp")
def resend():
    email = ((request.get_json(force=True, silent=True) or {}).get("email") or "").strip().lower()
    u = db().execute("SELECT * FROM users WHERE email=? AND verified=0", (email,)).fetchone()
    o = db().execute("SELECT sent_at FROM otps WHERE email=?", (email,)).fetchone()
    if not u: return jsonify(ok=True)  # don't reveal account existence
    if o and (datetime.utcnow() - datetime.fromisoformat(o["sent_at"])).seconds < OTP_RESEND_SEC and (datetime.utcnow() - datetime.fromisoformat(o["sent_at"])).days == 0:
        return err(f"Wait {OTP_RESEND_SEC} seconds before requesting another code.", 429)
    code, sent = issue_otp(email, u["name"])
    return jsonify(otp_response(code, sent))

@app.post("/api/auth/verify-otp")
def verify():
    d = request.get_json(force=True, silent=True) or {}
    email, code = (d.get("email") or "").strip().lower(), (d.get("otp") or "").strip()
    o = db().execute("SELECT * FROM otps WHERE email=?", (email,)).fetchone()
    if not o: return err("No code requested for this email.")
    if datetime.utcnow() > datetime.fromisoformat(o["expires_at"]): return err("Code expired. Request a new one.")
    if o["attempts"] >= OTP_MAX_TRIES: return err("Too many attempts. Request a new code.", 429)
    if not hmac.compare_digest(o["code_hash"], hash_otp(email, code)):
        db().execute("UPDATE otps SET attempts=attempts+1 WHERE email=?", (email,)); db().commit()
        return err("Incorrect code.")
    db().execute("UPDATE users SET verified=1 WHERE email=?", (email,))
    db().execute("DELETE FROM otps WHERE email=?", (email,)); db().commit()
    u = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    session.clear(); session["uid"] = u["id"]
    return jsonify(ok=True, user=user_json(u))

@app.post("/api/auth/login")
def login():
    d = request.get_json(force=True, silent=True) or {}
    email, pw = (d.get("email") or "").strip().lower(), d.get("password") or ""
    u = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not u or not check_password_hash(u["password_hash"], pw): return err("Invalid email or password.", 401)
    if not u["verified"]:
        code, sent = issue_otp(email, u["name"])
        return jsonify(otp_response(code, sent, need_verify=True, email=email)), 403
    session.clear(); session["uid"] = u["id"]
    return jsonify(ok=True, user=user_json(u))

@app.post("/api/auth/logout")
def logout(): session.clear(); return jsonify(ok=True)

@app.get("/api/auth/me")
@login_required
def me(): return jsonify(ok=True, user=user_json(g.user))

# ---------------- ML + records ----------------
@app.post("/api/predict")
@login_required
def predict():
    try: row = parse_inputs(request.get_json(force=True, silent=True) or {})
    except ValueError as e: return err(str(e))
    return jsonify(ok=True, **ml.predict(row))

def rec_json(r):
    return dict(id=f"REC-{r['id']}", rid=r["id"], date=r["created_at"].replace("T", " ")[:16], name=r["patient_name"],
                age=r["age"], bmi=r["bmi"], glucose=r["glucose"], bp=r["bp"], risk=r["risk"], prob=r["prob"])

@app.post("/api/records")
@login_required
def save_record():
    d = request.get_json(force=True, silent=True) or {}
    try: row = parse_inputs(d)
    except ValueError as e: return err(str(e))
    res = ml.predict(row)  # recomputed server-side so stored result can't be forged
    name = (d.get("patient_name") or "").strip() or g.user["name"]
    cur = db().execute("INSERT INTO records(user_id,patient_name,age,bmi,glucose,bp,activity,food_intake,stress,family_history,risk,prob,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (g.user["id"], name[:80], row["age"], row["bmi"], row["glucose"], row["bp"], row["activity"], row["food_intake"], row["stress"], row["family_history"], res["risk"], res["prob"], now()))
    db().commit()
    return jsonify(ok=True, record=rec_json(db().execute("SELECT * FROM records WHERE id=?", (cur.lastrowid,)).fetchone()))

@app.get("/api/records")
@login_required
def list_records():
    q = "SELECT * FROM records" + ("" if g.user["role"] == "doctor" else " WHERE user_id=?") + " ORDER BY id DESC LIMIT 500"
    rows = db().execute(q, () if g.user["role"] == "doctor" else (g.user["id"],)).fetchall()
    return jsonify(ok=True, records=[rec_json(r) for r in rows])

@app.delete("/api/records/<int:rid>")
@login_required
def delete_record(rid):
    r = db().execute("SELECT user_id FROM records WHERE id=?", (rid,)).fetchone()
    if not r: return err("Not found.", 404)
    if g.user["role"] != "doctor" and r["user_id"] != g.user["id"]: return err("Forbidden.", 403)
    db().execute("DELETE FROM records WHERE id=?", (rid,)); db().commit()
    return jsonify(ok=True)

@app.get("/api/model/metrics")
@login_required
def model_metrics():
    if g.user["role"] != "doctor": return err("Clinician access only.", 403)
    return jsonify(ok=True, **ml.metrics())

@app.get("/")
def index(): return send_from_directory("static", "index.html")

init_db()
ml.models()
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
