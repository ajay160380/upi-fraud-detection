from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sqlite3
import pandas as pd
import numpy as np
import json
import joblib
import os
from contextlib import asynccontextmanager

from src.features import build_features_single
from src.detectors import ISO_FEATURES

app = FastAPI(title="UPI Fraud Detection API")

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    models['clf'] = joblib.load("models/iso.pkl")
    with open("models/config.json", "r") as f:
        models['config'] = json.load(f)
    with open("models/iqr_bounds.json", "r") as f:
        models['bounds'] = json.load(f)
        
    init_db()
    yield
    # Shutdown

app.router.lifespan_context = lifespan

class Transaction(BaseModel):
    transaction_id: str
    sender_id: str
    receiver_id: str
    amount: float = Field(..., gt=0)
    city: str
    device: str
    timestamp: str

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/app.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            sender_id TEXT,
            receiver_id TEXT,
            amount REAL,
            city TEXT,
            device_type TEXT,
            timestamp TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            transaction_id TEXT PRIMARY KEY,
            risk TEXT,
            risk_score REAL,
            reasons TEXT
        )
    ''')
    conn.commit()
    
    # Check if empty
    c.execute("SELECT COUNT(*) FROM transactions")
    if c.fetchone()[0] == 0:
        print("Seeding database...")
        df = pd.read_csv("data/upi_transactions.csv")
        # Keep only required columns
        df_seed = df[['transaction_id', 'sender_id', 'receiver_id', 'amount', 'city', 'device_type', 'timestamp']]
        df_seed.to_sql('transactions', conn, if_exists='append', index=False)
    conn.close()

def get_history(sender_id: str, ts: str) -> pd.DataFrame:
    conn = sqlite3.connect("data/app.db")
    query = "SELECT * FROM transactions WHERE sender_id = ? AND timestamp < ? ORDER BY timestamp"
    df = pd.read_sql_query(query, conn, params=(sender_id, ts))
    conn.close()
    return df

def save_transaction(txn: dict):
    conn = sqlite3.connect("data/app.db")
    c = conn.cursor()
    # Check if exists to avoid UNIQUE constraint failed
    c.execute("SELECT 1 FROM transactions WHERE transaction_id = ?", (txn['transaction_id'],))
    if not c.fetchone():
        c.execute('''
            INSERT INTO transactions (transaction_id, sender_id, receiver_id, amount, city, device_type, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (txn['transaction_id'], txn['sender_id'], txn['receiver_id'], txn['amount'], txn['city'], txn['device_type'], txn['timestamp']))
    conn.commit()
    conn.close()

def save_alert(txn_id: str, risk: str, score: float, reasons: list):
    conn = sqlite3.connect("data/app.db")
    c = conn.cursor()
    c.execute("SELECT 1 FROM alerts WHERE transaction_id = ?", (txn_id,))
    if not c.fetchone():
        c.execute('''
            INSERT INTO alerts (transaction_id, risk, risk_score, reasons)
            VALUES (?, ?, ?, ?)
        ''', (txn_id, risk, score, json.dumps(reasons)))
    conn.commit()
    conn.close()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/alerts")
def get_alerts(limit: int = 50, risk: str = "HIGH"):
    conn = sqlite3.connect("data/app.db")
    # Join with transactions to get full details
    query = """
        SELECT a.transaction_id, a.risk, a.risk_score, a.reasons, t.sender_id, t.amount, t.timestamp
        FROM alerts a
        JOIN transactions t ON a.transaction_id = t.transaction_id
        WHERE a.risk = ?
        ORDER BY t.timestamp DESC LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(risk, limit))
    conn.close()
    
    # Parse json reasons
    if not df.empty:
        df['reasons'] = df['reasons'].apply(json.loads)
    
    return df.to_dict(orient="records")

@app.post("/score")
def score_transaction(txn: Transaction):
    txn_dict = txn.model_dump()
    # Map device -> device_type for history matching
    txn_mapped = txn_dict.copy()
    txn_mapped['device_type'] = txn_dict['device']
    
    # 1. Fetch History strictly before this transaction
    history_df = get_history(txn.sender_id, txn.timestamp)
    
    # 2. Build Features
    feats = build_features_single(txn_mapped, history_df)
    
    # 3. Predict Isolation Forest
    clf = models['clf']
    # Create single-row df
    feat_df = pd.DataFrame([feats])[ISO_FEATURES]
    iso_score_raw = -clf.score_samples(feat_df)[0]
    
    # 4. Normalize score and apply tiers
    config = models['config']
    iso_min = config.get("iso_min", 0.0)
    iso_max = config.get("iso_max", 1.0)
    
    if iso_max > iso_min:
        norm_iso = (iso_score_raw - iso_min) / (iso_max - iso_min)
    else:
        norm_iso = 0.0
    
    norm_iso = max(0.0, min(1.0, float(norm_iso)))
    
    # We use iso_score as risk_score since boosters are 0 in config, but let's implement boosters anyway
    w_iqr = config.get('weight_iqr', 0.0)
    w_ts = config.get('weight_ts', 0.0)
    
    bounds = models['bounds']
    iqr_f = 0
    ts_burst_f = 0
    reasons = []
    
    # Check IQR
    amt_ratio = feats['amt_ratio']
    if 'amt_ratio' in bounds:
        if amt_ratio > bounds['amt_ratio']['upper']:
            iqr_f = 1
            reasons.append(f"Amount is {amt_ratio:.1f}x the user's historical average.")
            
    txn_count_1h = feats['txn_count_1h']
    if 'txn_count_1h' in bounds:
        if txn_count_1h > bounds['txn_count_1h']['upper']:
            iqr_f = 1
            ts_burst_f = 1
            reasons.append(f"High velocity: {int(txn_count_1h)} transactions in the last hour.")
            
    if feats['is_night'] == 1:
        reasons.append("Transaction occurred during unusual late-night hours (12 AM - 5 AM).")
        
    if feats['city_changed'] == 1 and feats['device_changed'] == 1:
        reasons.append("Simultaneous new city and new device detected.")
        
    if feats['is_new_receiver'] == 1:
        reasons.append("First time payment to this receiver.")
        
    iso_f = 1 if norm_iso > 0.6 else 0
    
    risk_score = norm_iso + w_iqr * iqr_f + w_ts * ts_burst_f
    risk_score = max(0.0, min(1.0, risk_score))
    
    t_review = config.get('t_review', 0.5)
    t_high = config.get('t_high', 0.8)
    
    if risk_score >= t_high:
        risk = "HIGH"
    elif risk_score >= t_review:
        risk = "REVIEW"
    else:
        risk = "LOW"
        
    # Save to db
    save_transaction(txn_mapped)
    if risk in ["REVIEW", "HIGH"]:
        save_alert(txn.transaction_id, risk, risk_score, reasons)
        
    return {
        "risk": risk,
        "risk_score": risk_score,
        "flags": {
            "iqr": iqr_f,
            "iso": iso_f,
            "ts_burst": ts_burst_f
        },
        "reasons": reasons
    }
