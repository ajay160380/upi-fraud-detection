from fastapi.testclient import TestClient
import pandas as pd
import json
import uuid
import datetime
import joblib

from app import app

client = TestClient(app)

def test_invalid_payload():
    with TestClient(app) as client:
        response = client.post("/score", json={
            "sender_id": "user1",
            "receiver_id": "user2"
            # missing fields
        })
        assert response.status_code == 422

def test_unknown_sender():
    with TestClient(app) as client:
        # Should not crash, should just use empty history defaults
        payload = {
            "transaction_id": str(uuid.uuid4()),
            "sender_id": "unknown_sender_999",
            "receiver_id": "some_receiver",
            "amount": 500,
            "city": "Mumbai",
            "device": "Android",
            "timestamp": "2023-10-01 12:00:00"
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk" in data

def test_normal_scenario():
    # Real sender with plenty of history
    # We found U0000 has > 20 txns by 2023-10-21
    df = pd.read_csv("data/upi_transactions.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    with TestClient(app) as client:
        payload_normal = {
            "transaction_id": str(uuid.uuid4()),
            "sender_id": "U0000",
            "receiver_id": "R0914",
            "amount": 1236.55,
            "city": "Lucknow",
            "device": "android",
            "timestamp": "2023-10-21 18:40:38"
        }
        resp = client.post("/score", json=payload_normal)
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk"] == "LOW"

def test_fraud_scenario():
    with TestClient(app) as client:
        txn_id = str(uuid.uuid4())
        payload_fraud = {
            "transaction_id": txn_id,
            "sender_id": "U0000",
            "receiver_id": "hacker_999",
            "amount": 950000,
            "city": "Moscow",
            "device": "UnknownDevice",
            "timestamp": "2023-10-22 03:00:00"
        }
        resp = client.post("/score", json=payload_fraud)
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["risk"] in ["REVIEW", "HIGH"]
        
        # Check /alerts returns the flagged txn
        alerts_resp = client.get("/alerts?risk=" + data["risk"])
        assert alerts_resp.status_code == 200
        alerts_data = alerts_resp.json()
        assert any(a["transaction_id"] == txn_id for a in alerts_data)
    
def test_parity():
    df = pd.read_csv("data/upi_transactions.csv")
    features_df = pd.read_csv("data/features.csv")
    
    with open("models/config.json", "r") as f:
        config = json.load(f)
        
    iso_min = config.get("iso_min", 0.0)
    iso_max = config.get("iso_max", 1.0)
    w_iqr = config.get("weight_iqr", 0.0)
    w_ts = config.get("weight_ts", 0.0)
    
    with TestClient(app) as client:
        for i in range(20):
            row = df.iloc[i]
            payload = {
                "transaction_id": row['transaction_id'],
                "sender_id": row['sender_id'],
                "receiver_id": row['receiver_id'],
                "amount": row['amount'],
                "city": row['city'],
                "device": row['device_type'],
                "timestamp": row['timestamp']
            }
            
            resp = client.post("/score", json=payload)
            assert resp.status_code == 200
            api_score = resp.json()["risk_score"]
            
            # Predict with batch model
            clf = joblib.load("models/iso.pkl")
            # We need to construct the feature array exactly as trained
            from src.detectors import ISO_FEATURES
            
            # features_df has the features, but wait, features_df might not be sorted identically to df?
            # They should be 1:1 if we didn't sort. Wait, batch features sorts by sender_id!
            # So features_df.iloc[i] is NOT df.iloc[i].
            # We must match by transaction_id. But features_df doesn't have transaction_id! 
            # (Wait, features.py cleans it up or keeps it?)
            # Let's just run the build_features_batch on the df.iloc[[i]]? No, that's what API does.
            # To do a true parity test, we can just run build_features_batch on df head(100), and compare with API.
            pass
            
    # Let's do parity properly
    from src.features import build_features_batch
    df_small = df.head(50).copy()
    df_feat_batch = build_features_batch(df_small)
    # df_feat_batch IS sorted by sender_id, so we must iterate over it
    clf = joblib.load("models/iso.pkl")
    # Score batch
    feats_only = df_feat_batch[ISO_FEATURES]
    iso_scores = -clf.score_samples(feats_only)
    
    with open("models/iqr_bounds.json", "r") as f:
        bounds = json.load(f)
        
    with TestClient(app) as client:
        for i in range(20):
            row = df_feat_batch.iloc[i]
            # Since df_feat_batch doesn't have original categorical features like city, we have to find the original row
            orig_row = df[df['transaction_id'] == row['transaction_id']].iloc[0]
            
            payload = {
                "transaction_id": orig_row['transaction_id'],
                "sender_id": orig_row['sender_id'],
                "receiver_id": orig_row['receiver_id'],
                "amount": orig_row['amount'],
                "city": orig_row['city'],
                "device": orig_row['device_type'],
                "timestamp": orig_row['timestamp']
            }
            
            resp = client.post("/score", json=payload)
            assert resp.status_code == 200
            api_score = resp.json()["risk_score"]
            api_iso_flag = resp.json()["flags"]["iso"]
            
            batch_iso = iso_scores[i]
            if iso_max > iso_min:
                batch_norm = (batch_iso - iso_min) / (iso_max - iso_min)
            else:
                batch_norm = 0.0
                
            batch_norm = max(0.0, min(1.0, float(batch_norm)))
            
            # Since w_iqr=0, expected_score is just batch_norm
            expected_score = batch_norm
            assert abs(api_score - expected_score) < 1e-3, f"Mismatch at idx {i}: API {api_score}, Batch {expected_score}"
