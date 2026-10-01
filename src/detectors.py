"""
Phase 3: Detectors for UPI Fraud.
"""
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

CONFIG_PATH = "models/config.json"
IQR_BOUNDS_PATH = "models/iqr_bounds.json"
ISO_MODEL_PATH = "models/iso.pkl"

ISO_FEATURES = [
    'amt_ratio', 'amt_zscore', 'txn_count_1h', 'secs_since_last', 
    'recipient_repeat_count', 'is_new_receiver', 'device_changed', 
    'city_changed', 'is_night', 'same_receiver_count_24h', 
    'hour_sin', 'hour_cos'
]

def init_config():
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "iso_contamination": 0.05,
            "ts_zscore_threshold": 3.0,
            "user_burst_threshold": 3.0
        }
    
    config['iqr_multiplier'] = 3.0
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

def load_config():
    init_config()
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def train_iqr(df: pd.DataFrame):
    config = load_config()
    multiplier = config.get("iqr_multiplier", 3.0)
    bounds = {}
    for col in ['amt_ratio', 'txn_count_1h']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = {
            "lower": Q1 - multiplier * IQR,
            "upper": Q3 + multiplier * IQR
        }
    os.makedirs(os.path.dirname(IQR_BOUNDS_PATH), exist_ok=True)
    with open(IQR_BOUNDS_PATH, 'w') as f:
        json.dump(bounds, f, indent=4)
    return bounds

def iqr_flag(df: pd.DataFrame, bounds: dict) -> pd.Series:
    flags = np.zeros(len(df), dtype=int)
    for col in bounds:
        lower = bounds[col]['lower']
        upper = bounds[col]['upper']
        flags |= ((df[col] < lower) | (df[col] > upper)).astype(int)
    return pd.Series(flags, index=df.index)

def train_isolation_forest(df: pd.DataFrame):
    config = load_config()
    contamination = config.get("iso_contamination", 0.05)
    
    clf = IsolationForest(
        n_estimators=200, 
        contamination=contamination, 
        random_state=42
    )
    clf.fit(df[ISO_FEATURES])
    os.makedirs(os.path.dirname(ISO_MODEL_PATH), exist_ok=True)
    joblib.dump(clf, ISO_MODEL_PATH)

def apply_isolation_forest(df: pd.DataFrame):
    clf = joblib.load(ISO_MODEL_PATH)
    preds = clf.predict(df[ISO_FEATURES])
    iso_flag = (preds == -1).astype(int)
    # output iso_score (= -score_samples, higher = more anomalous)
    iso_score = -clf.score_samples(df[ISO_FEATURES])
    return pd.Series(iso_flag, index=df.index), pd.Series(iso_score, index=df.index)

def apply_time_series(df: pd.DataFrame) -> pd.Series:
    config = load_config()
    ts_z_thresh = config.get("ts_zscore_threshold", 3.0)
    user_burst_thresh = config.get("user_burst_threshold", 3.0)
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_floor'] = df['timestamp'].dt.floor('h')
    
    # 1. Global hourly anomalies
    hourly = df.groupby('hour_floor').agg(
        hourly_count=('transaction_id', 'count'),
        hourly_amount=('amount', 'sum')
    ).reset_index()
    
    # Fill missing hours
    if len(hourly) > 0:
        r = pd.date_range(start=hourly['hour_floor'].min(), end=hourly['hour_floor'].max(), freq='h')
        hourly = hourly.set_index('hour_floor').reindex(r).fillna(0).reset_index()
        hourly.rename(columns={'index': 'hour_floor'}, inplace=True)
    
    # 24h rolling
    hourly = hourly.set_index('hour_floor')
    rolling_mean_cnt = hourly['hourly_count'].rolling(24, min_periods=1).mean()
    rolling_std_cnt = hourly['hourly_count'].rolling(24, min_periods=1).std().fillna(0)
    
    rolling_mean_amt = hourly['hourly_amount'].rolling(24, min_periods=1).mean()
    rolling_std_amt = hourly['hourly_amount'].rolling(24, min_periods=1).std().fillna(0)
    
    hourly['z_cnt'] = (hourly['hourly_count'] - rolling_mean_cnt) / (rolling_std_cnt + 1e-5)
    hourly['z_amt'] = (hourly['hourly_amount'] - rolling_mean_amt) / (rolling_std_amt + 1e-5)
    
    hourly['global_ts_flag'] = ((hourly['z_cnt'].abs() > ts_z_thresh) | (hourly['z_amt'].abs() > ts_z_thresh)).astype(int)
    hourly = hourly.reset_index()
    
    df = df.merge(hourly[['hour_floor', 'global_ts_flag']], on='hour_floor', how='left')
    
    # 2. Per-user burst check (txn_count_1h z-score vs user baseline)
    # Use shift(1) to avoid data leakage from the current transaction
    grouped = df.groupby('sender_id')['txn_count_1h']
    user_mean_cnt = grouped.transform(lambda x: x.shift(1).expanding().mean())
    user_std_cnt = grouped.transform(lambda x: x.shift(1).expanding().std().fillna(0))
    
    # Fill first transaction with itself (z-score will be 0)
    user_mean_cnt = user_mean_cnt.fillna(df['txn_count_1h'])
    
    df['user_burst_z'] = (df['txn_count_1h'] - user_mean_cnt) / (user_std_cnt + 1e-5)
    df['user_ts_flag'] = (df['user_burst_z'] > user_burst_thresh).astype(int)
    
    # Combine flags
    ts_flag = (df['global_ts_flag'] | df['user_ts_flag']).fillna(0).astype(int)
    global_flag = df['global_ts_flag'].fillna(0).astype(int)
    user_flag = df['user_ts_flag'].fillna(0).astype(int)
    
    return ts_flag, user_flag, global_flag

if __name__ == "__main__":
    print("Loading features...")
    df = pd.read_csv("data/features.csv")
    
    print("Training detectors...")
    bounds = train_iqr(df)
    train_isolation_forest(df)
    
    print("Applying detectors...")
    df['iqr_flag'] = iqr_flag(df, bounds)
    df['iso_flag'], df['iso_score'] = apply_isolation_forest(df)
    df['ts_flag'], df['user_ts_flag'], df['global_ts_flag'] = apply_time_series(df)
    
    y_true = df['is_fraud']
    
    print("\n--- Detector Performance ---")
    for det in ['iqr_flag', 'iso_flag', 'ts_flag', 'user_ts_flag', 'global_ts_flag']:
        flags = df[det]
        num_flagged = flags.sum()
        precision = precision_score(y_true, flags, zero_division=0)
        recall = recall_score(y_true, flags, zero_division=0)
        print(f"{det}:")
        print(f"  Flagged:   {num_flagged}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}\n")
