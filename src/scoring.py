"""
Phase 4: Combined scoring logic.
"""
import pandas as pd
import numpy as np
import json
import os

CONFIG_PATH = "models/config.json"

def _ensure_scoring_config():
    if not os.path.exists(CONFIG_PATH):
        config = {}
    else:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            
    if 'weight_iqr' not in config:
        config['weight_iqr'] = 0.15
        config['weight_ts'] = 0.15
        config['t_review'] = 0.5
        config['t_high'] = 0.8
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
            
    return config

def compute_risk(df: pd.DataFrame, custom_config: dict = None) -> pd.DataFrame:
    """
    Computes final risk category and score based on Isolation Forest as primary, 
    with IQR and Time-Series as boosters.
    """
    df = df.copy()
    config = custom_config if custom_config else _ensure_scoring_config()
    
    w_iqr = config.get('weight_iqr', 0.15)
    w_ts = config.get('weight_ts', 0.15)
    t_review = config.get('t_review', 0.5)
    t_high = config.get('t_high', 0.8)
    
    # Normalize iso_score using min-max scaling
    # If the user is running the batch pipeline, we compute and save min/max
    # If not (e.g. single transaction in API), we load from config
    min_iso = df['iso_score'].min()
    max_iso = df['iso_score'].max()
    
    if 'iso_min' not in config or 'iso_max' not in config:
        config['iso_min'] = float(min_iso)
        config['iso_max'] = float(max_iso)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        # Actually, in training we want to update it. We can just always update it if df is large.
        if len(df) > 1:
            config['iso_min'] = float(min_iso)
            config['iso_max'] = float(max_iso)
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=4)
        else:
            min_iso = config.get('iso_min', min_iso)
            max_iso = config.get('iso_max', max_iso)
            
    if max_iso > min_iso:
        norm_iso = (df['iso_score'] - min_iso) / (max_iso - min_iso)
    else:
        norm_iso = 0.0
        
    df['norm_iso'] = norm_iso
    
    # Compute risk score
    df['risk_score'] = df['norm_iso'] + w_iqr * df['iqr_flag'] + w_ts * df['ts_flag']
    df['risk_score'] = df['risk_score'].clip(lower=0, upper=1)
    
    # Calculate risk category
    conditions = [
        (df['risk_score'] >= t_high),
        (df['risk_score'] >= t_review),
        (df['risk_score'] < t_review)
    ]
    choices = ['HIGH', 'REVIEW', 'LOW']
    df['risk'] = np.select(conditions, choices, default='LOW')
    
    return df
