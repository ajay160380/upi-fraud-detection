"""
Phase 2: Feature engineering.
Creates features from past transactions to avoid leakage.
"""
import pandas as pd
import numpy as np

def build_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by sender_id and timestamp
    df = df.sort_values(by=['sender_id', 'timestamp']).reset_index(drop=True)
    
    # Night
    df['is_night'] = df['timestamp'].dt.hour.apply(lambda x: 1 if 0 <= x <= 5 else 0)
    
    # Shifts
    df['prev_city'] = df.groupby('sender_id')['city'].shift(1)
    df['prev_device'] = df.groupby('sender_id')['device_type'].shift(1)
    df['prev_timestamp'] = df.groupby('sender_id')['timestamp'].shift(1)
    
    # Changes
    df['city_changed'] = ((df['city'] != df['prev_city']) & df['prev_city'].notnull()).astype(int)
    df['device_changed'] = ((df['device_type'] != df['prev_device']) & df['prev_device'].notnull()).astype(int)
    
    # Secs since last
    df['secs_since_last'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds().fillna(86400)
    
    # Recipient
    df['recipient_repeat_count'] = df.groupby(['sender_id', 'receiver_id']).cumcount()
    df['is_new_receiver'] = (df['recipient_repeat_count'] == 0).astype(int)
    
    # Amount stats
    grouped_amt = df.groupby('sender_id')['amount']
    df['past_mean'] = grouped_amt.transform(lambda x: x.shift(1).expanding().mean())
    df['past_std'] = grouped_amt.transform(lambda x: x.shift(1).expanding().std())
    
    df['amt_ratio'] = df['amount'] / df['past_mean']
    df['amt_ratio'] = df['amt_ratio'].fillna(1.0)
    
    df['amt_zscore'] = (df['amount'] - df['past_mean']) / (df['past_std'] + 1e-5)
    df['amt_zscore'] = df['amt_zscore'].fillna(0.0)
    
    # txn_count_1h and new features
    # df is already sorted by sender_id and timestamp
    senders = df['sender_id'].values
    timestamps = df['timestamp'].values
    receivers = df['receiver_id'].values
    
    txn_1h_list = np.zeros(len(df))
    same_rec_24h_list = np.zeros(len(df))
    unique_rec_1h_list = np.zeros(len(df))
    
    # We can keep track of start indices to optimize, but for 10k rows a simple loop with bounded lookback is fine
    for i in range(len(df)):
        ts = timestamps[i]
        rec = receivers[i]
        sender = senders[i]
        
        # Lookback bounds
        start_1h = ts - np.timedelta64(1, 'h')
        start_24h = ts - np.timedelta64(24, 'h')
        
        # Walk backwards to find the window
        j = i - 1
        txn_1h = 0
        same_rec_24h = 0
        unique_rec_1h_set = set()
        
        while j >= 0 and senders[j] == sender and timestamps[j] > start_24h:
            if timestamps[j] > start_1h:
                txn_1h += 1
                unique_rec_1h_set.add(receivers[j])
            
            if receivers[j] == rec:
                same_rec_24h += 1
                
            j -= 1
            
        txn_1h_list[i] = txn_1h
        same_rec_24h_list[i] = same_rec_24h
        unique_rec_1h_list[i] = len(unique_rec_1h_set)
        
    df['txn_count_1h'] = txn_1h_list
    df['same_receiver_count_24h'] = same_rec_24h_list
    df['receiver_unique_count_1h'] = unique_rec_1h_list
    
    df['new_receiver_amt_ratio'] = df['amt_ratio'] * df['is_new_receiver']
    
    # Cyclic hour features
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df = df.drop(columns=['hour'])
    
    # Clean up
    df = df.drop(columns=['prev_city', 'prev_device', 'prev_timestamp', 'past_mean', 'past_std'])
    
    # Re-sort to original order by timestamp if preferred, or leave sorted by sender_id
    df = df.sort_values(by=['sender_id', 'timestamp']).reset_index(drop=True)
    return df

def build_features_single(txn: dict, history_df: pd.DataFrame) -> dict:
    """
    Computes features for a single incoming transaction based on history.
    """
    ts = pd.to_datetime(txn['timestamp'])
    is_night = 1 if 0 <= ts.hour <= 5 else 0
    
    if history_df.empty:
        return {
            'amt_ratio': 1.0,
            'amt_zscore': 0.0,
            'txn_count_1h': 0.0,
            'secs_since_last': 86400.0,
            'recipient_repeat_count': 0,
            'is_new_receiver': 1,
            'device_changed': 0,
            'city_changed': 0,
            'is_night': is_night,
            'same_receiver_count_24h': 0.0,
            'receiver_unique_count_1h': 0.0,
            'new_receiver_amt_ratio': 1.0,
            'hour_sin': np.sin(2 * np.pi * ts.hour / 24),
            'hour_cos': np.cos(2 * np.pi * ts.hour / 24)
        }
        
    history_df = history_df.copy()
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    history_df = history_df.sort_values('timestamp')
    
    last_txn = history_df.iloc[-1]
    
    # Changes
    city_changed = 1 if txn['city'] != last_txn['city'] else 0
    device_changed = 1 if txn['device_type'] != last_txn['device_type'] else 0
    
    # Secs since last
    secs_since_last = (ts - last_txn['timestamp']).total_seconds()
    
    # Recipient
    recip_count = (history_df['receiver_id'] == txn['receiver_id']).sum()
    is_new_receiver = 1 if recip_count == 0 else 0
    
    # Amount stats
    past_mean = history_df['amount'].mean()
    # Ensure pandas standard deviation matches exactly with expanding().std() which has ddof=1
    past_std = history_df['amount'].std(ddof=1)
    
    amt_ratio = txn['amount'] / past_mean if past_mean > 0 else 1.0
    if pd.isna(past_std) or past_std == 0:
        amt_zscore = 0.0
    else:
        amt_zscore = (txn['amount'] - past_mean) / (past_std + 1e-5)
        
    # txn count 1h and new features
    one_hour_ago = ts - pd.Timedelta(hours=1)
    mask_1h = (history_df['timestamp'] > one_hour_ago)
    txn_count_1h = mask_1h.sum()
    receiver_unique_count_1h = history_df[mask_1h]['receiver_id'].nunique()
    
    one_day_ago = ts - pd.Timedelta(hours=24)
    same_receiver_count_24h = ((history_df['timestamp'] > one_day_ago) & (history_df['receiver_id'] == txn['receiver_id'])).sum()
    
    new_receiver_amt_ratio = amt_ratio * is_new_receiver
    
    hour_sin = np.sin(2 * np.pi * ts.hour / 24)
    hour_cos = np.cos(2 * np.pi * ts.hour / 24)
    
    return {
        'amt_ratio': amt_ratio,
        'amt_zscore': amt_zscore,
        'txn_count_1h': float(txn_count_1h),
        'secs_since_last': secs_since_last,
        'recipient_repeat_count': recip_count,
        'is_new_receiver': is_new_receiver,
        'device_changed': device_changed,
        'city_changed': city_changed,
        'is_night': is_night,
        'same_receiver_count_24h': float(same_receiver_count_24h),
        'receiver_unique_count_1h': float(receiver_unique_count_1h),
        'new_receiver_amt_ratio': new_receiver_amt_ratio,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos
    }

if __name__ == "__main__":
    import random
    df = pd.read_csv("data/upi_transactions.csv")
    df_feat = build_features_batch(df)
    
    # Save to csv
    df_feat.to_csv("data/features.csv", index=False)
    
    print("--- Feature Summary Stats ---")
    feat_cols = ['amt_ratio', 'amt_zscore', 'txn_count_1h', 'secs_since_last', 'recipient_repeat_count', 
                 'is_new_receiver', 'device_changed', 'city_changed', 'is_night', 
                 'same_receiver_count_24h', 'receiver_unique_count_1h', 'new_receiver_amt_ratio',
                 'hour_sin', 'hour_cos']
    print(df_feat[feat_cols].describe())
    
    print("\n--- Mean of Features: Normal vs Fraud ---")
    print(df_feat.groupby('is_fraud')[feat_cols].mean().T)
    
    # Test batch vs single
    print("\n--- Testing Single vs Batch for 20 Random Txns ---")
    np.random.seed(42)
    random.seed(42)
    
    test_indices = random.sample(range(100, len(df_feat)), 20)
    matches = True
    
    for idx in test_indices:
        txn_row = df_feat.iloc[idx]
        sender = txn_row['sender_id']
        ts = txn_row['timestamp']
        
        hist_df = df[(df['sender_id'] == sender) & (pd.to_datetime(df['timestamp']) < ts)]
        
        single_feats = build_features_single(txn_row.to_dict(), hist_df)
        
        for col in feat_cols:
            batch_val = txn_row[col]
            single_val = single_feats[col]
            if not np.isclose(batch_val, single_val, atol=1e-4):
                print(f"Mismatch at idx {idx} for {col}: batch={batch_val}, single={single_val}")
                matches = False
                
    if matches:
        print("Test passed! Single version matches Batch version perfectly.")
    else:
        print("Test failed. See mismatches above.")
