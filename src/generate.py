"""
Phase 1: Synthetic data generation for UPI transactions.
Generates 10,000 transactions over 30 days.
"""
import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
import os

def generate_synthetic_data(output_path: str = "data/upi_transactions.csv"):
    """
    Generates synthetic UPI transaction data.
    """
    np.random.seed(42)
    random.seed(42)

    num_users = 200
    num_normal = 9500
    num_fraud = 500
    days = 30
    start_date = datetime(2023, 10, 1)

    cities = ["Lucknow", "Delhi", "Mumbai", "Pune", "Bengaluru"]
    devices = ["android", "ios"]
    channels = ["PhonePe", "GPay", "Paytm", "BHIM", "AmazonPay"]
    merchants = ["groceries", "utilities", "food", "travel", "entertainment", "retail", "none"]

    # 1. Create User Profiles
    user_profiles = {}
    for i in range(num_users):
        user_id = f"U{i:04d}"
        avg_amt = np.random.lognormal(mean=6.5, sigma=0.5) # ~665 typical
        home_city = random.choice(cities)
        device = random.choice(devices)
        active_hours = list(range(7, 23)) # 7 AM to 10 PM
        receivers = [f"R{random.randint(1, 1000):04d}" for _ in range(random.randint(5, 10))]
        channel = random.choice(channels)
        
        user_profiles[user_id] = {
            "avg_amt": avg_amt,
            "city": home_city,
            "device": device,
            "active_hours": active_hours,
            "receivers": receivers,
            "channel": channel
        }

    transactions = []

    def make_txn(sender_id, timestamp, amount, receiver_id, city, device, is_fraud, fraud_type):
        txn_type = "P2M" if receiver_id.startswith("M") or random.random() < 0.3 else "P2P"
        merchant = random.choice(merchants[:-1]) if txn_type == "P2M" else "none"
        
        return {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "amount": round(amount, 2),
            "merchant_category": merchant,
            "transaction_type": txn_type,
            "city": city,
            "device_type": device,
            "upi_channel": user_profiles[sender_id]["channel"],
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "is_fraud": 1 if is_fraud else 0,
            "fraud_type": fraud_type
        }

    # 2. Generate Normal Transactions
    for _ in range(num_normal):
        sender = random.choice(list(user_profiles.keys()))
        prof = user_profiles[sender]
        
        # Pick timestamp
        day_offset = random.randint(0, days - 1)
        hour = random.choice(prof["active_hours"])
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts = start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
        
        # Amount: normal around average, prevent negative
        amt = max(10, np.random.normal(prof["avg_amt"], prof["avg_amt"] * 0.3))
        
        # Add some noise (1% chance of slightly unusual)
        if random.random() < 0.01:
            amt = amt * random.uniform(2, 5) # Festival day large amount
        
        if random.random() < 0.01:
            hour = random.choice([0, 1, 2, 23]) # Occasional late night
            ts = ts.replace(hour=hour)
        
        receiver = random.choice(prof["receivers"])
        city = prof["city"] if random.random() < 0.95 else random.choice(cities) # Rarely travel
        device = prof["device"]
        
        transactions.append(make_txn(sender, ts, amt, receiver, city, device, False, "none"))

    # 3. Generate Fraud Transactions
    # 5 patterns, 100 each
    
    # 3.1 Amount Spike (100)
    for _ in range(100):
        sender = random.choice(list(user_profiles.keys()))
        prof = user_profiles[sender]
        ts = start_date + timedelta(days=random.randint(0, days-1), hours=random.choice(prof["active_hours"]), minutes=random.randint(0, 59))
        amt = prof["avg_amt"] * random.uniform(10, 50)
        receiver = f"F_R_{random.randint(1000,9999)}"
        transactions.append(make_txn(sender, ts, amt, receiver, prof["city"], prof["device"], True, "amount_spike"))
        
    # 3.2 Velocity Burst (100 txns)
    bursts_needed = 100
    while bursts_needed > 0:
        sender = random.choice(list(user_profiles.keys()))
        prof = user_profiles[sender]
        burst_size = min(random.randint(5, 10), bursts_needed)
        ts = start_date + timedelta(days=random.randint(0, days-1), hours=random.choice(prof["active_hours"]), minutes=random.randint(0, 50))
        receiver = f"F_R_{random.randint(1000,9999)}"
        for _ in range(burst_size):
            amt = max(10, np.random.normal(prof["avg_amt"], prof["avg_amt"] * 0.1))
            transactions.append(make_txn(sender, ts, amt, receiver, prof["city"], prof["device"], True, "velocity_burst"))
            ts += timedelta(seconds=random.randint(10, 60)) # Within a minute
        bursts_needed -= burst_size
        
    # 3.3 Odd Hours (100)
    for _ in range(100):
        sender = random.choice(list(user_profiles.keys()))
        prof = user_profiles[sender]
        ts = start_date + timedelta(days=random.randint(0, days-1), hours=random.randint(2, 4), minutes=random.randint(0, 59))
        amt = max(10, np.random.normal(prof["avg_amt"], prof["avg_amt"] * 0.3))
        receiver = f"F_R_{random.randint(1000,9999)}"
        transactions.append(make_txn(sender, ts, amt, receiver, prof["city"], prof["device"], True, "odd_hours"))
        
    # 3.4 Device + City Change (100)
    for _ in range(100):
        sender = random.choice(list(user_profiles.keys()))
        prof = user_profiles[sender]
        ts = start_date + timedelta(days=random.randint(0, days-1), hours=random.choice(prof["active_hours"]), minutes=random.randint(0, 59))
        amt = max(10, np.random.normal(prof["avg_amt"], prof["avg_amt"] * 0.3))
        receiver = f"F_R_{random.randint(1000,9999)}"
        new_city = random.choice([c for c in cities if c != prof["city"]])
        new_device = "ios" if prof["device"] == "android" else "android"
        transactions.append(make_txn(sender, ts, amt, receiver, new_city, new_device, True, "device_city_change"))

    # 3.5 Repeated transfers to a brand-new receiver (100 txns)
    rep_needed = 100
    while rep_needed > 0:
        sender = random.choice(list(user_profiles.keys()))
        prof = user_profiles[sender]
        rep_size = min(4, rep_needed)
        ts = start_date + timedelta(days=random.randint(0, days-1), hours=random.choice(prof["active_hours"]), minutes=random.randint(0, 50))
        receiver = f"NEW_R_{random.randint(1000,9999)}"
        for _ in range(rep_size):
            amt = max(10, np.random.normal(prof["avg_amt"], prof["avg_amt"] * 0.3))
            transactions.append(make_txn(sender, ts, amt, receiver, prof["city"], prof["device"], True, "new_receiver_repeated"))
            ts += timedelta(hours=random.randint(1, 10))
        rep_needed -= rep_size
        
    # Shuffle and sort by time
    df = pd.DataFrame(transactions)
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} transactions.")
    print("\nFraud Types summary:")
    print(df['fraud_type'].value_counts())
    print(f"\nTotal Fraud %: {df['is_fraud'].mean()*100:.2f}%")

if __name__ == "__main__":
    generate_synthetic_data()
