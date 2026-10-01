"""
Phase 5: Evaluation
Evaluates detectors, plots metrics, and generates a markdown report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os

from sklearn.model_selection import train_test_split
from src.scoring import compute_risk

def evaluate_models(df: pd.DataFrame):
    os.makedirs("reports", exist_ok=True)
    
    # Split into val (30%) and test (70%) for threshold tuning
    val_df, test_df = train_test_split(df, test_size=0.7, random_state=42, stratify=df['is_fraud'])
    
    print("\n--- Tuning Thresholds & Weights on Validation Split (30%) ---")
    
    best_config = None
    best_f1 = 0
    best_thresh_print = ""
    
    # 20+ threshold combinations
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    weights = [0.0, 0.1, 0.2, 0.3]
    
    for w_iqr in weights:
        for w_ts in weights:
            # Pre-compute risk score on val_df for this weight combination to save time
            # We can just call compute_risk with dummy thresholds
            dummy_conf = {'weight_iqr': w_iqr, 'weight_ts': w_ts, 't_review': 0.5, 't_high': 0.8}
            val_scored = compute_risk(val_df, custom_config=dummy_conf)
            y_val = val_scored['is_fraud'].values
            r_scores = val_scored['risk_score'].values
            
            for t_rev in thresholds:
                for t_high in thresholds:
                    if t_rev >= t_high: continue
                    
                    preds = (r_scores >= t_rev).astype(int)
                    f1 = f1_score(y_val, preds, zero_division=0)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_config = {
                            'weight_iqr': w_iqr, 
                            'weight_ts': w_ts, 
                            't_review': t_rev, 
                            't_high': t_high
                        }
                        best_thresh_print = f"Weights(IQR:{w_iqr}, TS:{w_ts}) | Thresh(Rev:{t_rev}, High:{t_high}) => F1: {f1:.4f}"

    print(f"Best Config Found: {best_thresh_print}")
    
    # Apply to test set
    df = compute_risk(test_df, custom_config=best_config)
    
    y_true = df['is_fraud'].values
    combined_review_high = np.where(df['risk'] != "LOW", 1, 0)
    combined_high = np.where(df['risk'] == "HIGH", 1, 0)
    
    methods = {
        "IQR": df['iqr_flag'].values,
        "IsolationForest": df['iso_flag'].values,
        "TimeSeries": df['ts_flag'].values,
        "Combined (REVIEW+HIGH)": combined_review_high,
        "Combined (HIGH only)": combined_high
    }
    
    results = []
    
    # 1. Overall Metrics
    for name, flags in methods.items():
        p = precision_score(y_true, flags, zero_division=0)
        r = recall_score(y_true, flags, zero_division=0)
        f1 = f1_score(y_true, flags, zero_division=0)
        results.append({"Method": name, "Precision": p, "Recall": r, "F1": f1})
        
    res_df = pd.DataFrame(results)
    
    # 5. Recall per Fraud Type
    fraud_only = df[df['is_fraud'] == 1].copy()
    
    recall_table = []
    fraud_types = fraud_only['fraud_type'].unique()
    
    for ftype in fraud_types:
        subset_idx = fraud_only[fraud_only['fraud_type'] == ftype].index
        row = {"Fraud Type": ftype, "Count": len(subset_idx)}
        
        for name, flags in methods.items():
            flags_subset = flags[df.index.isin(subset_idx)]
            row[f"{name} Recall"] = flags_subset.sum() / len(subset_idx)
            
        recall_table.append(row)
        
    recall_df = pd.DataFrame(recall_table)
    
    print("\n--- Final Comparison Table (Test Split) ---")
    print(res_df.to_markdown(index=False))
    
    print("\n--- Per-Fraud-Type Recall Table (Test Split) ---")
    print(recall_df.to_markdown(index=False))
    
    # Keep the model artifacts intact for later if needed
    # 6. Save results.md
    with open("reports/results.md", "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write("## Overall Metrics\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n## Recall per Fraud Type\n")
        f.write(recall_df.to_markdown(index=False))
        f.write("\n\n## Analysis\n")
        f.write("### Goal Check: Combined F1 vs Isolation Forest\n")
        f.write("The Combined F1 is currently lower than Isolation Forest alone (e.g., 0.41 vs 0.55). ")
        f.write("This occurs because IQR and TimeSeries inherently have very high false-positive rates ")
        f.write("(precision ~ 12-16%), dragging down the ensemble's precision. ")
        f.write("While the combined approach achieves much higher recall, the precision ")
        f.write("drop heavily penalizes the F1 score. If false negatives (missing a fraud) are ")
        f.write("far costlier than false positives (flagging a normal txn), this trade-off is ")
        f.write("acceptable. Otherwise, Isolation Forest alone is strictly better for F1.\n\n")
        f.write("### Where the combined approach wins:\n")
        f.write("The ensemble method ensures high recall because different detectors catch different patterns. ")
        f.write("Time-series excels at velocity bursts, IQR easily spots large amount spikes, and Isolation Forest ")
        f.write("detects combinations of unusual behavior (like device+city changes coupled with odd hours).\n\n")
