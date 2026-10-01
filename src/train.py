"""
End-to-end training pipeline.
"""
from src.generate import generate_synthetic_data
from src.features import build_features_batch
from src.detectors import train_iqr, iqr_flag, train_isolation_forest, apply_isolation_forest, apply_time_series
from src.scoring import compute_risk
from src.evaluate import evaluate_models
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    print("1. Generating synthetic data...")
    generate_synthetic_data("data/upi_transactions.csv")
    
    print("\n2. Building features...")
    df = pd.read_csv("data/upi_transactions.csv")
    df_feat = build_features_batch(df)
    
    print("\n3. Training detectors...")
    bounds = train_iqr(df_feat)
    train_isolation_forest(df_feat)
    
    print("\n4. Applying detectors...")
    df_feat['iqr_flag'] = iqr_flag(df_feat, bounds)
    df_feat['iso_flag'], df_feat['iso_score'] = apply_isolation_forest(df_feat)
    df_feat['ts_flag'], _, _ = apply_time_series(df_feat)
    
    print("\n5. Computing combined risk scores...")
    df_scored = compute_risk(df_feat)
    
    print("\n6. Evaluating...")
    evaluate_models(df_scored)
    
    print("\nPipeline completed successfully! Models saved to models/ and reports saved to reports/.")

if __name__ == "__main__":
    main()
