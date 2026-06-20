"""
train.py - Entry point to generate data, preprocess, train and evaluate all models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator import generate_network_data
from src.preprocessor   import preprocess
from src.model          import train_all

FEATURES_EXTENDED = [
    'duration', 'bytes_sent', 'bytes_received', 'packets_sent',
    'packets_received', 'port', 'protocol', 'failed_logins',
    'num_connections', 'bytes_ratio', 'packet_ratio', 'bytes_per_second'
]

if __name__ == '__main__':
    print("=" * 60)
    print("  Network Anomaly Detection — Model Training")
    print("=" * 60)

    # 1. Generate / load data
    csv_path = 'data/network_traffic.csv'
    print("\n[1/3] Generating synthetic network traffic data …")
    df = generate_network_data(n_normal=2000, n_anomaly=400, save_path=csv_path)
    print(f"      Dataset: {len(df)} records  |  Anomaly rate: {df['label'].mean()*100:.1f}%")

    # 2. Preprocess
    print("\n[2/3] Preprocessing and feature engineering …")
    X, y, _ = preprocess(df, fit_scaler=True)
    print(f"      Feature matrix: {X.shape}")

    # 3. Train + evaluate
    print("\n[3/3] Training models …")
    results = train_all(X, y, FEATURES_EXTENDED)

    # Summary
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n  {model_name}")
        for k, v in metrics.items():
            print(f"    {k:<20} {v}")

    print("\n✅  Training complete. Models saved to /models/")
    print("    Run `python app.py` to start the web dashboard.\n")
