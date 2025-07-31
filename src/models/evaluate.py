# src/models/evaluate_model.py
import os
import sys
import pickle
import pandas as pd
import json
from sklearn.metrics import accuracy_score

if len(sys.argv) != 4:
    print(f"[ERROR] Expected 3 arguments, got {len(sys.argv)}", file=sys.stderr)
    exit(1)

model_path, features_path, metrics_path = sys.argv[1:]

print(f"[INFO] Evaluating model at: {model_path}")
print(f"[INFO] Loading features from: {features_path}/features.csv")
print(f"[INFO] Saving metrics to: {metrics_path}")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df_path = os.path.join(features_path, "features.csv")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"features.csv not found at {df_path}")

    df = pd.read_csv(df_path)
    if "target" not in df.columns:
        raise ValueError("Missing 'target' column in features")

    X = df.drop(columns=["target"], errors="ignore").select_dtypes(include=["number"])
    y = df["target"]
    preds = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, preds)
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Evaluation successful. Metrics: {metrics}")

except Exception as e:
    print(f"[ERROR] Evaluation failed: {e}", file=sys.stderr)
    exit(1)
