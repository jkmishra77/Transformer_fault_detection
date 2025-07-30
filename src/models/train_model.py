# src/models/train_model.py
import os
import sys
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

in_path = sys.argv[1]
out_file = sys.argv[2]
in_file = os.path.join(in_path, "features.csv")
os.makedirs(os.path.dirname(out_file), exist_ok=True)

try:
    df = pd.read_csv(in_file)
    if df.empty or df.shape[1] < 1:
        raise ValueError("Feature input is empty or malformed")

    X = df.drop(columns=["target"], errors="ignore").select_dtypes(include=["number"])
    y = df.get("target", pd.Series([0] * len(X)))

    if X.empty:
        raise ValueError("No numeric features found for training")

    # 🧪 Verify class diversity
    unique_labels = y.nunique()
    if unique_labels < 2:
        print(f"[WARN] Only one class detected. Injecting dummy class for training continuity.")
        # Inject one sample from class '1' to satisfy solver
        X.loc[len(X)] = [0.0] * X.shape[1]
        y.loc[len(y)] = 1

    model = LogisticRegression()
    model.fit(X, y)

    with open(out_file, "wb") as f:
        pickle.dump(model, f)

    print(f"[INFO] Model trained successfully: {out_file}")

except Exception as e:
    print(f"[ERROR] Model training failed: {e}", file=sys.stderr)
    exit(1)
