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
    if df.empty:
        raise ValueError("Feature input is empty")

    X = df.drop(columns=["target"], errors="ignore")
    y = df.get("target", pd.Series([0] * len(X)))

    model = LogisticRegression()
    model.fit(X, y)

    with open(out_file, "wb") as f:
        pickle.dump(model, f)

except Exception as e:
    print(f"[ERROR] Model training failed: {e}", file=sys.stderr)
    exit(1)
