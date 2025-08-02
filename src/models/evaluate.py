#!/usr/bin/env python3
import argparse
from pathlib import Path
from utils.util import utility as UT
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report
)
import json

def main():
    # Parse arguments (matches your exact command)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .joblib model")
    parser.add_argument("--test-data", required=True, help="Path to test CSV")
    args = parser.parse_args()

    # Load data and model
    model = UT.load_model(args.model)
    df = pd.read_csv(args.test_data)
    
    # Validate columns
    required_cols = {"mog_a", "devicetimestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in test data: {missing}")

    # Prepare data
    X_test = df.drop(columns=["mog_a", "devicetimestamp"])
    y_test = df["mog_a"]

    # Generate predictions
    try:
        preds = model.predict(X_test)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "recall": recall_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True)
    }

    # Print formatted results
    print("\n=== Evaluation Metrics ===")
    for name, val in metrics.items():
        if name not in ["confusion_matrix", "classification_report"]:
            print(f"{name:>12}: {val:.4f}")
    
    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    ))

    # Save full report
    report_path = Path("reports") / "evaluation_metrics.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

if __name__ == "__main__":
    main()