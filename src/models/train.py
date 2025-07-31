#!/usr/bin/env python3
import argparse
import logging
import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def setup_logger():
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train transformer-fault classifier"
    )
    parser.add_argument(
        "--input-file", dest="input_file", required=True,
        help="Path to features CSV"
    )
    parser.add_argument(
        "--config", dest="config_file", required=True,
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--output-model", dest="output_model", required=True,
        help="Path to save trained model (joblib)"
    )
    parser.add_argument(
        "--metrics-file", dest="metrics_file", required=True,
        help="Path to write metrics JSON"
    )
    parser.add_argument(
        "--random-state", dest="random_state", type=int,
        help="Optional override for random seed"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    # 1. load config
    logger.info(f"Loading config from {args.config_file}")
    with open(args.config_file, "r") as cf:
        config = json.load(cf)

    target_col = config.get("target_column", "target")
    test_size = config.get("test_size", 0.2)
    random_state = (
        args.random_state
        if args.random_state is not None
        else config.get("random_state", 42)
    )

    # 2. load data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)

    # 3. extract target & drop timestamp
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    if "devicetimestamp" not in df.columns:
        raise KeyError("Column 'devicetimestamp' not found in data")

    logger.info(f"Dropping columns: '{target_col}', 'devicetimestamp'")
    y = df[target_col]
    X = df.drop(columns=[target_col, "devicetimestamp"])

    # 4. split
    logger.info("Splitting train/test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 5. train
    logger.info("Training RandomForestClassifier")
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # 6. save model
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    logger.info(f"Saving model to {args.output_model}")
    joblib.dump(model, args.output_model)

    # 7. evaluate
    logger.info("Evaluating on test set")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "accuracy": acc,
        "classification_report": report
    }

    # 8. write metrics
    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
    logger.info(f"Writing metrics to {args.metrics_file}")
    with open(args.metrics_file, "w") as mf:
        json.dump(metrics, mf, indent=2)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
