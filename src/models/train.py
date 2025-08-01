#!/usr/bin/env python3
import argparse
import logging
import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib


os.makedirs(args.output_model_dir, exist_ok=True)


MODEL_MAPPING = {
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
    "LogisticRegression": LogisticRegression,
}


def setup_logger():
    logger = logging.getLogger("train_compare")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare multiple transformer-fault classifiers"
    )
    parser.add_argument(
        "--input-file", dest="input_file", required=True,
        help="Path to raw features CSV"
    )
    parser.add_argument(
        "--config", dest="config_file", required=True,
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--output-model-dir", dest="model_dir", required=True,
        help="Directory to save trained model files"
    )
    parser.add_argument(
        "--metrics-file", dest="metrics_file", required=True,
        help="Path to write combined metrics JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    # 1. Load config
    logger.info(f"Loading config from {args.config_file}")
    with open(args.config_file, "r") as cf:
        config = json.load(cf)

    target_col = config.get("target_column", "mog_a")
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    models_cfg = config.get("models", {})

    if not models_cfg:
        logger.error("No models defined in config['models']")
        return

    # 2. Load data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in input file")

    if "devicetimestamp" not in df.columns:
        raise KeyError("Column 'devicetimestamp' not found in input file")

    # 3. Split off validation set (20%)
    logger.info("Splitting off validation set (20%)")
    df_train_full, df_validate = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    os.makedirs("data/processed", exist_ok=True)
    validate_path = os.path.join("data/processed", "validate.csv")
    logger.info(f"Writing validation split to {validate_path}")
    df_validate.to_csv(validate_path, index=False)

    # 4. Prepare features/labels
    logger.info(f"Dropping columns: '{target_col}', 'devicetimestamp'")
    y_full = df_train_full[target_col]
    X_full = df_train_full.drop(columns=[target_col, "devicetimestamp"])

    logger.info("Splitting train/test on remaining 80%")
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_full,
    )

    # Ensure output dirs exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)

    all_metrics = {}

    # 5. Train & evaluate each model
    for name, mcfg in models_cfg.items():
        class_name = mcfg.get("class_name")

        if class_name not in MODEL_MAPPING:
            logger.warning(f"Skipping '{name}': unknown class '{class_name}'")
            continue

        # Extract all parameters except 'class_name'
        params = {k: v for k, v in mcfg.items() if k != "class_name"}

        logger.info(f"Training '{name}' ({class_name}) with params {params}")
        ModelClass = MODEL_MAPPING[class_name]

        try:
            model = ModelClass(**params)
        except TypeError as e:
            logger.error(f"Failed to initialize model '{name}': {e}")
            continue

        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            continue

        # Save model
        model_path = os.path.join(args.model_dir, f"{name}.joblib")
        logger.info(f"Saving model '{name}' to {model_path}")
        joblib.dump(model, model_path)

        # Evaluate
        logger.info(f"Evaluating '{name}' on test set")
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        all_metrics[name] = {
            "accuracy": acc,
            "classification_report": report
        }

    # 6. Write combined metrics
    logger.info(f"Writing all metrics to {args.metrics_file}")
    with open(args.metrics_file, "w") as mf:
        json.dump(all_metrics, mf, indent=2)

    logger.info("All training rounds complete")


if __name__ == "__main__":
    main()
