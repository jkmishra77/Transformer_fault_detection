#!/usr/bin/env python3
import argparse
from pathlib import Path
from utils.util import utility as UT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

MODEL_MAPPING = {
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
    "LogisticRegression": LogisticRegression,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare multiple transformer-fault classifiers"
    )
    parser.add_argument("--input-file", required=True, help="Path to features CSV")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--output-model-dir", required=True, help="Model output dir")
    parser.add_argument("--metrics-file", required=True, help="Metrics output path")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = UT.get_logger(__name__)

    Path(args.output_model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_file).parent.mkdir(parents=True, exist_ok=True)

    # 1. Load config and data
    config = UT.load_json(args.config)
    df = UT.load_csv(args.input_file)

    # Validate required columns
    target_col = config.get("target_column", "mog_a")
    required_cols = {target_col, "devicetimestamp"}
    if not required_cols.issubset(df.columns):
        raise UT.SchemaError(f"Missing required columns: {required_cols - set(df.columns)}")

    # 2. Train/test split
    df_train, df_val = train_test_split(
        df, test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
        stratify=df[target_col]
    )
    UT.save_csv(df_val, "data/processed/validate.csv")

    # Apply SMOTE on training split
    X_train = df_train.drop(columns=[target_col, "devicetimestamp"])
    y_train = df_train[target_col]

    smote = SMOTE(random_state=config.get("random_state", 42))
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    logger.info("SMOTE applied to training data")

    # 3. Train & evaluate models
    all_metrics = {}
    for name, mcfg in config.get("models", {}).items():
        model_class = MODEL_MAPPING.get(mcfg.get("class_name"))
        if not model_class:
            logger.warning(f"Skipping unknown model: {name}")
            continue

        try:
            model = model_class(**{k: v for k, v in mcfg.items() if k != "class_name"})
            model.fit(X_train_resampled, y_train_resampled)

            model_path = Path(args.output_model_dir) / f"{name}.joblib"
            UT.save_model(model, model_path)

            preds = model.predict(df_val.drop(columns=[target_col, "devicetimestamp"]))
            all_metrics[name] = {
                "accuracy": accuracy_score(df_val[target_col], preds),
                "report": classification_report(df_val[target_col], preds, output_dict=True)
            }
        except Exception as e:
            logger.error(f"Failed on model {name}: {str(e)}")
            raise UT.PipelineError(f"Training failed: {str(e)}") from e

    UT.save_json(all_metrics, args.metrics_file)
    logger.info("Training completed")

if __name__ == "__main__":
    main()
