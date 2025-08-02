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
    
    # Create directories using utility-compatible method
    Path(args.output_model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_file).parent.mkdir(parents=True, exist_ok=True)

    # 1. Load config and data
    config = UT.load_json(args.config)
    df = UT.load_csv(args.input_file)

    # Validate columns
    target_col = config.get("target_column", "mog_a")
    required_cols = {target_col, "devicetimestamp"}
    if not required_cols.issubset(df.columns):
        raise UT.SchemaError(f"Missing required columns: {required_cols - set(df.columns)}")

    # 2. Data splitting
    df_train, df_val = train_test_split(
        df, test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
        stratify=df[target_col]
    )
    UT.save_csv(df_val, "data/processed/validate.csv")

    # 3. Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(
        df_train.drop(columns=[target_col, "devicetimestamp"]),
        df_train[target_col]
    )
    logger.info(f"Resampled training data. Class distribution: {pd.Series(y_train_res).value_counts().to_dict()}")

    # 4. Model training
    all_metrics = {}
    for name, mcfg in config.get("models", {}).items():
        model_class = MODEL_MAPPING.get(mcfg.get("class_name"))
        if not model_class:
            logger.warning(f"Skipping unknown model: {name}")
            continue

        try:
            # Initialize model with all config params except 'class_name'
            model = model_class(**{k:v for k,v in mcfg.items() if k != "class_name"})
            
            # Train on resampled data
            model.fit(X_train_res, y_train_res)
            
            # Save model
            model_path = Path(args.output_model_dir) / f"{name}.joblib"
            UT.save_model(model, model_path)
            
            # Evaluate on original validation data (no resampling)
            preds = model.predict(df_val.drop(columns=[target_col, "devicetimestamp"]))
            all_metrics[name] = {
                "accuracy": accuracy_score(df_val[target_col], preds),
                "report": classification_report(df_val[target_col], preds, output_dict=True),
                "class_distribution": {
                    "train_original": df_train[target_col].value_counts().to_dict(),
                    "train_resampled": pd.Series(y_train_res).value_counts().to_dict(),
                    "validation": df_val[target_col].value_counts().to_dict()
                }
            }
            logger.info(f"Trained {name} | Validation accuracy: {all_metrics[name]['accuracy']:.3f}")

        except Exception as e:
            logger.error(f"Failed on model {name}: {str(e)}")
            raise UT.PipelineError(f"Training failed: {str(e)}") from e

    UT.save_json(all_metrics, args.metrics_file)
    logger.info(f"Training completed. Metrics saved to {args.metrics_file}")

if __name__ == "__main__":
    main()