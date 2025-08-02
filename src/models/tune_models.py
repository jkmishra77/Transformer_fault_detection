#!/usr/bin/env python3
import argparse
from pathlib import Path
import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from utils.util import utility as UT
from train import MODEL_MAPPING

def parse_args():
    parser = argparse.ArgumentParser(description="Tune models via GridSearchCV and log with MLflow")
    parser.add_argument("--input-file", required=True, help="Path to input feature CSV")
    parser.add_argument("--config", required=True, help="Path to params.json")
    parser.add_argument("--tune-config", required=True, help="Path to tune_config.json")
    parser.add_argument("--output-model-dir", required=True, help="Directory to save best models")
    parser.add_argument("--metrics-file", required=True, help="Path to save tuning metrics")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = UT.get_logger(__name__)
    
    # Load configs and data
    config = UT.load_json(args.config)
    tune_grid = UT.load_json(args.tune_config)
    df = UT.load_csv(args.input_file)
    target_col = config.get("target_column", "mog_a")

    # Split and resample training data
    df_train, df_val = UT.split_train_val(df, config)
    smote = SMOTE(random_state=config.get("random_state", 42))
    X_train, y_train = smote.fit_resample(
        df_train.drop(columns=[target_col, "devicetimestamp"]),
        df_train[target_col]
    )
    Path(args.output_model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_file).parent.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for name, mcfg in config.get("models", {}).items():
        if name not in tune_grid:
            logger.warning(f"No tuning grid for model '{name}', skipping...")
            continue

        model_class = MODEL_MAPPING.get(mcfg["class_name"])
        if not model_class:
            logger.warning(f"Unknown class for model '{name}', skipping...")
            continue

        logger.info(f"Starting tuning for {name}")
        clf = GridSearchCV(
            estimator=model_class(),
            param_grid=tune_grid[name],
            scoring="accuracy",
            cv=3,
            verbose=0
        )

        with mlflow.start_run(run_name=f"tune_{name}"):
            clf.fit(X_train, y_train)

            best_model = clf.best_estimator_
            preds = best_model.predict(df_val.drop(columns=[target_col, "devicetimestamp"]))
            acc = accuracy_score(df_val[target_col], preds)
            report = classification_report(df_val[target_col], preds, output_dict=True)

            # Save model + log everything
            model_path = Path(args.output_model_dir) / f"{name}_best.joblib"
            UT.save_model(best_model, model_path)

            mlflow.log_params(clf.best_params_)
            mlflow.log_param("model_name", name)
            mlflow.log_param("target_column", target_col)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_dict(report, f"{name}_tuned_report.json")
            mlflow.log_artifact(str(model_path))

            all_metrics[name] = {
                "best_params": clf.best_params_,
                "accuracy": acc,
                "report": report
            }
            logger.info(f"Tuned {name} | Accuracy: {acc:.3f}")

    UT.save_json(all_metrics, args.metrics_file)
    logger.info(f"Tuning complete. Metrics saved to {args.metrics_file}")

if __name__ == "__main__":
    main()
