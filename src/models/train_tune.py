#!/usr/bin/env python3
import argparse
from pathlib import Path
from utils.util import utility as UT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import mlflow
from mlflow.models.signature import infer_signature

MODEL_MAPPING = {
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
    "LogisticRegression": LogisticRegression,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and tune transformer fault prediction models"
    )
    parser.add_argument("--input-file", required=True, help="Path to features CSV")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--output-model-dir", required=True, help="Model output dir")
    parser.add_argument("--metrics-file", required=True, help="Metrics output path")
    return parser.parse_args()

def get_param_grid(model_name):
    """Optimized parameter grids"""
    if model_name == "RandomForestClassifier":
        return {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 10],
            'classifier__min_samples_split': [2, 5]
        }
    elif model_name == "SVC":
        return {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf'],
            'classifier__gamma': ['scale']
        }
    elif model_name == "LogisticRegression":
        return {
            'classifier__C': np.logspace(-3, 3, 5),
            'classifier__penalty': ['l2'],
            'classifier__solver': ['liblinear']
        }

def main():
    args = parse_args()
    logger = UT.get_logger(__name__)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Transformer_Fault_Tuning")

    # Load config and data
    config = UT.load_json(args.config)
    df = UT.load_csv(args.input_file)
    target_col = config["target_column"]

    with mlflow.start_run() as parent_run:
        # Data splitting
        df_train, df_val = train_test_split(
            df, 
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=df[target_col]
        )
        UT.save_csv(df_val, "data/processed/validate.csv")

        # Prepare features
        X_train = df_train.drop(columns=[target_col, "devicetimestamp"])
        y_train = df_train[target_col]
        X_val = df_val.drop(columns=[target_col, "devicetimestamp"])
        y_val = df_val[target_col]

        all_metrics = {}
        for name, mcfg in config["models"].items():
            model_name = mcfg["class_name"]
            with mlflow.start_run(nested=True, run_name=name) as child_run:
                # Create pipeline
                pipe = Pipeline([
                    ('smote', SMOTE(random_state=config["random_state"])),
                    ('classifier', MODEL_MAPPING[model_name]())
                ])

                # Configure GridSearch
                gs = GridSearchCV(
                    pipe,
                    get_param_grid(model_name),
                    scoring={
                        'f1_weighted': make_scorer(f1_score, average='weighted'),
                        'recall': make_scorer(recall_score),
                        'accuracy': make_scorer(accuracy_score)
                    },
                    refit='f1_weighted',
                    cv=5,
                    n_jobs=-1,
                    verbose=3
                )

                logger.info(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
                gs.fit(X_train, y_train)

                # Generate predictions
                preds = gs.predict(X_val)
                
                # Store metrics
                all_metrics[name] = {
                    "best_params": gs.best_params_,
                    "best_score": gs.best_score_,
                    "validation_metrics": {
                        "accuracy": accuracy_score(y_val, preds),
                        "f1_weighted": f1_score(y_val, preds, average='weighted'),
                        "recall": recall_score(y_val, preds)
                    },
                    "run_id": child_run.info.run_id  # Track MLflow run
                }

                # Log to MLflow
                mlflow.log_params(gs.best_params_)
                mlflow.log_metrics(all_metrics[name]["validation_metrics"])
                
                # Save model
                model_path = Path(args.output_model_dir) / f"{name}_best.joblib"
                UT.save_model(gs.best_estimator_, model_path)
                mlflow.sklearn.log_model(
                    sk_model=gs.best_estimator_,
                    artifact_path=name,
                    registered_model_name=f"tf_{name}"
                )

        # Auto-select best model
        best_model_name, best_metrics = max(
            all_metrics.items(),
            key=lambda x: x[1]["validation_metrics"]["f1_weighted"]
        )
        best_model_path = Path(args.output_model_dir) / f"{best_model_name}_best.joblib"
        
        # Tag and log best model
        mlflow.set_tag("best_model", best_model_name)
        mlflow.log_metric("best_model_f1", best_metrics["validation_metrics"]["f1_weighted"])
        
        # Save best model copy
        UT.save_model(
            UT.load_model(best_model_path),
            Path(args.output_model_dir) / "best_model.joblib"
        )
        
        logger.info(f"\n{'='*50}")
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info(f"F1 Score: {best_metrics['validation_metrics']['f1_weighted']:.3f}")
        logger.info(f"Saved to: models/best_model.joblib")
        logger.info(f"{'='*50}")

        # Save metrics
        UT.save_json(all_metrics, args.metrics_file)
        mlflow.log_artifact(args.metrics_file)

if __name__ == "__main__":
    main()