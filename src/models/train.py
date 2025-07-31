#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                            confusion_matrix, f1_score, precision_score,
                            recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def setup_logging():
    """Configure logging with file and console output."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler("logs/training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_and_validate_config(config_path):
    """Load and validate configuration file."""
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Validate required sections
        if "data_settings" not in config:
            raise ValueError("Missing required section: data_settings")
            
        if "models" not in config or not config["models"]:
            raise ValueError("Missing or empty models section")
            
        # Validate data settings
        if "target_column" not in config["data_settings"]:
            raise ValueError("Missing target_column in data_settings")
            
        return config
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in config file")
    except Exception as e:
        raise ValueError(f"Config error: {str(e)}")

def initialize_model(model_config):
    """Initialize model from configuration."""
    try:
        module_path, class_name = model_config["class_name"].rsplit(".", 1)
        module = import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class(**{k: v for k, v in model_config.items() if k != "class_name"})
    except Exception as e:
        raise ValueError(f"Model initialization failed: {str(e)}")

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }
    
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception as e:
            logging.warning(f"ROC AUC calculation skipped: {str(e)}")
    
    return metrics

def main():
    logger = setup_logging()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train machine learning models")
    parser.add_argument("--input-file", required=True, help="Path to input features CSV")
    parser.add_argument("--config", required=True, help="Path to configuration JSON")
    parser.add_argument("--output-model-dir", required=True, help="Directory to save models")
    parser.add_argument("--metrics-file", required=True, help="Path to save metrics JSON")
    args = parser.parse_args()
    
    try:
        # Ensure directories exist
        Path(args.output_model_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(args.metrics_file)).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Load and validate config
        config = load_and_validate_config(args.config)
        logger.info("Configuration validated successfully")
        
        # Load data
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Prepare features and target
        target_col = config["data_settings"]["target_column"]
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split data
        test_size = config["data_settings"].get("test_size", 0.2)
        random_state = config["data_settings"].get("random_state", 42)
        stratify = y if config["data_settings"].get("stratify", True) else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
        
        # Train models
        all_metrics = {}
        scale_features = config.get("feature_settings", {}).get("scale_features", False)
        
        for model_name, model_config in config["models"].items():
            logger.info(f"\n{'='*40}")
            logger.info(f"Training model: {model_name}")
            logger.info(f"{'='*40}")
            
            try:
                # Initialize model
                model = initialize_model(model_config)
                
                # Create pipeline if scaling enabled
                if scale_features:
                    model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                
                # Train model
                start_time = datetime.now()
                model.fit(X_train, y_train)
                train_time = (datetime.now() - start_time).total_seconds()
                
                # Evaluate
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
                
                metrics = calculate_metrics(y_val, y_pred, y_proba)
                metrics["training_time_seconds"] = train_time
                
                # Save model
                model_path = os.path.join(args.output_model_dir, f"{model_name}.joblib")
                joblib.dump(model, model_path)
                logger.info(f"Saved model to {model_path}")
                
                all_metrics[model_name] = metrics
                logger.info(f"Validation accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        # Save metrics
        with open(args.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"\nTraining complete. Metrics saved to {args.metrics_file}")
        
    except Exception as e:
        logger.error(f"Fatal error in training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()