import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import logging
import click
import json
from src.config.schema import TrainConfig

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> TrainConfig:
    with open(config_path) as f:
        raw = json.load(f)
    return TrainConfig(**raw)

def prepare_data(df, target_column, test_size, random_state):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@click.command()
@click.option('--data', type=click.Path(exists=True), required=True)
@click.option('--config', type=click.Path(exists=True), required=True)
@click.option('--model-out', type=click.Path(), default="models/model.pkl")
@click.option('--metrics-out', type=click.Path(), default="reports/train_metrics.json")
def train(data, config, model_out, metrics_out):
    cfg: TrainConfig = load_config(config)

    logger.info("Loading dataset...")
    df = pd.read_csv(data)
    print(df.columns.tolist())

    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        cfg.target_column,
        cfg.test_size,
        cfg.random_state
    )

    logger.info("Training model...")
    model = RandomForestClassifier(**cfg.model_params)
    model.fit(X_train, y_train)

    logger.info("Evaluating model...")
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=False)

    with open(metrics_out, "w") as f:
        f.write(report)

    joblib.dump(model, model_out)
    logger.info(f"Model saved to {model_out}")
    logger.info(f"Metrics saved to {metrics_out}")

if __name__ == "__main__":
    train()
