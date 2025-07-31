# src/models/predict.py

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained model and generate predictions on feature data."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the serialized model file",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="CSV of processed features",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Destination CSV for predictions",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    logger.info(f"Loading model from {args.model}")
    model = joblib.load(args.model)

    logger.info(f"Reading features from {args.input_file}")
    df = pd.read_csv(args.input_file)

    if "devicetimestamp" in df.columns:
        df = df.drop(columns=["devicetimestamp"])
    else:
        raise KeyError("Column 'devicetimestamp' not found in input data")
    target_col = "mog_a"   # or whatever you named the label in train.py
    if target_col in df.columns:
        df = df.drop(columns=[target_col])



    logger.info("Generating predictions")
    preds = model.predict(df.values)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving predictions to {out_path}")
    pd.DataFrame({"prediction": preds}).to_csv(out_path, index=False)

    logger.info("Prediction step completed")


if __name__ == "__main__":
    main()
