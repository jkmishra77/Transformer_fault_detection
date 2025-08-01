#!/usr/bin/env python3
import argparse
import pandas as pd
from utils.util import utility as UT


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic feature engineering placeholder"""
    df = df.copy()
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["datetime"].dt.hour
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df.drop(columns=["datetime"], inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Feature engineering script")
    parser.add_argument("--input-file", required=True, help="Path to input CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to save output")
    parser.add_argument("--manifest", required=True, help="Path to output manifest file")

    args = parser.parse_args()

    logger = UT.get_logger(__name__)
    logger.info(f"Loading dataset from {args.input_file}")

    df = UT.load_csv(args.input_file)
    processed_df = build_features(df)

    # Save feature CSV
    feature_path = f"{args.output_dir}/features.csv"
    UT.save_csv(processed_df, feature_path)

    # Save manifest
    manifest = {
        "n_rows": processed_df.shape[0],
        "n_columns": processed_df.shape[1],
        "columns": list(processed_df.columns),
    }
    UT.save_json(manifest, args.manifest)
    logger.info(f"Saved features to {feature_path} and manifest to {args.manifest}")


if __name__ == "__main__":
    main()
